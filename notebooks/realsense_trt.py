import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import time
from PIL.Image import Image
from sam2.utils.transforms import SAM2Transforms
from infer_trt import TRTWrapper

# --- CONFIGURATION ---
# Replace these with your actual engine paths/objects
ENCODER_PATH = "sam2_encoder.engine"
DECODER_PATH = "sam2_decoder.engine"
MAX_MEMORY_HISTORY = 150  # Number of frames to keep in tracking memory

class OnlineTRTSAM2:
    """Handles the streaming logic using TensorRT engines."""
    def __init__(self, trt_encoder, trt_decoder, history_limit=MAX_MEMORY_HISTORY, device="cuda"):
        self.encoder = trt_encoder
        self.decoder = trt_decoder
        self.history_limit = history_limit
        self.device = device
        self.mask_threshold = 0.0
        self._transforms = SAM2Transforms(
            resolution=1024,
            mask_threshold=self.mask_threshold,
            max_hole_area=0.0,
            max_sprinkle_area=0.0,
        )

        # Internal State
        self.frame_idx = 0
        self.memory_bank = {}      # { frame_idx: { obj_id: mask_logits } }
        self.features_cache = {}   # { frame_idx: encoder_features }

    def init_session(self):
        self.frame_idx = 0
        self.memory_bank = {}
        self.features_cache = {}

    def step(self, image, interactions, active_objects):

        idx = self.frame_idx
        if isinstance(image, np.ndarray):
            self._orig_hw = [image.shape[:2]]
        elif isinstance(image, Image):
            w, h = image.size
            self._orig_hw = [(h, w)]
        else:
            raise NotImplementedError("Image format not supported")

        input_image = self._transforms(image)
        input_image = input_image[None, ...].to(self.device)

        # TensorRT Encoder Inference
        features = self.encoder.infer({"image": input_image})
        self.features_cache[idx] = features

        final_results = []

        # 2. Process each active object
        for obj_id in active_objects:
            decoder_inputs = {
                "image_embed": features["image_embed"],
                "high_res_feat_0": features["high_res_feat_0"],
                "high_res_feat_1": features["high_res_feat_1"],
            }

            # Case A: User is interacting (Clicking)
            if interactions and obj_id in interactions:
                pts = np.array(interactions[obj_id]['points'], dtype=np.float32)
                lbls = np.array(interactions[obj_id]['labels'], dtype=np.int32)
                # print(f"Click: ", pts[:5])

                mask_input, point_coords, point_labels, boxes = self._prep_prompts(
                    point_coords=pts,
                    point_labels=lbls,
                    box=None,
                    mask_logits=None,
                    normalize_coords=True
                    # img_idx=idx
                )
                
                if point_coords is not None:
                    concat_points = (point_coords, point_labels)
                else:
                    concat_points = None

                # Embed prompts
                if boxes is not None:
                    box_coords = boxes.reshape(-1, 2, 2)
                    box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
                    box_labels = box_labels.repeat(boxes.size(0), 1)
                    # we merge "boxes" and "points" into a single "concat_points" input (where
                    # boxes are added at the beginning) to sam_prompt_encoder
                    if concat_points is not None:
                        concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                        concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                        concat_points = (concat_coords, concat_labels)
                    else:
                        concat_points = (box_coords, box_labels)

                # Predict masks
                batched_mode = (
                    concat_points is not None and concat_points[0].shape[0] > 1
                )  # multi object prediction

                # print(f"Scaled Post: ", point_coords[:5])
                decoder_inputs["point_coords"] = point_coords
                decoder_inputs["point_labels"] = point_labels
            
            # Case B: Tracking from previous frame memory
            elif idx > 0 and obj_id in self.memory_bank.get(idx-1, {}):
                decoder_inputs["mask_input"] = self.memory_bank[idx-1][obj_id]
                # SAM2 TRT decoder usually requires a dummy point if no clicks exist
                decoder_inputs["point_coords"] = torch.zeros((1, 1, 2), device=self.device)
                decoder_inputs["point_labels"] = torch.tensor([[-1]], dtype=torch.int32, device=self.device)
            
            else:
                continue # No prompt available for this object yet

            # Run Decoder Inference
            outputs = self.decoder.infer(decoder_inputs)
            low_res_masks = outputs["masks"][:, 0:1, :, :] # Take highest score mask
            # print(f"Mask Min: {low_res_masks.min()}, Max: {low_res_masks.max()}")
            # Store for next frame propagation
            if idx not in self.memory_bank: self.memory_bank[idx] = {}
            self.memory_bank[idx][obj_id] = low_res_masks

            # Resize mask back to original camera resolution for display

            masks = self._transforms.postprocess_masks(
                low_res_masks, self._orig_hw[-1]
            )
            low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
            masks = masks > self.mask_threshold
            # print(masks.shape)
            final_results.append((obj_id, masks.squeeze((0,1)).detach().cpu().numpy()))

        # 3. Cleanup Memory
        if idx > 0:
            self.features_cache.pop(idx-1, None)
            if idx > self.history_limit:
                self.memory_bank.pop(idx - self.history_limit, None)

        self.frame_idx += 1
        return final_results
    
    def _prep_prompts(
    self, point_coords, point_labels, box, mask_logits, normalize_coords, img_idx=-1
    ):

        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        # print(img_idx)
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = torch.as_tensor(
                point_coords, dtype=torch.float, device=self.device
            )
            unnorm_coords = self._transforms.transform_coords(
                point_coords, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )
            labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            if len(unnorm_coords.shape) == 2:
                unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float, device=self.device)
            unnorm_box = self._transforms.transform_boxes(
                box, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )  # Bx2x2
        if mask_logits is not None:
            mask_input = torch.as_tensor(
                mask_logits, dtype=torch.float, device=self.device
            )
            if len(mask_input.shape) == 3:
                mask_input = mask_input[None, :, :, :]
        return mask_input, unnorm_coords, labels, unnorm_box

class RealTimeApp:
    def __init__(self, encoder_engine, decoder_engine):
        # 1. Camera Setup
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # 2. Predictor Setup
        self.predictor = OnlineTRTSAM2(encoder_engine, decoder_engine)
        
        # 3. App State
        self.objects = {} # {id: {color, points, labels}}
        self.active_id = 1
        self.interaction_queue = {}
        self.add_new_object(1)

    def add_new_object(self, obj_id):
        color = np.random.randint(50, 255, size=3).tolist()
        self.objects[obj_id] = {'color': color, 'points': [], 'labels': []}
        self.active_id = obj_id

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.queue_interaction(x, y, 1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.queue_interaction(x, y, 0)

    def queue_interaction(self, x, y, label):
        if self.active_id not in self.interaction_queue:
            self.interaction_queue[self.active_id] = {'points': [], 'labels': []}
        
        self.interaction_queue[self.active_id]['points'].append([x, y])
        self.interaction_queue[self.active_id]['labels'].append(label)
        # Persistent storage for the "Annotating" logic
        self.objects[self.active_id]['points'].append([x, y])
        self.objects[self.active_id]['labels'].append(label)

    def run(self):
        profile = self.pipeline.start(self.config)
        cv2.namedWindow("TRT-SAM2 RealTime", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("TRT-SAM2 RealTime", self.mouse_callback)
        
        self.predictor.init_session()
        print("Streaming... Press 'n' for new object, 'Tab' to switch, 'q' to quit.")

        try:
            while True:
                start_tick = time.time()
                
                # A. Get Camera Frame
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                # color_frame = cv2.imread('images/handme_Color.png')
                if not color_frame: continue
                
                frame_bgr = np.asanyarray(color_frame.get_data())
                # frame_bgr = color_frame.copy()
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # B. Process Interactions

                current_interactions = self.interaction_queue
                # self.interaction_queue = {} # Clear for next frame

                # C. Inference Step (TensorRT)
                # Returns list of (obj_id, binary_mask)
                results = self.predictor.step(frame_rgb, current_interactions, list(self.objects.keys()))

                # D. Visualization
                display = frame_bgr.copy()
                for obj_id, mask in results:
                    
                    color = self.objects[obj_id]['color']
                    overlay = np.zeros_like(display)
                    # print(f"Object {obj_id} mask shape: {mask.shape}, {overlay.shape}")
                    overlay[mask] = color
                    # Fast blending
                    display = cv2.addWeighted(display, 1.0, overlay, 0.4, 0)

                # UI Text
                fps = 1.0 / (time.time() - start_tick)
                cv2.putText(display, f"FPS: {int(fps)} | Obj: {self.active_id}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("TRT-SAM2 RealTime", display)

                # E. Keyboard Controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('n'):
                    new_id = max(self.objects.keys()) + 1
                    self.add_new_object(new_id)
                    print(f"Added Object {new_id}")
                elif key == 9: # Tab
                    ids = list(self.objects.keys())
                    self.active_id = ids[(ids.index(self.active_id) + 1) % len(ids)]
                elif key == ord('r'):
                    self.predictor.init_session()
                    self.objects = {}
                    self.add_new_object(1)

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load your TensorRT engines here using your existing loader
    # e.g., trt_encoder = TRTModule("encoder.engine")
    encoder_trt = TRTWrapper("sam2_encoder.engine")
    decoder_trt = TRTWrapper("sam2_decoder_video.engine")
    app = RealTimeApp(encoder_trt, decoder_trt)
    app.run()
    # pass