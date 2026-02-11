import argparse
import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from sam2.build_sam import build_sam2_video_predictor
from trt_base import SAM2VideoPredictor_TRT, benchmark_interaction
from sam2.sam2_video_predictor import SAM2VideoPredictor
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time SAM 2 with Intel RealSense")
    parser.add_argument(
        "--trt", action="store_true", help="Use TensorRT optimized model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/sam2_mem_attn_tiny.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_l_trt.yaml",
        help="Path to model config",
    )
    parser.add_argument(
        "--history", type=int, default=20, help="Max history length for tracking"
    )
    parser.add_argument("--width", type=int, default=640, help="RealSense stream width")
    parser.add_argument(
        "--height", type=int, default=480, help="RealSense stream height"
    )
    parser.add_argument("--fps", type=int, default=30, help="RealSense target FPS")
    return parser.parse_args()


class OnlineSAM2(torch.nn.Module):
    def __init__(
        self, base_class: SAM2VideoPredictor_TRT, history_limit: int, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.inference_state = None
        self.frame_idx = 0
        self.history_limit = history_limit
        self.predictor = base_class
        self.device = base_class.device
        self.image_size = base_class.image_size

        self.pixel_mean = (
            torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        )
        self.pixel_std = (
            torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
        )

    def init_online_session(self, first_frame):
        self.frame_idx = 0
        self.inference_state = {}

        # Standard SAM2 initialization
        self.inference_state["images"] = {}
        self.inference_state["num_frames"] = 999999
        self.inference_state["offload_video_to_cpu"] = False
        self.inference_state["offload_state_to_cpu"] = False
        self.inference_state["video_height"] = first_frame.shape[0]
        self.inference_state["video_width"] = first_frame.shape[1]
        self.inference_state["device"] = self.device
        self.inference_state["storage_device"] = self.device
        self.inference_state["point_inputs_per_obj"] = {}
        self.inference_state["mask_inputs_per_obj"] = {}
        self.inference_state["cached_features"] = {}
        self.inference_state["constants"] = {}
        self.inference_state["obj_id_to_idx"] = {}
        self.inference_state["obj_idx_to_id"] = {}
        self.inference_state["obj_ids"] = []
        self.inference_state["output_dict_per_obj"] = {}
        self.inference_state["temp_output_dict_per_obj"] = {}
        self.inference_state["frames_tracked_per_obj"] = {}

        self._process_new_frame(first_frame, 0)
        return self.inference_state

    def _process_new_frame(self, frame, idx):
        # Normalize
        img_tensor = torch.from_numpy(frame).to(self.device).float().permute(2, 0, 1)
        img_tensor = (img_tensor / 255.0 - self.pixel_mean) / self.pixel_std

        if (
            img_tensor.shape[1] != self.image_size
            or img_tensor.shape[2] != self.image_size
        ):
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        self.inference_state["images"][idx] = img_tensor
        self.predictor._get_image_feature(self.inference_state, idx, batch_size=1)

    def _purge_old_history(self, current_idx):
        """Efficiently drop frames using a threshold instead of list searching."""
        cutoff = current_idx - self.history_limit
        if cutoff <= 0:
            return

        # Rapidly pop keys below the cutoff
        # Using list(keys) to avoid 'dictionary changed size during iteration'
        for rm_idx in [k for k in self.inference_state["images"].keys() if k < cutoff]:

            for obj_idx in self.inference_state["output_dict_per_obj"]:
                obj_data = self.inference_state["output_dict_per_obj"][obj_idx]
                obj_data["cond_frame_outputs"].pop(rm_idx, None)
                obj_data["non_cond_frame_outputs"].pop(rm_idx, None)

    def step(self, frame, active_interactions=None):
        idx = self.frame_idx

        # 1. Embed new frame
        if idx > 0:
            self._process_new_frame(frame, idx)

        # 2. Handle Interactions
        interacted_obj_idxs = []
        if active_interactions:

            for obj_id, data in active_interactions.items():
                _, out_obj_ids, _ = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=idx,
                    obj_id=obj_id,
                    points=data["points"],
                    labels=data["labels"],
                    clear_old_points=True,
                )

                interacted_obj_idxs.extend(
                    [
                        self.predictor._obj_id_to_idx(self.inference_state, oid)
                        for oid in out_obj_ids
                    ]
                )

        # 3. Handle Tracking
        all_obj_idxs = list(self.inference_state["obj_id_to_idx"].values())
        objs_to_track = [o for o in all_obj_idxs if o not in interacted_obj_idxs]

        if objs_to_track:

            for obj_idx in objs_to_track:
                obj_output_dict = self.inference_state["output_dict_per_obj"][obj_idx]
                current_out, _ = self.predictor._run_single_frame_inference(
                    inference_state=self.inference_state,
                    output_dict=obj_output_dict,
                    frame_idx=idx,
                    batch_size=1,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=False,
                    run_mem_encoder=True,
                )

                self.inference_state["temp_output_dict_per_obj"][obj_idx][
                    "non_cond_frame_outputs"
                ][idx] = current_out

        # 4. Consolidate & Encode Memory
        consolidated_out = self.predictor._consolidate_temp_output_across_obj(
            self.inference_state,
            idx,
            is_cond=(len(interacted_obj_idxs) > 0),
            consolidate_at_video_res=True,
        )

        for obj_idx in all_obj_idxs:
            obj_temp_output = self.inference_state["temp_output_dict_per_obj"][obj_idx]
            storage_key = (
                "cond_frame_outputs"
                if obj_idx in interacted_obj_idxs
                else "non_cond_frame_outputs"
            )

            if idx in obj_temp_output[storage_key]:
                out = obj_temp_output[storage_key][idx]

                if out["maskmem_features"] is None:
                    high_res_masks = F.interpolate(
                        out["pred_masks"],
                        size=(self.image_size, self.image_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                    maskmem_features, maskmem_pos_enc = (
                        self.predictor._run_memory_encoder(
                            self.inference_state,
                            idx,
                            1,
                            high_res_masks,
                            out["object_score_logits"],
                            is_mask_from_pts=(obj_idx in interacted_obj_idxs),
                        )
                    )
                    out["maskmem_features"] = maskmem_features
                    out["maskmem_pos_enc"] = maskmem_pos_enc

                self.inference_state["output_dict_per_obj"][obj_idx][storage_key][
                    idx
                ] = out

        if idx > 0:
            self.inference_state["images"].pop(idx - 1, None)
            self.inference_state["cached_features"].pop(idx - 1, None)

        if idx > self.history_limit:  # Only check every 10 frames to save CPU
            self._purge_old_history(idx)

        self.frame_idx += 1

        return consolidated_out["pred_masks_video_res"]


class RealTimeApp:
    def __init__(self, args):

        self.args = args
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(
            rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps
        )
        # print("args.trt)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_base = build_sam2_video_predictor(
            args.config, args.checkpoint, device=device, trt_optimized=args.trt
        )
        self.predictor = OnlineSAM2(
            base_class=model_base,
            history_limit=args.history,
        )

        self.objects = {}
        self.active_obj = 1
        self.add_obj(1)
        self.interaction_queue = {}

    def add_obj(self, obj_id):
        self.objects[obj_id] = {
            "color": np.random.randint(50, 255, 3).tolist(),
            "points": [],
            "labels": [],
        }
        self.active_obj = obj_id

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.queue_click(x, y, 1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.queue_click(x, y, 0)

    def queue_click(self, x: int, y: int, label: int):
        if self.active_obj not in self.interaction_queue:
            self.interaction_queue[self.active_obj] = {"points": [], "labels": []}
        self.interaction_queue[self.active_obj]["points"].append([x, y])
        self.interaction_queue[self.active_obj]["labels"].append(label)
        self.objects[self.active_obj]["points"].append([x, y])
        self.objects[self.active_obj]["labels"].append(label)

    def run(self):
        self.pipeline.start(self.config)
        cv2.namedWindow("Online SAM 2", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Online SAM 2", self.mouse_callback)
        cv2.startWindowThread()  # Safe threading for Windows

        print("Initializing...")
        frames = self.pipeline.wait_for_frames()
        frame0 = np.asanyarray(frames.get_color_frame().get_data())
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)

        self.predictor.init_online_session(frame0)

        print(f"Running! History limit: {self.predictor.history_limit} frames.")

        try:
            while True:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                frame_bgr = np.asanyarray(color_frame.get_data())
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                interactions = self.interaction_queue
                self.interaction_queue = {}

                with torch.inference_mode():  # Ensure no gradients accumulate
                    mask_logits = self.predictor.step(frame_rgb, interactions)

                if mask_logits is not None:
                    for i, obj_id in enumerate(
                        self.predictor.inference_state["obj_ids"]
                    ):
                        if obj_id in self.objects:
                            mask = (mask_logits[i, 0] > 0.0).cpu().numpy()
                            color = self.objects[obj_id]["color"]

                            frame_bgr[mask] = (
                                frame_bgr[mask] * 0.5 + np.array(color) * 0.5
                            )

                end_event.record()
                torch.cuda.synchronize()
                fps = (
                    1000.0 / start_event.elapsed_time(end_event)
                    if start_event.elapsed_time(end_event) > 0
                    else 0
                )
                cv2.putText(
                    frame_bgr,
                    f"FPS: {int(fps)} | Obj: {self.active_obj}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame_bgr,
                    f"Obj: {self.active_obj} | Mem: {len(self.predictor.inference_state['images'])}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Online SAM 2", frame_bgr)

                key = cv2.waitKey(1)
                if key == ord("q"):
                    break
                elif key == ord("n"):
                    self.add_obj(max(self.objects.keys()) + 1)
                elif key == ord("\t"):
                    keys = list(self.objects.keys())
                    curr = keys.index(self.active_obj)
                    self.active_obj = keys[(curr + 1) % len(keys)]
                elif key == ord(" "):
                    self.predictor.reset_state(self.predictor.inference_state)

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    app = RealTimeApp(args)
    app.run()
