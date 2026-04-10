import argparse
import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from sam2.build_sam import build_sam2_video_predictor
from online_sam2 import OnlineSAM2
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
        if args.trt:
            title = "Realtime SAM 2 (TensorRT Optimized)"
        else:
            title = "Realtime SAM 2 (PyTorch)"

        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(title, self.mouse_callback)
        cv2.startWindowThread()  # Safe threading for Windows

        print("Initializing...")
        frames = self.pipeline.wait_for_frames()
        frame0 = np.asanyarray(frames.get_color_frame().get_data())
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)

        self.predictor.init_online_session(frame0)

        print(
            "Controls: Left Click = Add Positive Point, Right Click = Add Negative Point, q = Quit"
        )
        print(
            "          n = Add New Object, Tab = Switch Active Object, Space = Reset Tracking"
        )
        prev_time = time.time()
        fps = 0
        try:
            while True:

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
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time + 1e-6)
                prev_time = curr_time

                # Background Banner for readability
                cv2.rectangle(frame_bgr, (0, 0), (self.args.width, 40), (40, 40, 40), -1)
                
                # Active Object & FPS
                status_text = f"FPS: {fps:.1f} | Active Obj: {self.active_obj} | Total: {len(self.objects)}"
                cv2.putText(frame_bgr, status_text, (10, 28), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

                # Legend at bottom
                legend = "[N]: New Obj  [Tab]: Switch  [Space]: Reset  [Q]: Quit"
                cv2.putText(frame_bgr, legend, (10, self.args.height - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                
                # Visual Indicator for Active Object Color
                active_color = self.objects[self.active_obj]["color"]
                cv2.circle(frame_bgr, (self.args.width - 25, 22), 10, active_color, -1)
                cv2.circle(frame_bgr, (self.args.width - 25, 22), 10, (255, 255, 255), 2)

                cv2.imshow(title, frame_bgr)

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
                    self.predictor.predictor.reset_state(self.predictor.inference_state)

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    app = RealTimeApp(args)
    app.run()
