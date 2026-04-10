#!/usr/bin/env python3
import time
from collections import deque

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from sam2.build_sam import build_sam2_video_predictor
from trt_realsense.online_sam2 import OnlineSAM2


class Sam2RealtimeRosNode(Node):
    def __init__(self):
        super().__init__("sam2_realtime_node")

        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("checkpoint", "checkpoints/sam2_mem_attn_tiny.pt")
        self.declare_parameter("config", "configs/sam2.1/sam2.1_hiera_l_trt.yaml")
        self.declare_parameter("use_trt", True)
        self.declare_parameter("history", 20)
        self.declare_parameter("queue_size", 1)
        self.declare_parameter("show_window", True)

        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        checkpoint = self.get_parameter("checkpoint").get_parameter_value().string_value
        config = self.get_parameter("config").get_parameter_value().string_value
        use_trt = self.get_parameter("use_trt").get_parameter_value().bool_value
        history = self.get_parameter("history").get_parameter_value().integer_value
        queue_size = self.get_parameter("queue_size").get_parameter_value().integer_value
        self.show_window = self.get_parameter("show_window").get_parameter_value().bool_value

        self.bridge = CvBridge()
        self.frame_queue = deque(maxlen=max(1, queue_size))

        self.subscription = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_base = build_sam2_video_predictor(
            config,
            checkpoint,
            device=self.device,
            trt_optimized=use_trt,
        )
        self.predictor = OnlineSAM2(base_class=self.model_base, history_limit=history)

        self.objects = {}
        self.active_obj = 1
        self.interaction_queue = {}
        self.add_obj(1)

        self.initialized = False
        self.prev_time = time.time()

        self.timer = self.create_timer(0.0, self.process_latest_frame)

        if self.show_window:
            self.window_title = "Realtime SAM2 ROS2"
            cv2.namedWindow(self.window_title, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(self.window_title, self.mouse_callback)

        self.get_logger().info(
            f"Listening on {image_topic}; TRT={'on' if use_trt else 'off'}; device={self.device}"
        )

    def add_obj(self, obj_id: int):
        self.objects[obj_id] = {
            "color": np.random.randint(50, 255, 3).tolist(),
            "points": [],
            "labels": [],
        }
        self.active_obj = obj_id

    def queue_click(self, x: int, y: int, label: int):
        if self.active_obj not in self.interaction_queue:
            self.interaction_queue[self.active_obj] = {"points": [], "labels": []}

        self.interaction_queue[self.active_obj]["points"].append([x, y])
        self.interaction_queue[self.active_obj]["labels"].append(label)
        self.objects[self.active_obj]["points"].append([x, y])
        self.objects[self.active_obj]["labels"].append(label)

    def mouse_callback(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.queue_click(x, y, 1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.queue_click(x, y, 0)

    def image_callback(self, msg: Image):
        frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.frame_queue.append(frame_bgr)

    def process_latest_frame(self):
        if not self.frame_queue:
            return

        frame_bgr = self.frame_queue.pop()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if not self.initialized:
            self.predictor.init_online_session(frame_rgb)
            self.initialized = True
            self.get_logger().info("SAM2 online session initialized")
            return

        interactions = self.interaction_queue
        self.interaction_queue = {}

        with torch.inference_mode():
            mask_logits = self.predictor.step(frame_rgb, interactions)

        if mask_logits is not None:
            for i, obj_id in enumerate(self.predictor.inference_state["obj_ids"]):
                if obj_id in self.objects:
                    mask = (mask_logits[i, 0] > 0.0).cpu().numpy()
                    color = self.objects[obj_id]["color"]
                    frame_bgr[mask] = frame_bgr[mask] * 0.5 + np.array(color) * 0.5

        if self.show_window:
            now = time.time()
            fps = 1.0 / (now - self.prev_time + 1e-6)
            self.prev_time = now

            h, w = frame_bgr.shape[:2]
            cv2.rectangle(frame_bgr, (0, 0), (w, 40), (40, 40, 40), -1)
            status_text = (
                f"FPS: {fps:.1f} | Active Obj: {self.active_obj} | Total: {len(self.objects)}"
            )
            cv2.putText(
                frame_bgr,
                status_text,
                (10, 28),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            legend = "[N]: New Obj [Tab]: Switch [Space]: Reset [Q]: Quit"
            cv2.putText(
                frame_bgr,
                legend,
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )

            active_color = self.objects[self.active_obj]["color"]
            cv2.circle(frame_bgr, (w - 25, 22), 10, active_color, -1)
            cv2.circle(frame_bgr, (w - 25, 22), 10, (255, 255, 255), 2)
            cv2.imshow(self.window_title, frame_bgr)

            key = cv2.waitKey(1)
            if key == ord("q"):
                self.get_logger().info("Quit requested from OpenCV window")
                rclpy.shutdown()
            elif key == ord("n"):
                self.add_obj(max(self.objects.keys()) + 1)
            elif key == ord("\t"):
                keys = list(self.objects.keys())
                curr = keys.index(self.active_obj)
                self.active_obj = keys[(curr + 1) % len(keys)]
            elif key == ord(" "):
                self.predictor.predictor.reset_state(self.predictor.inference_state)


def main(args=None):
    rclpy.init(args=args)
    node = Sam2RealtimeRosNode()
    try:
        rclpy.spin(node)
    finally:
        if node.show_window:
            cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
