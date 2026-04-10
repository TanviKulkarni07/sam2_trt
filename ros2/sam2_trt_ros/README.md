# sam2_trt_ros

ROS 2 (`ament_python`) package that runs `OnlineSAM2` TensorRT/PyTorch inference on frames from a ROS image topic (`sensor_msgs/msg/Image`).

## Features

- Subscribes to an image topic (`/camera/color/image_raw` by default)
- Converts ROS images with `cv_bridge`
- Runs the same online SAM2 tracking loop used in `trt_realsense/infer_realtime.py`
- Optional OpenCV interaction window:
  - Left click: positive point
  - Right click: negative point
  - `n`: new object
  - `Tab`: switch active object
  - `Space`: reset state
  - `q`: quit

## Build

From your ROS2 workspace root:

```bash
colcon build --packages-select sam2_trt_ros
source install/setup.bash
```

## Run

```bash
ros2 run sam2_trt_ros realtime_node \
  --ros-args \
  -p image_topic:=/camera/color/image_raw \
  -p checkpoint:=checkpoints/sam2_mem_attn_tiny.pt \
  -p config:=configs/sam2.1/sam2.1_hiera_l_trt.yaml \
  -p use_trt:=true \
  -p show_window:=true
```

> Ensure this repository root is in `PYTHONPATH` (or installed as a Python package) so imports like `sam2` and `trt_realsense` resolve from the ROS node.
