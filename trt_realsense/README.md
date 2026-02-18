# TensorRT Speedup for SAM 2 Inference
### Real-time Segmentation with Intel RealSense D435

[![GitHub](https://img.shields.io/badge/GitHub-View_Project-blue?logo=github)](https://github.com/TanviKulkarni07/sam2_trt)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-orange)](https://huggingface.co/TanviKulkarni07/sam2-trt-engines)

<p align="center">
  <img src="../assets/sam2_trt.gif" width="80%" alt="SAM2 TensorRT Demo" />
</p>

---

## 🛠️ System Requirements
**Environment:** Ubuntu 22.04 | WSL2 (tested)  
**GPU:** NVIDIA GeForce RTX 4070 (16GB VRAM)

| Component | Version |
| :--- | :--- |
| **Python** | 3.10.19 |
| **PyTorch** | 2.9.1 |
| **TensorRT** | 10.15.1.29 |
| **pyrealsense2** | 2.56.5.9235 |

---

## 🚀 Getting Started

### 1. Installation
```bash
git clone https://github.com/TanviKulkarni07/sam2_trt.git
cd sam2_trt/
conda env create -f environment.yml

```

### 2. Model Export Pipeline
Step A: Convert the SAM2 model with preloaded 'checkpoint' and 'config' files to ONNX models/.
OR 
download pre-exported ONNX models from [hugging face](https://huggingface.co/TanviKulkarni07/sam2-trt-engines). 

```bash
python trt_realsense/sam2onnx.py --checkpoint checkpoint --config config --output-dir output-directory
```
Step B: Store the downloaded onnx models from previous step in a new folder as follows.

```bash
cd trt_realsense/ && mkdir tiny_onnx_models/
```

Step C: To generate each TRT engine for the respective ONNX model defined in 'config_file' ( trt_realsense/onnx2trt_config/ folder), run

```bash
python onnx2trt.py --config onnx2trt_config/config_file
```
## 📊 Performance Analysis

### Benchmarking TensorRT vs Pytorch Inference time 
#### 1. Offline/Pre-recorded video
Refer trt_video_predictor_example.ipynb. for TensorRT implemention of object tracking on an example video. 
To compare the inference times, run

```bash
python trt_realsense/benchmark_new.py         # For Pytorch inference
python trt_realsense/benchmark_new.py --trt   # For TensorRT-optimized inference
```

| <small>Model</small> | <small>New Object Addition</small> | <small>Propagation Time</small> |
| :--- | :--- | :--- |
| <small>SAM 2.1 Hiera-Tiny (Base)</small> | <small>6.5 FPS</small> | <small>40 FPS</small> |
| **<small>SAM 2.1 Hiera-Tiny (TensorRT)</small>** | **<small>26 FPS</small>** | **<small>39 FPS</small>** |

> **Analysis**: We observe a 4x performance gain during new object addition. While the standard PyTorch decoder for the "tiny" model is already well-optimized, the TensorRT engine provides a critical speedup during the encoder stage for new interactions. Propagation performance is currently capped by CPU overheads.

#### 2. Online/Real-time video 
```
python trt_realsense/infer_realtime.py --checkpoint checkpoints/sam2.1_hiera_tiny.pt --config configs/sam2.1/sam2.1_hiera_t.yaml # For Pytorch inference

python trt_realsense/infer_realtime.py --trt # For TensorRT-optimized inference 
 ```

![fps-comparison](../assets/fps_comparison.png) 
> **Analysis**: TensorRT optimization achieves a throughput increase during the interaction phase by significantly accelerating the encoder step for new object additions. While the standard PyTorch tiny decoder is natively efficient, the overall propagation performance is currently capped by CPU pre-processing overhead and WSL-related USB streaming latency on my personal setup.

### Acknowledgements
This project builds upon the [Segment Anything 2 framework](https://ai.meta.com/research/sam2/) and optimizes it realtime industrial vision applications. Special thanks to the [TierIV](https://github.com/tier4/sam2_pytorch2onnx) repository for providing the foundation for exporting SAM 2 models to ONNX.