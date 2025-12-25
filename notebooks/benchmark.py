import cv2
import torch
from PIL import Image
import torch
import numpy as np
import onnxruntime as ort
from sam2.build_sam import build_sam2

from sam2onnx import EncoderWrapper, DecoderWrapper
from sam2_image_predictor_general import Dataprocessor
from infer_trt import TRTWrapper

# ==========================================
# BENCHMARK
# ==========================================
def benchmark(name, func, *args):
    iterations=100
    # Warmup
    for _ in range(10):
        func(*args)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        func(*args)
    end.record()
    
    torch.cuda.synchronize()
    avg_time = start.elapsed_time(end) / iterations
    fps = 1000 / avg_time
    
    print(f"{name: <25} | {avg_time:6.2f} ms | {fps:6.1f} FPS")

def run_onnx_session(session, input_dict):
    outputs = session.run(None, input_dict)

def benchmark_torch():

    sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    DEVICE = torch.device("cuda")
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)

    sam2enc = EncoderWrapper(sam2_model)
    sam2dec = DecoderWrapper(sam2_model)

    DUMMY_IMAGE = torch.randn(1, 3, 1024, 1024).to(DEVICE)
    DUMMY_POINT_COORDS = torch.randint(0, 1024, (1, 1, 2)).float().to(DEVICE)
    DUMMY_POINT_LABELS = torch.randint(0, 1, (1, 1)).float().to(DEVICE)

    benchmark("Torch Encoder", sam2enc, DUMMY_IMAGE)
    DUMMY_EMBED, DUMMY_HIGH_RES_0, DUMMY_HIGH_RES_1, = sam2enc(DUMMY_IMAGE)
    benchmark("Torch Decoder", sam2dec, 
          DUMMY_EMBED, DUMMY_HIGH_RES_0, DUMMY_HIGH_RES_1,
          DUMMY_POINT_COORDS, DUMMY_POINT_LABELS)
    
def benchmark_onnx():

    image = Image.open('images/handme_Color.png')
    image = np.array(image.convert("RGB"))
    image = cv2.resize(image, (1024, 1024))
    image = image.astype(np.float32).transpose(2, 0, 1)[None, :, :, :] 

    providers = ["CUDAExecutionProvider"]
    encoder_session = ort.InferenceSession("sam2_encoder.onnx", providers=providers)
    decoder_session = ort.InferenceSession("sam2_decoder_new.onnx", providers=providers)
    print("ONNX", encoder_session.get_providers(), decoder_session.get_providers())

    image_embed, high_res_0, high_res_1 = encoder_session.run(None, {"image": image})
    points = np.array([[[300, 275], [600,50]]], dtype=np.float32)
    labels = np.array([[1, 0]], dtype=np.int32) # 1 = foreground
    decoder_inputs_onnx = {
    "image_embed": image_embed,
    "high_res_feat_0": high_res_0,
    "high_res_feat_1": high_res_1,
    "point_coords": points,
    "point_labels": labels,     # False
    }

    benchmark("ONNX Encoder", run_onnx_session, encoder_session, {"image": image})
    benchmark("ONNX Decoder", run_onnx_session, decoder_session, decoder_inputs_onnx)                    

def benchmark_trt():

    DEVICE = torch.device("cuda")
    encoder_trt = TRTWrapper("sam2_encoder.engine")
    decoder_trt = TRTWrapper("sam2_decoder.engine")

    DUMMY_IMAGE = torch.randn(1, 3, 1024, 1024).to(DEVICE)
    DUMMY_POINT_COORDS = torch.randint(0, 1024, (1, 1, 2)).float().to(DEVICE)
    DUMMY_POINT_LABELS = torch.randint(0, 1, (1, 1)).float().to(DEVICE)
    DUMMY_EMBED = torch.randn(1, 256, 64, 64).to(DEVICE)
    DUMMY_HIGH_RES_0 = torch.randn(1, 256, 256, 256).to(DEVICE) # Hiera-Large specific shape, might vary for Tiny/Base
    DUMMY_HIGH_RES_1 = torch.randn(1, 256, 128, 128).to(DEVICE)

    decoder_inputs_trt = {
    "image_embed": DUMMY_EMBED,
    "high_res_feat_0": DUMMY_HIGH_RES_0,
    "high_res_feat_1": DUMMY_HIGH_RES_1,
    "point_coords": DUMMY_POINT_COORDS,
    "point_labels": DUMMY_POINT_LABELS
    }

    benchmark("TensorRT Encoder", encoder_trt.infer, {"image": DUMMY_IMAGE})
    benchmark("TensorRT Decoder", decoder_trt.infer, decoder_inputs_trt)        

if __name__ == "__main__":
    
    print("-" * 60)
    print(f"{'Engine': <25} | {'Latency': <9} | {'Throughput'}")
    print("-" * 60)

    # benchmark_torch()
    # benchmark_onnx()
    benchmark_trt() 