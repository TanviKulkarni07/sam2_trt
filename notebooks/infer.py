from PIL import Image
import torch
import numpy as np
import onnxruntime as ort
from sam2.build_sam import build_sam2

from sam2onnx import EncoderWrapper, DecoderWrapper
from sam2_image_predictor_general import Dataprocessor
from infer_trt import TRTWrapper
from display_utils import show_masks

sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = torch.device("cuda")
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)

img = Image.open('images/handme_Color.png')
input_point = np.array([[300, 275], [600,50]])
input_label = np.array([1, 0])

# sam2enc = EncoderWrapper(sam2_model)
# sam2dec = DecoderWrapper(sam2_model)

# providers = ["CUDAExecutionProvider"]
# encoder_session = ort.InferenceSession("sam2_encoder.onnx", providers=providers)
# decoder_session = ort.InferenceSession("sam2_decoder_new.onnx", providers=providers)

encoder_trt = TRTWrapper("sam2_encoder.engine")
decoder_trt = TRTWrapper("sam2_decoder.engine")

# dataproc_torch = Dataprocessor(sam2enc, sam2dec, 
#                          device=DEVICE,
#                          framework="torch")  # Use "onnx" for ONNX runtime

# dataproc_onnx = Dataprocessor(encoder_session, decoder_session, 
#                          device=DEVICE,
#                          framework="onnx")  # Use "onnx" for ONNX runtime

dataproc_trt = Dataprocessor(encoder_trt, decoder_trt, 
                         device=DEVICE,
                         framework="trt")  # Use "trt" for TensorRT runtime

dataproc_trt.set_image(img)
masks, iou_scores,_ = dataproc_trt.predict(
    point_coords=input_point,
    point_labels=input_label
    )
show_masks(img, masks, iou_scores, point_coords=input_point, 
           box_coords=None, input_labels=input_label, borders=True)