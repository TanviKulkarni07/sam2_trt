import torch.nn as nn
from torch.export import Dim
import torch
import os
import numpy as np
from sam2.build_sam import build_sam2

# --- PART A: EXPORT IMAGE ENCODER ---

class EncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.image_encoder

    def forward(self, x):
        out = self.model(x)
        return out["vision_features"], out["backbone_fpn"][0], out["backbone_fpn"][1]

# --- PART B: EXPORT MASK DECODER ---
# The decoder takes the features + prompts.
# We must allow dynamic axes for 'point_coords' and 'point_labels'

class DecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.sam_mask_decoder
        self.pe_layer = model.sam_prompt_encoder

    def forward(self, image_embed, 
                high_res_0, 
                high_res_1, 
                point_coords,
                point_labels,
        ):
        # 1. Prepare Features List
        high_res_features = [self.model.conv_s0(high_res_0), self.model.conv_s1(high_res_1)]
        
        # 2. Embed Prompts
        sparse_embeddings, dense_embeddings = self.pe_layer(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )

        # 3. Run Decoder
        low_res_masks, iou_predictions, _, _ = self.model(
            image_embeddings=image_embed,
            image_pe=self.pe_layer.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=True, # Standard inference mode
            high_res_features=high_res_features
        )
        return low_res_masks, iou_predictions
    
def create_onnx_models():
    
    OPSET = 18  # SAM 2 requires a high opset (16 or 17) for modern operators
    DEVICE = torch.device("cuda")
    sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    DEVICE = torch.device("cuda")
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)

    dummy_img = torch.randn(1, 3, 1024, 1024).to(DEVICE)
    # Dummy Inputs for Decoder
    dummy_embed = torch.randn(1, 256, 64, 64).to(DEVICE)
    dummy_feat_0 = torch.randn(1, 256, 256, 256).to(DEVICE) # Hiera-Large specific shape, might vary for Tiny/Base
    dummy_feat_1 = torch.randn(1, 256, 128, 128).to(DEVICE)

    # Dummy Point: 1 point per image. Shape: (Batch, Num_Points, 2)
    dummy_coords = torch.randint(0, 1024, (1, 1, 2), dtype=torch.float32).to(DEVICE)
    dummy_labels = torch.randint(0, 1, (1, 1), dtype=torch.int32).to(DEVICE)

    # Define the dynamic dimension
    num_points = Dim("num_points", min=1)

    # CORRECTED dictionary: Includes ALL input arguments
    # Map static inputs to None, and dynamic inputs to their Dim specs
    dynamic_shapes = {
        "image_embed": None,   # Static
        "high_res_0": None,    # Static
        "high_res_1": None,    # Static
        "point_coords": {1: num_points},  # Dynamic dim 1
        "point_labels": {1: num_points},  # Dynamic dim 1
    }

    if not os.path.exists("sam2_encoder.onnx"):
        print("Exporting Encoder...")
        torch.onnx.export(
            EncoderWrapper(sam2_model),
            dummy_img,
            "sam2_encoder.onnx",
            input_names=["image"],
            output_names=["image_embed", "high_res_feat_0", "high_res_feat_1"],
            opset_version=OPSET,
            do_constant_folding=False,
        )
        print("Encoder exported successfully!")
    else:
        print("Onnx Encoder already exists")
    
    if not os.path.exists("sam2_decoder_new.onnx"):
        print("Exporting Decoder with corrected dynamic_shapes...")

        torch.onnx.export(
            DecoderWrapper(sam2_model),
            # Tuple of inputs matching the keys above
            (dummy_embed, dummy_feat_0, dummy_feat_1, 
             dummy_coords, dummy_labels),
            "sam2_decoder_new.onnx",
            input_names=["image_embed", "high_res_feat_0", 
                         "high_res_feat_1", "point_coords", "point_labels"],
            output_names=["masks", "iou_predictions"],
            opset_version=OPSET,
            dynamic_shapes=dynamic_shapes,
        )

        print("Decoder exported successfully!")
    else:
        print("Onnx Decoder already exists")

if __name__ == "__main__":
    create_onnx_models()