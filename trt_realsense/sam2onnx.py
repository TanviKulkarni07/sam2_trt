import torch.nn as nn
import torch.nn.functional as F
from torch.export import Dim
import torch
import os
import argparse
import numpy as np
from sam2.build_sam import build_sam2_video_predictor


class SAM2EncoderExport(nn.Module):
    def __init__(self, sam2_model):
        super().__init__()
        self.model = sam2_model
        self.image_encoder = sam2_model.image_encoder
        self.decoder = sam2_model.sam_mask_decoder

    def forward(self, x):
        """Get the image feature on the input batch."""
        backbone_out = self.image_encoder(x)
        vision_pos_enc_2 = backbone_out["vision_pos_enc"][2]
        vision_features = backbone_out["backbone_fpn"][2]

        high_res_0 = self.decoder.conv_s0(backbone_out["backbone_fpn"][0])
        high_res_1 = self.decoder.conv_s1(backbone_out["backbone_fpn"][1])

        return vision_features, vision_pos_enc_2, high_res_0, high_res_1


class SAM2DecoderExport_points(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.sam_mask_decoder
        self.pe_layer = model.sam_prompt_encoder

    def forward(self, image_embed, high_res_0, high_res_1, point_coords, point_labels):
        # 1. Prepare Features List
        high_res_features = [high_res_0, high_res_1]

        # 2. Embed Prompts
        sparse_embeddings, dense_embeddings = self.pe_layer(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )

        # 3. Run Decoder
        low_res_multimasks, ious, sam_output_tokens, object_score_logits = self.model(
            image_embeddings=image_embed,
            image_pe=self.pe_layer.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,  # Standard inference mode
            high_res_features=high_res_features,
        )
        return low_res_multimasks, ious, sam_output_tokens, object_score_logits


class SAM2DecoderExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.sam_mask_decoder
        self.pe_layer = model.sam_prompt_encoder

    def forward(
        self, image_embed, high_res_0, high_res_1, point_coords, point_labels, masks
    ):
        # 1. Prepare Features List
        high_res_features = [high_res_0, high_res_1]

        # 2. Embed Prompts
        sparse_embeddings, dense_embeddings = self.pe_layer(
            points=(point_coords, point_labels),
            boxes=None,
            masks=masks,
        )

        # 3. Run Decoder
        low_res_multimasks, ious, sam_output_tokens, object_score_logits = self.model(
            image_embeddings=image_embed,
            image_pe=self.pe_layer.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,  # Standard inference mode
            high_res_features=high_res_features,
        )
        return low_res_multimasks, ious, sam_output_tokens, object_score_logits


class SAM2MemoryEncoderExport(torch.nn.Module):
    def __init__(self, sam2_model):
        super().__init__()
        self.mem_enc = sam2_model.memory_encoder

    def forward(self, pix_feat, mask_for_mem):
        # Apply sigmoid to mask as expected by SAM2 memory encoder
        # Explicitly return a tuple of Tensors (NOT a dict)
        out = self.mem_enc(pix_feat, mask_for_mem, skip_mask_sigmoid=True)
        return out["vision_features"], out["vision_pos_enc"]


class SAM2ProjExport(torch.nn.Module):
    def __init__(self, sam2_model):
        super().__init__()
        self.proj = sam2_model.obj_ptr_tpos_proj

    def forward(self, obj_pos):
        proj_obj_pos = self.proj(obj_pos)
        return proj_obj_pos


class SAM2ObjPtrProjExport(torch.nn.Module):
    def __init__(self, sam2_model):
        super().__init__()
        self.proj = sam2_model.obj_ptr_proj

    def forward(self, sam_output_token):
        obj_ptr = self.proj(sam_output_token)
        return obj_ptr


def parse_args():
    parser = argparse.ArgumentParser(description="Export SAM2 modules to ONNX.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to SAM2 checkpoint (e.g., sam2.1_hiera_tiny.pt)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Model config file (e.g., sam2.1_hiera_t.yaml)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="engines/", help="Folder to save ONNX models"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use: cuda or cpu"
    )
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    return parser.parse_args()


def create_onnx_models():

    args = parse_args()

    OPSET = args.opset
    DEVICE = torch.device(args.device)
    OUT_DIR = args.output_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading model: {args.checkpoint}")
    sam2_model = build_sam2_video_predictor(args.config, args.checkpoint, device=DEVICE)

    # Define dynamic paths based on output_dir
    paths = {
        "encoder": os.path.join(OUT_DIR, "sam2_encoder.onnx"),
        "decoder": os.path.join(OUT_DIR, "sam2_decoder.onnx"),
        "decoder_pts": os.path.join(OUT_DIR, "sam2_decoder_points.onnx"),
        "memory": os.path.join(OUT_DIR, "sam2_mem_encoder.onnx"),
        "proj": os.path.join(OUT_DIR, "sam2_proj.onnx"),
        "objproj": os.path.join(OUT_DIR, "sam2_objproj.onnx"),
        "attn": os.path.join(OUT_DIR, "sam2_mem_attn.onnx"),
    }

    # --- Dummy Inputs ---
    dummy_img = torch.randn(1, 3, 1024, 1024).to(DEVICE)
    dummy_embed = torch.randn(1, 256, 64, 64).to(DEVICE)
    dummy_feat_0 = torch.randn(1, 32, 256, 256).to(DEVICE)
    dummy_feat_1 = torch.randn(1, 64, 128, 128).to(DEVICE)
    dummy_coords = torch.randint(0, 1024, (1, 1, 2), dtype=torch.float32).to(DEVICE)
    dummy_labels = torch.randint(0, 1, (1, 1), dtype=torch.int32).to(DEVICE)
    dummy_masks = torch.randn(1, 1, 256, 256).to(DEVICE)

    if not os.path.exists(paths["encoder"]):

        print("Exporting Encoder...")
        dynamic_axes = {
            "image": {0: "num_objects"},  # Axis 0 is N (dynamic)
        }
        torch.onnx.export(
            SAM2EncoderExport(sam2_model).to(DEVICE).eval(),
            dummy_img,
            paths["encoder"],
            input_names=["image"],
            output_names=[
                "vision_features",
                "vision_pos_enc",
                "high_res_0",
                "high_res_1",
            ],
            opset_version=OPSET,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
        print("Encoder exported successfully!")

    else:
        print("Onnx Encoder already exists.")

    if not os.path.exists(paths["decoder"]):
        print("Exporting Decoder...")
        torch.onnx.export(
            SAM2DecoderExport(sam2_model).to(DEVICE).eval(),
            # Tuple of inputs matching the keys above
            (
                dummy_embed,
                dummy_feat_0,
                dummy_feat_1,
                dummy_coords,
                dummy_labels,
                dummy_masks,
            ),
            paths["decoder"],
            input_names=[
                "image_embed",
                "high_res_0",
                "high_res_1",
                "point_coords",
                "point_labels",
                "masks",
            ],
            output_names=[
                "low_res_multimasks",
                "ious",
                "sam_output_tokens",
                "object_score_logits",
            ],
            opset_version=OPSET,
            dynamic_axes={
                "image_embed": {0: "num_images"},
                "high_res_0": {0: "num_images"},
                "high_res_1": {0: "num_images"},
                "point_coords": {0: "num_images", 1: "num_points"},
                "point_labels": {0: "num_images", 1: "num_points"},
                "masks": {0: "num_images", 1: "num_objects"},  # Axis 0 is N (dynamic)
            },
        )
        print("Decoder exported successfully!")
    else:
        print("Onnx Decoder already exists.")

    if not os.path.exists(paths["decoder_pts"]):
        print("Exporting Decoder Points...")
        torch.onnx.export(
            SAM2DecoderExport_points(sam2_model).to(DEVICE).eval(),
            # Tuple of inputs matching the keys above
            (dummy_embed, dummy_feat_0, dummy_feat_1, dummy_coords, dummy_labels),
            paths["decoder_pts"],
            input_names=[
                "image_embed",
                "high_res_0",
                "high_res_1",
                "point_coords",
                "point_labels",
            ],
            output_names=[
                "low_res_multimasks",
                "ious",
                "sam_output_tokens",
                "object_score_logits",
            ],
            opset_version=OPSET,
            dynamic_axes={
                "image_embed": {0: "num_images"},
                "high_res_0": {0: "num_images"},
                "high_res_1": {0: "num_images"},
                "point_coords": {0: "num_images", 1: "num_points"},
                "point_labels": {0: "num_images", 1: "num_points"},
            },
            dynamo=False,
        )
        print("Decoder Points exported successfully!")

    else:
        print("Onnx Decoder for points already exists.")

    if not os.path.exists(paths["memory"]):
        print("Exporting Memory Encoder...")
        dummy_pix_feat = torch.randn(1, 256, 64, 64).to(DEVICE)  # Example shape
        dummy_mask_mem = torch.randn(1, 1, 1024, 1024).to(DEVICE)
        dynamic_axes = {
            "pix_feat": {0: "batch"},
            "mask_for_mem": {0: "batch"},
            "memory": {0: "batch"},
            "memory_pos_enc": {0: "batch"},
        }
        torch.onnx.export(
            SAM2MemoryEncoderExport(sam2_model)
            .to(DEVICE)
            .eval(),  # <--- Pass the instance here
            (
                dummy_pix_feat,
                dummy_mask_mem,
            ),  # <--- Pass the dummy inputs as a tuple here
            paths["memory"],
            opset_version=OPSET,
            input_names=["pix_feat", "mask_for_mem"],
            output_names=["memory_features", "memory_pos_enc"],
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
        print("Memory Encoder exported successfully!")
    else:
        print("Onnx Memory Encoder already exists.")

    if not os.path.exists(paths["proj"]):
        dynamic_axes = {
            "obj_pos": {0: "num_objects"},  # Axis 0 is N (dynamic)
            "proj_obj_pos": {0: "num_objects"},  # Output must match input N
        }
        torch.onnx.export(
            SAM2ProjExport(sam2_model).to(DEVICE).eval(),
            (torch.randn(1, 1, 256).to(DEVICE)),
            paths["proj"],
            opset_version=OPSET,
            input_names=["obj_pos"],
            output_names=["obj_pos_proj"],
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
        print("Proj exported successfully!")
    else:
        print("Onnx Proj path already exits.")

    if not os.path.exists(paths["objproj"]):
        dynamic_axes = {
            "sam_output_tokens": {0: "num_objects"},  # Axis 0 is N (dynamic)
            "proj_obj_tokens": {0: "num_objects"},  # Output must match input N
        }
        torch.onnx.export(
            SAM2ObjPtrProjExport(sam2_model)
            .to(DEVICE)
            .eval(),  # <--- Pass the instance here
            (
                torch.randn(1, 256).to(DEVICE)
            ),  # <--- Pass the dummy inputs as a tuple here
            paths["objproj"],
            opset_version=OPSET,
            input_names=["sam_output_tokens"],
            output_names=["proj_obj_tokens"],
            # Use dynamo=False to use the legacy (more stable) exporter for SAM2
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
        print("ObjProj exported successfully!")
    else:
        print("Onnx ObjProj path already exits.")

    return


if __name__ == "__main__":
    create_onnx_models()
