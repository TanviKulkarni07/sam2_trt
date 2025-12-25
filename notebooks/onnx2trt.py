import tensorrt as trt
import os

# --- CONFIGURATION ---
ONNX_PATH = "sam2_decoder_new.onnx"
ENGINE_PATH = "sam2_decoder.engine"


# 1. Setup Builder
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
config = builder.create_builder_config()

# Enable FP16 (Crucial for Encoder speed)
if builder.platform_has_fast_fp16:
    print("✅ FP16 Enabled")
    config.set_flag(trt.BuilderFlag.FP16)

# 2. Define Network & Parser
# Explicit Batch flag is required for modern TensorRT
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

# 3. Load ONNX
print(f"Loading {ONNX_PATH}...")
with open(ONNX_PATH, "rb") as model:
    if not parser.parse(model.read()):
        print("❌ Failed to parse ONNX file:")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        raise RuntimeError("ONNX Parsing Failed")

# 4. Define Optimization Profile (Static Shape)
# Even for static inputs, explicit batch mode often requires a profile.
# We set Min = Opt = Max to the same fixed shape.
profile = builder.create_optimization_profile()


#For Encoder
# INPUT_NAME = "image"  # Must match the name used in torch.onnx.export
# INPUT_SHAPE = (1, 3, 1024, 1024) # (Batch, Channels, Height, Width)
# profile.set_shape(INPUT_NAME, INPUT_SHAPE, INPUT_SHAPE, INPUT_SHAPE)

#For Decoder (uncomment if building decoder engine)
profile.set_shape("image_embed", (1, 256, 64, 64), (1, 256, 64, 64), (1, 256, 64, 64))
profile.set_shape("high_res_feat_0", (1, 256, 256, 256), (1, 256, 256, 256), (1, 256, 256, 256))
profile.set_shape("high_res_feat_1", (1, 256, 128, 128), (1, 256, 128, 128), (1, 256, 128, 128))

# Dynamic inputs (Points):
# (Min, Opt, Max) -> We allow 1 to 10 points
profile.set_shape("point_coords", (1, 1, 2), (1, 3, 2), (1, 10, 2))
profile.set_shape("point_labels", (1, 1), (1, 3), (1, 10))


config.add_optimization_profile(profile)

# 5. Build Engine
print("Building TensorRT Engine... (This will take a few minutes)")
try:
    # Note: build_serialized_network is the modern API (TRT 8.5+)
    serialized_engine = builder.build_serialized_network(network, config)
    
    with open(ENGINE_PATH, "wb") as f:
        f.write(serialized_engine)
    print(f"🎉 Success! Engine saved to {ENGINE_PATH}")

except Exception as e:
    print(f"❌ Build Failed: {e}")