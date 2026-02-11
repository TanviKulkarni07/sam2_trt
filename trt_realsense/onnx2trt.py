import tensorrt as trt
import os
import yaml


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Post-process: Combine folder paths and convert lists to tuples
    folder = config["settings"]["folder"]
    config["onnx_path"] = os.path.join(folder, config["settings"]["onnx_filename"])
    config["engine_path"] = os.path.join(
        config["settings"]["engine_folder"], config["settings"]["engine_filename"]
    )

    # TensorRT expects tuples for shapes, YAML loads them as lists
    # We convert: List[List[int]] -> List[Tuple[int]]
    shapes = {}
    for name, profiles in config["proj_shapes"].items():
        shapes[name] = [tuple(p) for p in profiles]

    return config["settings"], shapes, config["onnx_path"], config["engine_path"]


def build_engine(onnx_path, engine_path, dynamic_shapes=None, fp16=True):
    """
    Converts an ONNX file to a TensorRT engine.

    Args:
        onnx_path: Path to the input ONNX file.
        engine_path: Path where the .engine file will be saved.
        dynamic_shapes: Dictionary mapping input names to (min, opt, max) shapes.
                        Example: {"input": [(1,1,256), (4,1,256), (16,1,256)]}
        fp16: Whether to enable FP16 precision.
    """
    # 1. Initialize Logger, Builder, and Network
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)

    # EXPLICIT_BATCH is required for ONNX models
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()

    # 2. Parse ONNX File
    if not os.path.exists(onnx_path):
        print(f"ONNX file {onnx_path} not found.")
        return

    # with open(onnx_path, "rb") as model:
    if not parser.parse_from_file(onnx_path):
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        return

    # 3. Handle Dynamic Shapes (Optimization Profile)
    if dynamic_shapes:
        profile = builder.create_optimization_profile()
        for input_name, shapes in dynamic_shapes.items():
            min_shape, opt_shape, max_shape = shapes
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

    # 4. Set Builder Configuration
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Set memory pool limit (e.g., 2GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * (1024**3))

    # 5. Build and Serialize Engine
    print(f"Building engine for {onnx_path}... This may take a few minutes.")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("Failed to build engine.")
        return

    # 6. Save Engine to Disk
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"Engine saved successfully to {engine_path}")


if __name__ == "__main__":
    # Load configuration
    settings, proj_shapes, onnx_path, engine_path = load_config(
        "onnx2trt_config/sam2_proj.yaml"
    )

    print(f"Building engine: {engine_path}")

    build_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        dynamic_shapes=proj_shapes,
        fp16=settings["fp16"],
    )
