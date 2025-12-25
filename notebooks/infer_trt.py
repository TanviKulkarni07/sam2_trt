import tensorrt as trt
import torch
import cv2
import numpy as np
import os

class TRTWrapper:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine not found: {engine_path}")
            
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        
        self.io_bindings = {}
        self.inputs = []
        self.outputs = []
        
        # Inspect Engine Tensors
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            
            # Map TRT Type -> Torch Type
            torch_dtype = self._trt_to_torch_dtype(dtype)
            
            # Identify if this tensor is input or output
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(name)
            else:
                self.outputs.append(name)

            # --- KEY FIX: Handle Dynamic Shapes (-1) ---
            if -1 in shape:
                # print(f"Dynamic shape detected for {name}: {shape}. deferring allocation.")
                # We cannot allocate yet because we don't know the size.
                # We set it to None and will allocate in infer()
                self.io_bindings[name] = None
            else:
                # Static shape: Allocate immediately
                tensor = torch.zeros(tuple(shape), dtype=torch_dtype, device="cuda")
                self.io_bindings[name] = tensor

    def _trt_to_torch_dtype(self, trt_dtype):
        # Mapping TensorRT data types to PyTorch
        if trt_dtype == trt.float32: return torch.float32
        if trt_dtype == trt.float16: return torch.float16
        if trt_dtype == trt.int32:   return torch.int32
        if trt_dtype == trt.int8:    return torch.int8
        if trt_dtype == trt.bool:    return torch.bool
        raise TypeError(f"Unsupported TRT dtype: {trt_dtype}")

    def infer(self, feed_dict):
        """
        feed_dict: A dictionary { "input_name": torch_tensor }
        """
        # 1. Update Input Shapes & Bindings
        for name, tensor in feed_dict.items():
            # Only process inputs that exist in the engine
            if name not in self.inputs:
                continue

            # A. Inform TensorRT of the dynamic shape
            if -1 in self.engine.get_tensor_shape(name):
                 self.context.set_input_shape(name, tensor.shape)
            
            # B. Re-allocate buffer if needed
            current_binding = self.io_bindings.get(name)
            if current_binding is None or current_binding.shape != tensor.shape:
                self.io_bindings[name] = torch.empty_like(tensor)

            # C. Copy data: User Tensor -> Internal Binding Buffer
            self.io_bindings[name].copy_(tensor)

        # 2. Handle Outputs (Resize if dynamic)
        for name in self.outputs:
            out_shape = self.context.get_tensor_shape(name)
            current_binding = self.io_bindings.get(name)
            
            # Re-allocate output buffer if shape changed
            if current_binding is None or tuple(current_binding.shape) != tuple(out_shape):
                dtype = self._trt_to_torch_dtype(self.engine.get_tensor_dtype(name))
                self.io_bindings[name] = torch.empty(tuple(out_shape), dtype=dtype, device="cuda")

        # 3. Set Addresses (Must be done every pass for dynamic buffers)
        for name, tensor in self.io_bindings.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        # 4. Execute Async
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        
        # 5. Return Outputs
        return {name: self.io_bindings[name] for name in self.outputs}

def preprocess_image(image_path):
    # Load Image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found")
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to 1024x1024 (Standard SAM2 size)
    # Note: For best results, preserve aspect ratio and pad, but resizing is simpler for a demo
    img = cv2.resize(img, (1024, 1024))
    
    # Normalize (SAM2 / Hiera standard mean-std)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    img = (img / 255.0 - mean) / std
    
    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0) # Add batch dim -> (1, 3, 1024, 1024)
    
    # Convert to contiguous array to avoid memory errors
    img = np.ascontiguousarray(img, dtype=np.float32)
    
    # To Torch Tensor on GPU
    return torch.from_numpy(img).cuda()

# --- Main Execution ---
if __name__ == "__main__":
    ENGINE_PATH = "sam2_encoder.engine"  # Update this
    DEC_PATH = "sam2_decoder.engine"
    IMAGE_PATH = 'images/handme_Color.png'     # Update this

    # try:
    # 1. Initialize
    print("Loading TensorRT Engine...")
    trt_model = TRTWrapper(ENGINE_PATH)

    # 2. Preprocess
    print(f"Processing {IMAGE_PATH}...")
    input_tensor = preprocess_image(IMAGE_PATH)
    
    # 3. Run Inference
    print("Running Inference...")
    enc_outputs = trt_model.infer({"image": input_tensor})
    
    point_coords = torch.tensor([[[500, 500]]], dtype=torch.float32, device="cuda")
    # 2. Point Labels (Batch, Num_Points) - Int32
    # 1 = Foreground, 0 = Background
    point_labels = torch.tensor([[1]], dtype=torch.int32, device="cuda")
    print(point_coords.shape, point_labels.shape)

    # 4. Process Results
    print("\n--- Inference Successful ---")
    for name, tensor in enc_outputs.items():
        # Move to CPU for printing/saving
        shape = tensor.shape
        print(f"Output '{name}': {shape} | Device: {tensor.device}")
        
        # Example: Save embeddings if needed
        # np.save(f"{name}.npy", tensor.cpu().numpy())

    print("Loading Decoder...")
    decoder = TRTWrapper(DEC_PATH)
    print("Running Decoder...")
    decoder_inputs = {
        "image_embed":     enc_outputs["image_embed"],
        "high_res_feat_0": enc_outputs["high_res_feat_0"],
        "high_res_feat_1": enc_outputs["high_res_feat_1"],
        "point_coords":    point_coords,
        "point_labels":    point_labels
    }
    
    dec_outputs = decoder.infer(decoder_inputs)

    for name, tensor in dec_outputs.items():
        # Move to CPU for printing/saving
        shape = tensor.shape
        print(f"Output '{name}': {shape} | Device: {tensor.device}")
    
    # except Exception as e:
    #     print(f"Error: {e}")