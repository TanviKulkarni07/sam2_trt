import time
import os

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

trt_optimized = False  # Set to False to use non-TensorRT model in real-time
if trt_optimized:
    sam2_checkpoint = "checkpoints/sam2_mem_attn_tiny.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l_trt.yaml"
else:
    print("WARNING: Running non-TensorRT optimized model in Real-Time App.")
    sam2_checkpoint = "checkpoints/sam2.1_hiera_tiny.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

predictor = build_sam2_video_predictor(
    model_cfg, sam2_checkpoint, device=device, trt_optimized=trt_optimized
)

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
video_dir = "notebooks/videos/bedroom"

# scan all the JPEG frame names in this directory
frame_names = [
    p
    for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# take a look the first video frame
frame_idx = 0

inference_state = predictor.init_state(video_path=video_dir)

ann_frame_idx = 0
ann_obj_id = 1
prompts = {}

points = np.array([[210, 350], [250, 220], [200, 250]], dtype=np.float32)
labels = np.array([1, 1, 1], np.int32)
prompts[ann_obj_id] = points, labels

# --- BENCHMARKING INTERACTION ---
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
(out_mask_info, out_obj_ids, out_mask_logits) = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)
end_event.record()
torch.cuda.synchronize()
interaction_time = start_event.elapsed_time(end_event)
fps = 1000.0 / interaction_time if interaction_time > 0 else 0
print(f"Obj 1 (3 points) interaction time: {interaction_time:.3f} ms | FPS: {fps:.2f}")

plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))

for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.show()

# --- BENCHMARKING PROPAGATION ---
print(f"Starting propagation for {len(frame_names)} frames...")
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()

video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
    inference_state
):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

end_event.record()
torch.cuda.synchronize()
total_prop_time = start_event.elapsed_time(end_event)
fps = len(frame_names) / (total_prop_time / 1000.0)  # Convert ms to seconds
print(f"[Propagation] Total time: {total_prop_time:.2f} ms | Avg FPS: {fps:.2f}")
