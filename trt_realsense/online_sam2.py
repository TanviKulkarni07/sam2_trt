import torch
import torch.nn.functional as F

from trt_base import SAM2VideoPredictor_TRT


class OnlineSAM2(torch.nn.Module):
    def __init__(
        self, base_class: SAM2VideoPredictor_TRT, history_limit: int, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.inference_state = None
        self.frame_idx = 0
        self.history_limit = history_limit
        self.predictor = base_class
        self.device = base_class.device
        self.image_size = base_class.image_size

        self.pixel_mean = (
            torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        )
        self.pixel_std = (
            torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
        )

    def init_online_session(self, first_frame):
        self.frame_idx = 0
        self.inference_state = {}

        self.inference_state["images"] = {}
        self.inference_state["num_frames"] = 999999
        self.inference_state["offload_video_to_cpu"] = False
        self.inference_state["offload_state_to_cpu"] = False
        self.inference_state["video_height"] = first_frame.shape[0]
        self.inference_state["video_width"] = first_frame.shape[1]
        self.inference_state["device"] = self.device
        self.inference_state["storage_device"] = self.device
        self.inference_state["point_inputs_per_obj"] = {}
        self.inference_state["mask_inputs_per_obj"] = {}
        self.inference_state["cached_features"] = {}
        self.inference_state["constants"] = {}
        self.inference_state["obj_id_to_idx"] = {}
        self.inference_state["obj_idx_to_id"] = {}
        self.inference_state["obj_ids"] = []
        self.inference_state["output_dict_per_obj"] = {}
        self.inference_state["temp_output_dict_per_obj"] = {}
        self.inference_state["frames_tracked_per_obj"] = {}

        self._process_new_frame(first_frame, 0)
        return self.inference_state

    def _process_new_frame(self, frame, idx):
        img_tensor = torch.from_numpy(frame).to(self.device).float().permute(2, 0, 1)
        img_tensor = (img_tensor / 255.0 - self.pixel_mean) / self.pixel_std

        if (
            img_tensor.shape[1] != self.image_size
            or img_tensor.shape[2] != self.image_size
        ):
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        self.inference_state["images"][idx] = img_tensor
        self.predictor._get_image_feature(self.inference_state, idx, batch_size=1)

    def _purge_old_history(self, current_idx):
        cutoff = current_idx - self.history_limit
        if cutoff <= 0:
            return

        for rm_idx in [k for k in self.inference_state["images"].keys() if k < cutoff]:
            for obj_idx in self.inference_state["output_dict_per_obj"]:
                obj_data = self.inference_state["output_dict_per_obj"][obj_idx]
                obj_data["cond_frame_outputs"].pop(rm_idx, None)
                obj_data["non_cond_frame_outputs"].pop(rm_idx, None)

    def step(self, frame, active_interactions=None):
        idx = self.frame_idx

        if idx > 0:
            self._process_new_frame(frame, idx)

        interacted_obj_idxs = []
        if active_interactions:
            for obj_id, data in active_interactions.items():
                _, out_obj_ids, _ = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=idx,
                    obj_id=obj_id,
                    points=data["points"],
                    labels=data["labels"],
                    clear_old_points=True,
                )

                interacted_obj_idxs.extend(
                    [
                        self.predictor._obj_id_to_idx(self.inference_state, oid)
                        for oid in out_obj_ids
                    ]
                )

        all_obj_idxs = list(self.inference_state["obj_id_to_idx"].values())
        objs_to_track = [o for o in all_obj_idxs if o not in interacted_obj_idxs]

        if objs_to_track:
            for obj_idx in objs_to_track:
                obj_output_dict = self.inference_state["output_dict_per_obj"][obj_idx]
                current_out, _ = self.predictor._run_single_frame_inference(
                    inference_state=self.inference_state,
                    output_dict=obj_output_dict,
                    frame_idx=idx,
                    batch_size=1,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=False,
                    run_mem_encoder=True,
                )

                self.inference_state["temp_output_dict_per_obj"][obj_idx][
                    "non_cond_frame_outputs"
                ][idx] = current_out

        consolidated_out = self.predictor._consolidate_temp_output_across_obj(
            self.inference_state,
            idx,
            is_cond=(len(interacted_obj_idxs) > 0),
            consolidate_at_video_res=True,
        )

        for obj_idx in all_obj_idxs:
            obj_temp_output = self.inference_state["temp_output_dict_per_obj"][obj_idx]
            storage_key = (
                "cond_frame_outputs"
                if obj_idx in interacted_obj_idxs
                else "non_cond_frame_outputs"
            )

            if idx in obj_temp_output[storage_key]:
                out = obj_temp_output[storage_key][idx]

                if out["maskmem_features"] is None:
                    high_res_masks = F.interpolate(
                        out["pred_masks"],
                        size=(self.image_size, self.image_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                    maskmem_features, maskmem_pos_enc = self.predictor._run_memory_encoder(
                        self.inference_state,
                        idx,
                        1,
                        high_res_masks,
                        out["object_score_logits"],
                        is_mask_from_pts=(obj_idx in interacted_obj_idxs),
                    )
                    out["maskmem_features"] = maskmem_features
                    out["maskmem_pos_enc"] = maskmem_pos_enc

                self.inference_state["output_dict_per_obj"][obj_idx][storage_key][idx] = out

        if idx > 0:
            self.inference_state["images"].pop(idx - 1, None)
            self.inference_state["cached_features"].pop(idx - 1, None)

        if idx > self.history_limit:
            self._purge_old_history(idx)

        self.frame_idx += 1

        return consolidated_out["pred_masks_video_res"]
