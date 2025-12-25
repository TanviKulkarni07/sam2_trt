import numpy as np
import torch
from PIL.Image import Image
from typing import List, Optional, Tuple, Union

from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.transforms import SAM2Transforms


class Dataprocessor:
        
    def __init__(self,         
        sam_encoder,
        sam_decoder,
        device,
        mask_threshold=0.0,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
        framework: str = "torch",
        **kwargs,
    ) -> None:
        
        self._transforms = SAM2Transforms(
            resolution=1024,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )

        # Predictor state
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        # Whether the predictor is set for single image or a batch of images
        self._is_batch = False
        # Predictor config
        self.mask_threshold = mask_threshold

        self.sam_encoder = sam_encoder
        self.sam_decoder = sam_decoder

        self.device = device
        self.framework = framework

    @torch.no_grad()
    def set_image(
        self,
        image: Union[np.ndarray, Image],
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
        image (np.ndarray or PIL Image): The input image to embed in RGB format. The image should be in HWC format if np.ndarray, or WHC format if PIL Image
        with pixel values in [0, 255].
        image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        self.reset_predictor()
        # Transform the image to the form expected by the model
        if isinstance(image, np.ndarray):
            self._orig_hw = [image.shape[:2]]
        elif isinstance(image, Image):
            w, h = image.size
            self._orig_hw = [(h, w)]
        else:
            raise NotImplementedError("Image format not supported")

        input_image = self._transforms(image)
        input_image = input_image[None, ...].to(self.device)

        assert (
            len(input_image.shape) == 4 and input_image.shape[1] == 3
        ), f"input_image must be of size 1x3xHxW, got {input_image.shape}"

        if self.framework == "torch":
            self._features = self.sam_encoder(input_image)
        
        elif self.framework == "onnx":
            encoder_inputs = {"image": input_image.detach().cpu().numpy()}
            self._features = self.sam_encoder.run(None, encoder_inputs)

        elif self.framework == "trt":
            encoder_inputs = {"image": input_image}
            self._features = self.sam_encoder.infer(encoder_inputs)

        else:
            raise NotImplementedError(f"Framework '{self.framework}' not supported")

        self._is_image_set = True

    def _prep_prompts(
    self, point_coords, point_labels, box, mask_logits, normalize_coords, img_idx=-1
    ):

        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = torch.as_tensor(
                point_coords, dtype=torch.float, device=self.device
            )
            unnorm_coords = self._transforms.transform_coords(
                point_coords, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )
            labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            if len(unnorm_coords.shape) == 2:
                unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float, device=self.device)
            unnorm_box = self._transforms.transform_boxes(
                box, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )  # Bx2x2
        if mask_logits is not None:
            mask_input = torch.as_tensor(
                mask_logits, dtype=torch.float, device=self.device
            )
            if len(mask_input.shape) == 3:
                mask_input = mask_input[None, :, :, :]
        return mask_input, unnorm_coords, labels, unnorm_box

    def predict(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords = True,
        img_idx: int = -1,
    ):

        mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
            point_coords, point_labels, boxes, mask_input, normalize_coords
        )

        masks, iou_predictions, low_res_masks = self._predict(
            unnorm_coords,
            labels,
            unnorm_box,
            mask_input,
            multimask_output,
            return_logits=return_logits,
        )

        masks_np = masks.squeeze(0).float().detach().cpu().numpy()
        iou_predictions_np = iou_predictions.squeeze(0).float().detach().cpu().numpy()
        low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np
    

    def _predict(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        img_idx: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        if point_coords is not None:
            concat_points = (point_coords, point_labels)
        else:
            concat_points = None

        # Embed prompts
        if boxes is not None:
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
            box_labels = box_labels.repeat(boxes.size(0), 1)
            # we merge "boxes" and "points" into a single "concat_points" input (where
            # boxes are added at the beginning) to sam_prompt_encoder
            if concat_points is not None:
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)

        # Predict masks
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )  # multi object prediction
        if self.framework == "torch":
            low_res_masks, iou_predictions = self.sam_decoder(
                self._features[0], 
                self._features[1], 
                self._features[2], 
                concat_points[0],
                concat_points[1],
                True,
                True,            
            )
        elif self.framework == "onnx":
            decoder_inputs = {
                "image_embed": self._features[0],
                "high_res_feat_0": self._features[1],
                "high_res_feat_1": self._features[2],
                "point_coords": concat_points[0].detach().cpu().numpy(),
                "point_labels": concat_points[1].detach().cpu().numpy(),
            } 
            
            low_res_masks, iou_predictions = self.sam_decoder.run(None, decoder_inputs)
            low_res_masks = torch.from_numpy(low_res_masks).to(self.device)
            iou_predictions = torch.from_numpy(iou_predictions).to(self.device)
        
        elif self.framework == "trt":
            decoder_inputs = {
                "image_embed": self._features["image_embed"],
                "high_res_feat_0": self._features["high_res_feat_0"],
                "high_res_feat_1": self._features["high_res_feat_1"],
                "point_coords": concat_points[0],
                "point_labels": concat_points[1],
            } 

            decoder_outputs = self.sam_decoder.infer(decoder_inputs)
            low_res_masks = decoder_outputs["masks"]
            iou_predictions = decoder_outputs["iou_predictions"]

        else:
            raise NotImplementedError(f"Framework '{self.framework}' not supported")

        # Upscale the masks to the original image resolution
        masks = self._transforms.postprocess_masks(
            low_res_masks, self._orig_hw[img_idx]
        )
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        if not return_logits:
            masks = masks > self.mask_threshold

        return masks, iou_predictions, low_res_masks

    def reset_predictor(self) -> None:
        """
        Resets the image embeddings and other state variables.
        """
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        self._is_batch = False