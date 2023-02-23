import math
import os

import cv2
import mmcv
import numpy as np
import torch
from mmcv.runner import get_dist_info

from ...core.mask.structures import BitmapMasks
from ..builder import DETECTORS, build_head, build_loss
from ..utils import offset_roof_to_footprint
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class LOFT(TwoStageDetector):
    def __init__(
        self,
        backbone,
        rpn_head=None,
        roi_head=None,
        train_cfg=None,
        test_cfg=None,
        neck=None,
        offset_angle_head=None,
        nadir_angle_head=None,
        loss_offset_angle_consistency=None,
        pretrained=None,
        init_cfg=None,
    ):
        super(LOFT, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
        )

        if offset_angle_head is not None:
            self.with_offset_angle_head = True
            self.gt_footprint_mask_as_condition = offset_angle_head.pop(
                "gt_footprint_mask_as_condition", False
            )
            self.gt_footprint_mask_repeat_num = offset_angle_head.pop(
                "gt_footprint_mask_repeat_num", 32
            )
            if self.gt_footprint_mask_as_condition:
                offset_angle_head["in_channels"] += self.gt_footprint_mask_repeat_num
            self.offset_angle_head = build_head(offset_angle_head)
            self.offset_angle_head.init_weights()
        else:
            self.with_offset_angle_head = False

        if loss_offset_angle_consistency is not None:
            self.loss_offset_angle_consistency_regular_lambda = loss_offset_angle_consistency.pop(
                "regular_lambda"
            )
            self.loss_offset_angle_consistency = build_loss(loss_offset_angle_consistency)
        else:
            self.loss_offset_angle_consistency_regular_lambda = None
            self.loss_offset_angle_consistency = None

        if nadir_angle_head is not None:
            self.with_nadir_angle_head = True
            self.gt_height_mask_as_condition = nadir_angle_head.pop(
                "gt_height_mask_as_condition", False
            )
            self.gt_height_mask_repeat_num = nadir_angle_head.pop("gt_height_mask_repeat_num", 32)
            if self.gt_height_mask_as_condition:
                nadir_angle_head["in_channels"] += self.gt_height_mask_repeat_num
            self.nadir_angle_head = build_head(nadir_angle_head)
            self.nadir_angle_head.init_weights()
        else:
            self.with_nadir_angle_head = False

        self.anchor_bbox_vis = [[287, 433, 441, 541]]
        self.with_vis_feat = True

        if train_cfg:
            self.pseudo_rpn_bboxes_wh_ratio = train_cfg.pseudo_rpn_bboxes_wh_ratio
            self.offset_scale = train_cfg.offset_scale
            self.resolution = train_cfg.resolution
            self.shrunk_losses = train_cfg.shrunk_losses
            self.shrunk_factor = train_cfg.shrunk_factor
            self.pseudo_rpn_bbox_scale = train_cfg.pseudo_rpn_bbox_scale
            self.use_pred_for_offset_angle_consistency = (
                train_cfg.use_pred_for_offset_angle_consistency
            )
            self.footprint_mask_fro_loss_lambda = train_cfg.footprint_mask_fro_loss_lambda

    def forward_train(
        self,
        img,
        img_metas,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        gt_masks=None,
        proposals=None,
        gt_offsets=None,
        gt_heights=None,
        gt_height_masks=None,
        height_mask_shape=None,
        gt_footprint_masks=None,
        gt_image_scale_footprint_masks=None,
        image_scale_footprint_mask_shape=None,
        gt_footprint_bboxes=None,
        gt_offset_angles=None,
        gt_nadir_angles=None,
        gt_is_semi_supervised_sample=None,
        gt_is_valid_height_sample=None,
        **kwargs,
    ):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

            gt_offsets (None, list[Tensor]): offsets corresponding to each box

            gt_heights (None, list[Tensor]): heights corresponding to each box

            gt_footprint_masks (None, list[Tensor]): footprint mask corresponding to each box

            gt_footprint_bboxes (None, list[Tensor]): footprint bboxes corresponding to each box

            gt_is_semi_supervised_sample(None, int): whether this is a semi-supervised batch


        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        # Judge whether in semi-supervised stage.
        is_semi_supervised_stage = bool(kwargs.pop("is_semi_supervised_stage")[0][0])

        # Judge whether this batch is a semi-supervised one.
        is_semi_supervised_batch = bool(gt_is_semi_supervised_sample[0][0])

        # Judge whether this batch has valid height annotation.
        is_valid_height_batch = bool(gt_is_valid_height_sample[0][0])

        # Judge whether the pseudo_gt can be used for BP.
        is_valid_pseudo_gt_for_bp = (
            self.with_offset_angle_head and self.with_nadir_angle_head and is_valid_height_batch
        )

        if self.with_offset_angle_head:
            if self.gt_footprint_mask_as_condition:
                image_scale_footprint_mask_shape = (
                    image_scale_footprint_mask_shape[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .copy()
                    .astype(np.int32)
                )
                gt_image_scale_footprint_masks_np = np.stack(
                    [
                        m.resize(image_scale_footprint_mask_shape.tolist()).masks.repeat(32, axis=0)
                        for m in gt_image_scale_footprint_masks
                    ],
                    axis=0,
                )
                gt_image_scale_footprint_masks_t = torch.from_numpy(
                    gt_image_scale_footprint_masks_np
                ).to(x[0].device)
                offset_angle_head_input = torch.cat([x[0], gt_image_scale_footprint_masks_t], dim=1)
            else:
                offset_angle_head_input = x[0]
            offset_angles_pred, offset_angle_loss = self.offset_angle_head.forward_train(
                offset_angle_head_input, img_metas, gt_offset_angles
            )
            if is_semi_supervised_batch:
                offset_angle_loss["loss_offset_angle"] *= 0.0

            losses.update(offset_angle_loss)

        if self.with_nadir_angle_head:
            if self.gt_height_mask_as_condition:
                height_mask_shape = (
                    height_mask_shape[0].detach().cpu().numpy().copy().astype(np.int32)
                )
                gt_height_masks_np = np.stack(
                    [
                        m.resize(height_mask_shape.tolist()).masks.repeat(32, axis=0)
                        for m in gt_height_masks
                    ],
                    axis=0,
                )
                gt_height_masks_t = torch.from_numpy(gt_height_masks_np).to(x[0].device)
                nadir_angle_head_input = torch.cat([x[0], gt_height_masks_t], dim=1)
            else:
                nadir_angle_head_input = x[0]
            nadir_angles_pred, nadir_angle_loss = self.nadir_angle_head.forward_train(
                nadir_angle_head_input, img_metas, gt_nadir_angles
            )

            if is_semi_supervised_batch:
                nadir_angle_loss["loss_nadir_angle"] *= 0.0

            losses.update(nadir_angle_loss)

        # Calculate pseudo_gt_bboxes when training a semi-supervised batch.
        if is_semi_supervised_batch:
            if is_valid_pseudo_gt_for_bp:
                (
                    pseudo_gt_bboxes,
                    pseudo_gt_offsets,
                ) = self._calculate_rpn_pseudo_bboxes_from_angle_and_height(
                    gt_footprint_bboxes,
                    gt_heights,
                    offset_angles_pred,
                    nadir_angles_pred,
                    img_metas,
                )
            else:
                pseudo_gt_bboxes, _ = self._calculate_rpn_pseudo_bboxes_from_scale_up_footprint(
                    gt_footprint_bboxes, img_metas
                )

            gt_bboxes_for_vis = None
        else:
            pseudo_gt_bboxes = None
            gt_bboxes_for_vis = gt_bboxes

        # Visualization.
        if self.with_offset_angle_head:
            offset_angles_for_vis = offset_angles_pred
        else:
            offset_angles_for_vis = None

        if self.with_nadir_angle_head:
            nadir_angles_for_vis = nadir_angles_pred
        else:
            nadir_angles_for_vis = None

        # visualize_bboxes_offset_angles(
        #     img,
        #     img_metas,
        #     gt_bboxes_for_vis,
        #     pseudo_gt_bboxes,
        #     gt_footprint_bboxes,
        #     gt_offset_angles,
        #     offset_angles_for_vis,
        #     gt_nadir_angles,
        #     nadir_angles_for_vis,
        #     is_semi_supervised_batch,
        #     is_valid_height_batch,
        # )

        # Choose the gts for RPN and RoI.
        if is_semi_supervised_batch:
            gt_bboxes_ = pseudo_gt_bboxes
            if is_valid_pseudo_gt_for_bp:
                gt_offsets_ = pseudo_gt_offsets
                gt_masks_ = _offset_footprint_to_roof(pseudo_gt_offsets, gt_footprint_masks)
            else:
                gt_offsets_ = [torch.zeros_like(gt_offset) for gt_offset in gt_offsets]
                gt_masks_ = [gt_mask for gt_mask in gt_masks]
        else:
            gt_bboxes_ = gt_bboxes
            gt_offsets_ = gt_offsets
            gt_masks_ = gt_masks

        # RPN forward and loss.
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get("rpn_proposal", self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes_,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                is_semi_supervised_batch=is_semi_supervised_batch,
                **kwargs,
            )

            # Deal with rpn losses according to config and training samples.
            if is_semi_supervised_batch:
                if is_semi_supervised_stage:
                    # RPN loss should be ignored when training height-invalid batch.
                    if not is_valid_height_batch:
                        rpn_losses["loss_rpn_cls"] = [e * 0.0 for e in rpn_losses["loss_rpn_cls"]]
                        rpn_losses["loss_rpn_bbox"] = [e * 0.0 for e in rpn_losses["loss_rpn_bbox"]]
                    # RPN loss shrinks or not.
                    if "rpn_cls" in self.shrunk_losses:
                        rpn_losses["loss_rpn_cls"] = [
                            e * self.shrunk_factor for e in rpn_losses["loss_rpn_cls"]
                        ]
                    if "rpn_bbox" in self.shrunk_losses:
                        rpn_losses["loss_rpn_bbox"] = [
                            e * self.shrunk_factor for e in rpn_losses["loss_rpn_bbox"]
                        ]
                else:
                    rpn_losses["loss_rpn_cls"] = [e * 0.0 for e in rpn_losses["loss_rpn_cls"]]
                    rpn_losses["loss_rpn_bbox"] = [e * 0.0 for e in rpn_losses["loss_rpn_bbox"]]

            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        if self.with_offset_angle_head:
            if is_semi_supervised_batch:
                offset_angles_for_roi = offset_angles_pred
            else:
                offset_angles_for_roi = gt_offset_angles
        else:
            offset_angles_for_roi = gt_offset_angles

        roi_losses = self.roi_head.forward_train(
            x,
            img_metas,
            proposal_list,
            gt_bboxes_,
            gt_labels,
            gt_bboxes_ignore,
            gt_masks_,
            gt_offsets_,
            gt_heights,
            gt_footprint_masks,
            gt_footprint_bboxes,
            offset_angles_for_roi,
            self.loss_offset_angle_consistency,
            self.loss_offset_angle_consistency_regular_lambda,
            is_semi_supervised_batch,
            is_semi_supervised_stage,
            is_valid_height_batch,
            self.use_pred_for_offset_angle_consistency,
            self.with_offset_angle_head,
            self.shrunk_losses,
            self.shrunk_factor,
            self.footprint_mask_fro_loss_lambda,
            img,
            **kwargs,
        )

        losses.update(roi_losses)

        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, "Bbox head must be implemented."
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals
        offset_angle = (
            self.offset_angle_head.simple_test(x[0]).cpu().numpy()
            if self.with_offset_angle_head
            else [None for _ in img_metas]
        )
        nadir_angle = (
            self.nadir_angle_head.simple_test(x[0]).cpu().numpy()
            if self.with_nadir_angle_head
            else [None for _ in img_metas]
        )
        original_results = self.roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)
        original_results = list(original_results)
        original_results.append(offset_angle)
        original_results.append(nadir_angle)
        return tuple(original_results)

    def _calculate_rpn_pseudo_bboxes_from_scale_up_footprint(self, gt_footprint_bboxes, img_metas):
        device = gt_footprint_bboxes[0].device
        pseudo_gt_bboxes = []
        for bboxes, img_meta in zip(gt_footprint_bboxes, img_metas):
            # Get the box boundary of each footprint bbox.
            footprint_x_tl = torch.unsqueeze(bboxes[:, 0], 1)
            footprint_y_tl = torch.unsqueeze(bboxes[:, 1], 1)
            footprint_x_br = torch.unsqueeze(bboxes[:, 2], 1)
            footprint_y_br = torch.unsqueeze(bboxes[:, 3], 1)

            # Get the wh of each footprint bbox.
            footprint_w = footprint_x_br - footprint_x_tl
            footprint_h = footprint_y_br - footprint_y_tl

            # Calculate the center of each footprint bbox.
            footprint_x_c = (footprint_x_br + footprint_x_tl) / 2
            footprint_y_c = (footprint_y_br + footprint_y_tl) / 2

            # The enlarged footprint bbox with tl point fixed.
            bbox_x_tl = footprint_x_c - footprint_w * self.pseudo_rpn_bboxes_wh_ratio[0] / 2
            bbox_y_tl = footprint_y_c - footprint_h * self.pseudo_rpn_bboxes_wh_ratio[1] / 2
            bbox_x_br = footprint_x_c + footprint_w * self.pseudo_rpn_bboxes_wh_ratio[0] / 2
            bbox_y_br = footprint_y_c + footprint_h * self.pseudo_rpn_bboxes_wh_ratio[1] / 2

            # Calculate the image boundary of each image.
            img_x_min = torch.zeros(footprint_x_tl.shape, device=device)
            img_y_min = torch.zeros(footprint_y_tl.shape, device=device)
            img_x_max = torch.ones(footprint_x_tl.shape, device=device)
            img_y_max = torch.ones(footprint_y_tl.shape, device=device)

            img_x_max *= img_meta["img_shape"][1]
            img_y_max *= img_meta["img_shape"][0]

            # Clip the building bbox with image.
            bbox_x_tl = torch.max(torch.cat((bbox_x_tl, img_x_min), 1), 1, keepdim=True)[0]
            bbox_y_tl = torch.max(torch.cat((bbox_y_tl, img_y_min), 1), 1, keepdim=True)[0]
            bbox_x_br = torch.min(torch.cat((bbox_x_br, img_x_max), 1), 1, keepdim=True)[0]
            bbox_y_br = torch.min(torch.cat((bbox_y_br, img_y_max), 1), 1, keepdim=True)[0]

            pseudo_gt_bboxes.append(torch.concat((bbox_x_tl, bbox_y_tl, bbox_x_br, bbox_y_br), 1))

        return pseudo_gt_bboxes, pseudo_gt_bboxes

    def _calculate_rpn_pseudo_bboxes_from_angle_and_height(
        self, gt_footprint_bboxes, gt_heights, offset_angles, nadir_angles, img_metas
    ):
        device = gt_footprint_bboxes[0].device
        pseudo_gt_bboxes = []
        pseudo_gt_roof_bboxes = []
        pseudo_gt_offsets = []
        for bboxes, heights, o_angle, n_angle, img_meta in zip(
            gt_footprint_bboxes, gt_heights, offset_angles, nadir_angles, img_metas
        ):
            o_angle = o_angle.detach()
            n_angle = n_angle.detach()

            # Get the box boundary of each footprint bbox.
            footprint_x_tl = torch.unsqueeze(bboxes[:, 0], 1)
            footprint_y_tl = torch.unsqueeze(bboxes[:, 1], 1)
            footprint_x_br = torch.unsqueeze(bboxes[:, 2], 1)
            footprint_y_br = torch.unsqueeze(bboxes[:, 3], 1)

            # Calculate the offset.
            offset_norm = heights * n_angle / self.resolution
            offset_norm /= math.sqrt(o_angle[0] ** 2 + o_angle[1] ** 2)
            offset_norm *= self.offset_scale
            offset_x = offset_norm * o_angle[1]
            offset_y = offset_norm * o_angle[0]
            offset = torch.cat((offset_x, offset_y), dim=1)

            # Get the box boundary of each roof bbox.
            roof_x_tl = footprint_x_tl + offset_x
            roof_y_tl = footprint_y_tl + offset_y
            roof_x_br = footprint_x_br + offset_x
            roof_y_br = footprint_y_br + offset_y

            # Get the box center of each roof bbox.
            roof_x_c = (roof_x_tl + roof_x_br) / 2
            roof_y_c = (roof_y_tl + roof_y_br) / 2

            # Get the wh of each roof bbox.
            roof_w = roof_x_br - roof_x_tl
            roof_h = roof_y_br - roof_y_tl

            # scale the box boundary of each roof bbox.
            roof_x_tl = roof_x_c - roof_w / 2 * self.pseudo_rpn_bbox_scale
            roof_x_br = roof_x_c + roof_w / 2 * self.pseudo_rpn_bbox_scale
            roof_y_tl = roof_y_c - roof_h / 2 * self.pseudo_rpn_bbox_scale
            roof_y_br = roof_y_c + roof_h / 2 * self.pseudo_rpn_bbox_scale

            # Get the box boundary of each building bbox.
            bbox_x_tl = torch.min(torch.cat((footprint_x_tl, roof_x_tl), 1), 1, keepdim=True)[0]
            bbox_y_tl = torch.min(torch.cat((footprint_y_tl, roof_y_tl), 1), 1, keepdim=True)[0]
            bbox_x_br = torch.max(torch.cat((footprint_x_br, roof_x_br), 1), 1, keepdim=True)[0]
            bbox_y_br = torch.max(torch.cat((footprint_y_br, roof_y_br), 1), 1, keepdim=True)[0]

            # Calculate the image boundary of each image.
            img_x_min = torch.zeros(footprint_x_tl.shape, device=device)
            img_y_min = torch.zeros(footprint_y_tl.shape, device=device)
            img_x_max = torch.ones(footprint_x_tl.shape, device=device)
            img_y_max = torch.ones(footprint_y_tl.shape, device=device)

            img_x_max *= img_meta["img_shape"][1]
            img_y_max *= img_meta["img_shape"][0]

            # Clip the building bbox with image.
            bbox_x_tl = torch.max(torch.cat((bbox_x_tl, img_x_min), 1), 1, keepdim=True)[0]
            bbox_y_tl = torch.max(torch.cat((bbox_y_tl, img_y_min), 1), 1, keepdim=True)[0]
            bbox_x_br = torch.min(torch.cat((bbox_x_br, img_x_max), 1), 1, keepdim=True)[0]
            bbox_y_br = torch.min(torch.cat((bbox_y_br, img_y_max), 1), 1, keepdim=True)[0]

            pseudo_gt_bboxes.append(torch.concat((bbox_x_tl, bbox_y_tl, bbox_x_br, bbox_y_br), 1))
            pseudo_gt_roof_bboxes.append(
                torch.concat((roof_x_tl, roof_y_tl, roof_x_br, roof_y_br), 1)
            )
            pseudo_gt_offsets.append(offset)

        return pseudo_gt_bboxes, pseudo_gt_offsets

    def show_result(
        self,
        img,
        result,
        score_thr=0.8,
        bbox_color="green",
        text_color="green",
        thickness=1,
        font_scale=0.5,
        win_name="",
        show=False,
        wait_time=0,
        out_file=None,
    ):
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            if self.with_vis_feat:
                bbox_result, segm_result, offset, offset_features = result
            else:
                bbox_result, segm_result, offset = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        if isinstance(offset, tuple):
            offsets = offset[0]
        else:
            offsets = offset

        # rotate offset
        # offsets = self.offset_rotate(offsets, 0)

        bboxes = np.vstack(bbox_result)
        scores = bboxes[:, -1]
        bboxes = bboxes[:, 0:-1]

        w, h = bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]
        area = w * h
        # valid_inds = np.argsort(area, axis=0)[::-1].squeeze()
        valid_inds = np.where(np.sqrt(area) > 50)[0]

        if segm_result is not None:  # non empty
            segms = mmcv.concat_list(segm_result)
            inds = np.where(scores > 0.4)[0][:]

            masks = []
            offset_results = []
            bbox_results = []
            offset_feats = []
            for i in inds:
                if i not in valid_inds:
                    continue
                mask = segms[i]
                offset = offsets[i]
                if self.with_vis_feat:
                    offset_feat = offset_features[i]
                else:
                    offset_feat = []
                bbox = bboxes[i]

                gray = np.array(mask * 255, dtype=np.uint8)

                contours = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours[0] if len(contours) == 2 else contours[1]

                if contours != []:
                    cnt = max(contours, key=cv2.contourArea)
                    mask = np.array(cnt).reshape(1, -1).tolist()[0]
                else:
                    continue

                masks.append(mask)
                offset_results.append(offset)
                bbox_results.append(bbox)
                offset_feats.append(offset_feat)

    def offset_coordinate_transform(self, offset, transform_flag="xy2la"):
        """transform the coordinate of offsets

        Args:
            offset (list): list of offset
            transform_flag (str, optional): flag of transform. Defaults to 'xy2la'.

        Raises:
            NotImplementedError: [description]

        Returns:
            list: transformed offsets
        """
        if transform_flag == "xy2la":
            offset_x, offset_y = offset
            length = math.sqrt(offset_x**2 + offset_y**2)
            angle = math.atan2(offset_y, offset_x)
            offset = [length, angle]
        elif transform_flag == "la2xy":
            length, angle = offset
            offset_x = length * np.cos(angle)
            offset_y = length * np.sin(angle)
            offset = [offset_x, offset_y]
        else:
            raise NotImplementedError

        return offset

    def offset_rotate(self, offsets, rotate_angle):
        offsets = [
            self.offset_coordinate_transform(offset, transform_flag="xy2la") for offset in offsets
        ]

        offsets = [[offset[0], offset[1] + rotate_angle * np.pi / 180.0] for offset in offsets]

        offsets = [
            self.offset_coordinate_transform(offset, transform_flag="la2xy") for offset in offsets
        ]

        return np.array(offsets, dtype=np.float32)


def visualize_bboxes_offset_angles(
    img,
    img_metas,
    gt_bboxes,
    pseudo_gt_bboxes,
    footprint_bboxes,
    gt_offset_angles,
    pred_offset_angles,
    gt_nadir_angles,
    pred_nadir_angles,
    is_semi_supervised_batch,
    is_valid_height_batch,
):
    """The func for visualizing gt or pseudo bboxes or angles."""
    save_path = "tmp/RPN_bboxes_visualization_" + "4/"
    rank, _ = get_dist_info()
    if rank == 0:
        if not os.path.exists(save_path):
            os.mkdir(save_path)

    mean = np.array([123.675, 116.28, 103.53], dtype=np.float64).reshape(1, -1)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float64).reshape(1, -1)

    img_batch = img.detach().cpu().numpy().copy()

    footprint_bboxes = [sb.detach().cpu().numpy().astype(np.int32) for sb in footprint_bboxes]

    if gt_bboxes:
        gt_bboxes = [sb.detach().cpu().numpy().astype(np.int32) for sb in gt_bboxes]
    else:
        gt_bboxes = [None for _ in footprint_bboxes]

    if pseudo_gt_bboxes:
        pseudo_gt_bboxes = [sb.detach().cpu().numpy().astype(np.int32) for sb in pseudo_gt_bboxes]
    else:
        pseudo_gt_bboxes = [None for _ in footprint_bboxes]

    if pred_offset_angles is None:
        pred_offset_angles = [None for _ in footprint_bboxes]

    if pred_nadir_angles is None:
        pred_nadir_angles = [None for _ in footprint_bboxes]

    for img, sbs, spbs, sfbs, goa, poa, gna, pna, img_meta in zip(
        img_batch,
        gt_bboxes,
        pseudo_gt_bboxes,
        footprint_bboxes,
        gt_offset_angles,
        pred_offset_angles,
        gt_nadir_angles,
        pred_nadir_angles,
        img_metas,
    ):
        img = np.ascontiguousarray(img.transpose(1, 2, 0))
        cv2.multiply(img, std, img)
        cv2.add(img, mean, img)
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)
        img = img.astype(np.uint8).copy()

        if sbs is not None:
            pts = [
                np.array([[sb[0], sb[1]], [sb[2], sb[1]], [sb[2], sb[3]], [sb[0], sb[3]]])
                for sb in sbs
            ]
            cv2.polylines(img, pts, True, (144, 238, 144), 2)

        if spbs is not None:
            pts = [
                np.array([[sb[0], sb[1]], [sb[2], sb[1]], [sb[2], sb[3]], [sb[0], sb[3]]])
                for sb in spbs
            ]
            cv2.polylines(img, pts, True, (0, 0, 200), 2)

        pts = [
            np.array([[sb[0], sb[1]], [sb[2], sb[1]], [sb[2], sb[3]], [sb[0], sb[3]]])
            for sb in sfbs
        ]
        cv2.polylines(img, pts, True, (200, 0, 0), 2)

        canvas = np.zeros_like(img, dtype=np.uint8)
        go = _get_offset_from_offset_angle(goa[0])
        cv2.polylines(canvas, go, True, (0, 0, 255), 4)
        if poa is not None:
            po = _get_offset_from_offset_angle(poa)
            cv2.polylines(canvas, po, True, (0, 255, 0), 4)
        gn = _get_offset_from_nadir_angle(gna[0])
        cv2.polylines(canvas, gn, True, (0, 255, 255), 4)
        if pna is not None:
            pn = _get_offset_from_nadir_angle(pna)
            cv2.polylines(canvas, pn, True, (128, 0, 0), 4)
        cv2.circle(canvas, (512, 512), 500, (255, 255, 255))

        img = np.concatenate((img, canvas), axis=1)

        file_name = img_meta["filename"].rsplit("/", 1)[1]
        file_name_split = file_name.rsplit(".", 1)
        file_name_split[0] += "_pseudo" if is_semi_supervised_batch else "_gt"
        file_name_split[0] += "_vh" if is_valid_height_batch else "_ih"
        file_name = save_path + ".".join((file_name_split[0], file_name_split[1]))
        cv2.imwrite(file_name, img)


def _get_offset_from_offset_angle(angle):
    angle = angle.detach().cpu().numpy().astype(np.float32)
    start_x, start_y = [512, 512]
    norm = 500
    offset_y, offset_x = angle[0] * norm, angle[1] * norm
    stop_point = [start_x + offset_x, start_y + offset_y]
    return [np.array([[start_x, start_y], stop_point], dtype=np.int32)]


def _get_offset_from_nadir_angle(angle):
    angle = angle.detach().cpu().numpy().astype(np.float32)
    start_x, start_y = [512, 512]
    norm = 500
    angle_norm = math.sqrt(angle**2 + 1.0)
    angle = [angle / angle_norm, 1.0 / angle_norm]
    offset_y, offset_x = angle[0] * norm, angle[1] * norm
    stop_point = [start_x + offset_x, start_y + offset_y]
    return [np.array([[start_x, start_y], stop_point], dtype=np.int32)]


def _offset_footprint_to_roof(offsets, footprints):
    footprints_ = [[[e for e in m.masks]] for m in footprints]
    h, w = footprints_[0][0][0].shape
    roofs = offset_roof_to_footprint(offsets, footprints_)
    return [BitmapMasks(np.array(r)[0], h, w) for r in roofs]
