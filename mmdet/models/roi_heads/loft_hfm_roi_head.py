# -*- encoding: utf-8 -*-

import copy
import math

import cv2
import numpy as np
import torch

from mmdet.core import bbox2result, bbox2roi
from ..builder import HEADS, build_head, build_roi_extractor
from ..utils import offset_roof_to_footprint
from .standard_roi_head import StandardRoIHead
from .test_mixins import FootprintMaskFromRoofOffsetTestMixin, HeightTestMixin, OffsetTestMixin


@HEADS.register_module()
class LoftHFMRoIHead(  # pylint: disable=abstract-method, too-many-ancestors
    StandardRoIHead,
    OffsetTestMixin,
    HeightTestMixin,
    FootprintMaskFromRoofOffsetTestMixin,
):
    """The base head of all the task-specific head, e.g. offset head, mask head."""

    def __init__(
        self,
        offset_roi_extractor=None,
        offset_head=None,
        height_roi_extractor=None,
        height_head=None,
        footprint_mask_from_roof_offset_head=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if offset_head:
            self.init_offset_head(offset_roi_extractor, offset_head)
            self.offset_expand_feature_num = offset_head.expand_feature_num
        if height_head:
            self.init_height_head(height_roi_extractor, height_head)
        if footprint_mask_from_roof_offset_head:
            self.init_footprint_mask_from_roof_offset_head(footprint_mask_from_roof_offset_head)

        # self.with_vis_feat = False

    def init_offset_head(self, offset_roi_extractor, offset_head):
        """Build offset roi extractor and offset head."""
        self.offset_roi_extractor = build_roi_extractor(offset_roi_extractor)
        self.offset_head = build_head(offset_head)

    def init_height_head(self, height_roi_extractor, height_head):
        """Build height roi extractor and height head."""
        self.height_roi_extractor = build_roi_extractor(height_roi_extractor)
        self.height_head = build_head(height_head)

    def init_footprint_mask_from_roof_offset_head(self, footprint_mask_from_roof_offset_head):
        """Build head that predicts footprint mask from offset and roof mask heads' output."""
        self.footprint_mask_from_roof_offset_head = build_head(footprint_mask_from_roof_offset_head)

    def init_weights(self):
        super().init_weights()
        if self.with_offset:
            self.offset_head.init_weights()
        if self.with_height:
            self.height_head.init_weights()
        if self.with_footprint_mask_from_roof_offset:
            self.footprint_mask_from_roof_offset_head.init_weights()

    @property
    def with_offset(self):
        """bool: whether the RoI head contains a `offset head`"""
        return hasattr(self, "offset_head") and self.offset_head is not None

    @property
    def with_height(self):
        """bool: whether the RoI head contains a `height_head`"""
        return hasattr(self, "height_head") and self.height_head is not None

    @property
    def with_footprint_mask_from_roof_offset(self):
        """bool: whether the RoI head contains a `footprint_mask_from_roof_offset_head`"""
        return (
            hasattr(self, "footprint_mask_from_roof_offset_head")
            and self.footprint_mask_from_roof_offset_head is not None
        )

    def forward_train(  # pylint: disable=arguments-differ
        self,
        x,
        img_metas,
        proposal_list,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        gt_masks=None,
        gt_offsets=None,
        gt_heights=None,
        gt_footprint_masks=None,
        gt_footprint_bboxes=None,
        gt_offset_angles=None,
        loss_offset_angle_consistency=None,
        regular_lambda=None,
        is_semi_supervised_batch=False,
        is_semi_supervised_stage=False,
        is_valid_height_batch=True,
        use_pred_for_offset_angle_consistency=False,
        with_offset_angle_head=False,
        shrunk_losses=None,
        shrunk_factor=1.0,
        footprint_mask_fro_loss_lambda=1.0,
        img=None,
    ):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            proposals (list[Tensors]): list of region proposals.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            gt_heights (None | Tensor): each item are truth heights for each box.

            gt_footprint_masks (None | Tensor): each item are truth footprint mask for each box.

            is_semi_supervised_batch (bool): whether this batch is a semi supervised batch.

            is_semi_supervised_stage (bool): whether in semi-supervised stage or out.

            shrunk_losses(None, set[str]): whether shrink corresponding losses
                when facing a ssl batch.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if shrunk_losses is None:
            shrunk_losses = set()

        # Assign gts and sample proposals.
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            assign_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
                )
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x],
                )
                assign_results.append(assign_result)
                sampling_results.append(sampling_result)

        losses = dict()

        # Bbox head forward without loss.
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(
                x,
                sampling_results,
                gt_bboxes,
                gt_labels,
                img_metas,
            )
            bbox_loss = bbox_results["loss_bbox"]
            if is_semi_supervised_batch:
                if is_semi_supervised_stage:
                    if "bbox" in shrunk_losses:
                        bbox_loss["loss_bbox"] *= shrunk_factor
                    if "cls" in shrunk_losses:
                        bbox_loss["loss_cls"] *= shrunk_factor
                    if not is_valid_height_batch:
                        bbox_loss["loss_bbox"] *= 0.0
                        bbox_loss["loss_cls"] *= 0.0
                else:
                    bbox_loss["loss_bbox"] *= 0.0
                    bbox_loss["loss_cls"] *= 0.0

            losses.update(bbox_loss)

        # Mask head forward and loss.
        if self.with_mask:
            mask_results = self._mask_forward_train(
                x,
                sampling_results,
                bbox_results["bbox_feats"],
                gt_masks,
                img_metas,
            )
            # TOD: Support empty tensor input. #2280
            if mask_results["loss_mask"] is not None:
                mask_loss = mask_results["loss_mask"]
                if is_semi_supervised_batch:
                    if is_semi_supervised_stage:
                        if "mask" in shrunk_losses:
                            mask_loss["loss_mask"] *= shrunk_factor
                        if not is_valid_height_batch:
                            mask_loss["loss_mask"] *= 0.0
                    else:
                        mask_loss["loss_mask"] *= 0.0
                losses.update(mask_loss)

        # Offset head forward and loss.
        if self.with_offset:
            offset_results = self._offset_forward_train(
                x, sampling_results, bbox_results["bbox_feats"], gt_offsets, img_metas
            )
            # TOD: Support empty tensor input. #2280
            if offset_results["loss_offset"] is not None:
                loss_offset = offset_results["loss_offset"]["loss_offset"]
                offset_results["loss_offset"]["loss_offset"] = torch.where(
                    torch.isinf(loss_offset), torch.full_like(loss_offset, 0), loss_offset
                )
                offset_loss = offset_results["loss_offset"]
                if is_semi_supervised_batch:
                    if is_semi_supervised_stage:
                        if "offset" in shrunk_losses:
                            offset_loss["loss_offset"] *= shrunk_factor
                        if not is_valid_height_batch:
                            offset_loss["loss_offset"] *= 0.0
                    else:
                        offset_loss["loss_offset"] *= 0.0

                losses.update(offset_loss)

            instances_cnt = [len(sr.pos_gt_bboxes) for sr in sampling_results]

            if loss_offset_angle_consistency is not None:
                assert gt_offset_angles is not None

                gt_offset_angles_repeated = []
                offsets_pred = []
                start_idx = 0
                for i, cnt in enumerate(instances_cnt):
                    gt_offset_angles_repeated.append(gt_offset_angles[i].repeat((cnt, 1)))
                    offsets_pred.append(
                        offset_results["offset_pred"][start_idx : start_idx + cnt, :]
                    )
                    start_idx += cnt * self.offset_expand_feature_num

                gt_offset_angles_t = torch.cat(gt_offset_angles_repeated, 0)
                offsets_pred_t = torch.cat(offsets_pred, 0)

                gt_offsets_cat = torch.cat(
                    [
                        go[sr.pos_assigned_gt_inds.long(), :]
                        for sr, go in zip(sampling_results, gt_offsets)
                    ],
                    0,
                )
                offsets_pred_norm = offsets_pred_t[:, 0] ** 2 + offsets_pred_t[:, 1] ** 2
                gt_offsets_norm = gt_offsets_cat[:, 0] ** 2 + gt_offsets_cat[:, 1] ** 2
                loss_offsets_norm = loss_offset_angle_consistency(
                    offsets_pred_norm, gt_offsets_norm
                )

                loss_tan_cot = loss_offset_angle_consistency(
                    offsets_pred_t[:, 0] * gt_offset_angles_t[:, 0],
                    offsets_pred_t[:, 1] * gt_offset_angles_t[:, 1],
                )

                loss_ = regular_lambda[0] * loss_offsets_norm + regular_lambda[1] * loss_tan_cot

                if is_semi_supervised_batch:
                    if is_semi_supervised_stage:
                        if with_offset_angle_head:
                            if use_pred_for_offset_angle_consistency:
                                if "offset_angle" in shrunk_losses:
                                    loss_ *= shrunk_factor
                            else:
                                loss_ *= 0.0
                        else:
                            loss_ *= 0.0
                    else:
                        loss_ *= 0.0

                losses.update(loss_offset_angle_consistency=loss_)

        # Footprint mask from roof offset head forward and loss.
        if self.with_offset and self.with_mask and self.with_footprint_mask_from_roof_offset:
            start_idx = 0
            offsets_pred = []
            for i, cnt in enumerate(instances_cnt):
                offsets_pred.append(offset_results["offset_pred"][start_idx : start_idx + cnt, :])
                start_idx += cnt * self.offset_expand_feature_num

            offsets_pred_t = torch.cat(offsets_pred, 0)
            footprint_mask_fro_results = self._footprint_mask_from_roof_offset_forward_train(
                offsets_pred_t,
                mask_results["mask_pred"],
                sampling_results,
                gt_footprint_masks,
                img_metas,
            )
            if footprint_mask_fro_results["loss_footprint_mask_from_roof_offset"] is not None:
                footprint_mask_fro_loss = footprint_mask_fro_results[
                    "loss_footprint_mask_from_roof_offset"
                ]
                if is_semi_supervised_batch:
                    if is_semi_supervised_stage:
                        footprint_mask_fro_loss[
                            "loss_footprint_mask_from_roof_offset"
                        ] *= footprint_mask_fro_loss_lambda
                    else:
                        footprint_mask_fro_loss["loss_footprint_mask_from_roof_offset"] *= 0.0
                losses.update(footprint_mask_fro_loss)

        # Height head forward and loss.
        if self.with_height:
            height_results = self._height_forward_train(
                x, sampling_results, bbox_results["bbox_feats"], gt_heights, img_metas
            )
            if height_results["loss_height"] is not None:
                height_loss = height_results["loss_height"]
                if not is_valid_height_batch:
                    height_loss["loss_height"] *= 0.0
                losses.update(height_loss)

        return losses

    def _calculate_offset_angle_from_offset(self, offsets):
        norm = torch.sqrt(torch.pow(offsets[:, 0], 2) + torch.pow(offsets[:, 1], 2))
        angle = torch.cat(
            (torch.unsqueeze(offsets[:, 1] / norm, 1), torch.unsqueeze(offsets[:, 0] / norm, 1)), 1
        )
        return angle

    def _calculate_offset_from_angle(self, offsets, angles):
        offset_x_from_angle = torch.unsqueeze(
            offsets[:, 1] / (angles[:, 0] + 1e-2) * angles[:, 1], 1
        )
        offset_y_from_angle = torch.unsqueeze(
            offsets[:, 0] / (angles[:, 1] + 1e-2) * angles[:, 0], 1
        )
        return torch.cat((offset_x_from_angle, offset_y_from_angle), 1)

    def _offset_forward_train(self, x, sampling_results, bbox_feats, gt_offsets, img_metas):
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        offset_results = self._offset_forward(x, pos_rois)
        offset_targets = self.offset_head.get_targets(sampling_results, gt_offsets, self.train_cfg)
        loss_offset = self.offset_head.loss(offset_results["offset_pred"], offset_targets)
        offset_results.update(loss_offset=loss_offset, offset_targets=offset_targets)
        return offset_results

    def _offset_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        assert (rois is not None) ^ (pos_inds is not None and bbox_feats is not None)
        if rois is not None:
            offset_feats = self.offset_roi_extractor(
                x[: self.offset_roi_extractor.num_inputs], rois
            )
        else:
            assert bbox_feats is not None
            offset_feats = bbox_feats[pos_inds]

        # self._show_offset_feat(rois, offset_feats)

        offset_pred = self.offset_head(offset_feats)
        offset_results = dict(offset_pred=offset_pred, offset_feats=offset_feats)
        return offset_results

    def _height_forward_train(self, x, sampling_results, bbox_feats, gt_heights, img_metas):
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        height_results = self._height_forward(x, pos_rois)
        height_targets = self.height_head.get_targets(sampling_results, gt_heights, self.train_cfg)
        loss_height = self.height_head.loss(height_results["height_pred"], height_targets)
        height_results.update(loss_height=loss_height, height_targets=height_targets)
        return height_results

    def _height_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        assert (rois is not None) ^ (pos_inds is not None and bbox_feats is not None)
        if rois is not None:
            height_feats = self.height_roi_extractor(
                x[: self.height_roi_extractor.num_inputs], rois
            )
        else:
            assert bbox_feats is not None
            height_feats = bbox_feats[pos_inds]

        height_pred = self.height_head(height_feats)
        height_results = dict(height_pred=height_pred, height_feats=height_feats)
        return height_results

    def _footprint_mask_from_roof_offset_forward_train(
        self, offset_pred, roof_pred, sampling_results, gt_footprint_masks, img_metas
    ):
        """Run forward function and calculate loss for footprint mask from roof offset head."""
        footprint_mask_fro_results = self._footprint_mask_from_roof_offset_forward(
            offset_pred, roof_pred
        )

        footprint_mask_fro_targets = self.footprint_mask_from_roof_offset_head.get_targets(
            sampling_results, gt_footprint_masks, self.train_cfg
        )
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_footprint_mask_ro = self.footprint_mask_from_roof_offset_head.loss(
            footprint_mask_fro_results["footprint_mask_from_roof_offset_pred"],
            footprint_mask_fro_targets,
            pos_labels,
        )

        footprint_mask_fro_results.update(
            loss_footprint_mask_from_roof_offset=loss_footprint_mask_ro,
            footprint_mask_from_roof_offset_targets=footprint_mask_fro_targets,
        )
        return footprint_mask_fro_results

    def _footprint_mask_from_roof_offset_forward(self, offset_pred, roof_pred):
        """Footprint mask from roof offset head forward function used in training and testing."""
        footprint_mask_fro_pred = self.footprint_mask_from_roof_offset_head(offset_pred, roof_pred)
        footprint_mask_fro_results = dict(
            footprint_mask_from_roof_offset_pred=footprint_mask_fro_pred,
        )
        return footprint_mask_fro_results

    def simple_test(
        self,
        x,
        proposal_list,
        img_metas,
        proposals=None,
        rescale=False,
    ):
        """Test without augmentation."""
        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale
        )
        bbox_results = [
            bbox2result(bboxes, labels, self.bbox_head.num_classes)
            for bboxes, labels in zip(det_bboxes, det_labels)
        ]

        if self.with_offset:
            offset_results_ = [
                self.simple_test_offset(x, img_metas, bboxes, labels, rescale=rescale)
                for bboxes, labels in zip(det_bboxes, det_labels)
            ]
            offset_results = [ele[0] for ele in offset_results_]
            instances_cnt = [len(ele[1]) for ele in offset_results_]
            offset_preds = torch.concat(
                [ele[1][: int(cnt / 4), :] for ele, cnt in zip(offset_results_, instances_cnt)]
            )
        else:
            offset_results, offset_preds = [None for _ in det_bboxes], None

        if self.with_mask:
            roof_mask_results, roof_mask_preds = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale
            )
        else:
            roof_mask_results, roof_mask_preds = [None for _ in det_bboxes], None

        if self.with_height:
            height_results = [
                self.simple_test_height(x, img_metas, bboxes, labels, rescale=rescale)
                for bboxes, labels in zip(det_bboxes, det_labels)
            ]
        else:
            height_results = [None for _ in det_bboxes]

        if self.with_offset and self.with_mask and self.with_footprint_mask_from_roof_offset:
            footprint_mask_fro_results = self.simple_test_footprint_mask_fro(
                img_metas, offset_preds, roof_mask_preds, det_bboxes, det_labels
            )
        else:
            footprint_mask_fro_results = [None for _ in det_bboxes]

        footprint_from_roof = (
            offset_roof_to_footprint(offset_results, roof_mask_results, True)
            if self.with_offset and self.with_mask
            else [None for _ in det_bboxes]
        )

        return (
            bbox_results,
            offset_results,
            roof_mask_results,
            height_results,
            footprint_from_roof,
            footprint_mask_fro_results,
        )


def visualize_masks_bboxes(
    img,
    img_metas,
    building_bboxes,
    roof_masks,
    is_semi_supervised_batch=False,
    footprint_bboxes=None,
    footprint_masks=None,
):
    save_path = "tmp/ROI_bboxes_masks_visualization_angle_bbox/"
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float64).reshape(1, -1)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float64).reshape(1, -1)

    img_batch = img.detach().cpu().numpy().copy()
    building_bboxes = [sb.detach().cpu().numpy().astype(np.int32) for sb in building_bboxes]
    roof_masks = [sm.masks.astype(np.uint8) for sm in roof_masks]

    if footprint_bboxes:
        footprint_bboxes = [sb.detach().cpu().numpy().astype(np.int32) for sb in footprint_bboxes]
    else:
        footprint_bboxes = [None for _ in building_bboxes]

    if footprint_masks:
        footprint_masks = [sm.masks.astype(np.uint8) for sm in footprint_masks]
    else:
        footprint_masks = [None for _ in roof_masks]

    for img, sbs, sms, sfbs, sfms, img_meta in zip(
        img_batch, building_bboxes, roof_masks, footprint_bboxes, footprint_masks, img_metas
    ):
        file_name = img_meta["filename"].rsplit("/", 1)[1]
        file_name_split = file_name.rsplit(".", 1)
        file_name_split[0] += "_pseudo_gt" if is_semi_supervised_batch else "_gt"
        file_name = save_path + ".".join((file_name_split[0], file_name_split[1]))

        img_o = np.ascontiguousarray(img.transpose(1, 2, 0))
        cv2.multiply(img_o, std, img_o)
        cv2.add(img_o, mean, img_o)
        cv2.cvtColor(img_o, cv2.COLOR_RGB2BGR, img_o)
        img = img_o.astype(np.uint8).copy()
        img_o = img_o.astype(np.uint8).copy()

        alpha = 0.6
        beta = 1 - alpha

        sms = np.sum(sms, axis=0).astype(np.uint8)
        sms[np.where(sms != 0)] = 1
        sms = np.expand_dims(sms, axis=2).repeat(3, axis=2)
        sms *= np.array([136, 14, 79], dtype=np.uint8)

        cv2.addWeighted(img, alpha, sms, beta, 0, img)
        img = img.astype(np.float64)
        img[np.where(sms != 1)] *= float(4.99999 / 3)
        img = img.astype(np.uint8)

        if sfms is not None:
            sfms = np.sum(sfms, axis=0).astype(np.uint8)
            sfms[np.where(sfms != 0)] = 1
            sfms = np.expand_dims(sfms, axis=2).repeat(3, axis=2)
            sfms *= np.array([0, 180, 0], dtype=np.uint8)

            cv2.addWeighted(img, alpha, sfms, beta, 0, img)
            img = img.astype(np.float64)
            img[np.where(sms != 1)] *= float(4.99999 / 3)
            img = img.astype(np.uint8)

        bbox_pts = [
            np.array([[sb[0], sb[1]], [sb[2], sb[1]], [sb[2], sb[3]], [sb[0], sb[3]]]) for sb in sbs
        ]
        cv2.polylines(img, bbox_pts, True, (180, 0, 0), 2)

        if sfbs is not None:
            f_bbox_pts = [
                np.array([[sb[0], sb[1]], [sb[2], sb[1]], [sb[2], sb[3]], [sb[0], sb[3]]])
                for sb in sfbs
            ]
            cv2.polylines(img, f_bbox_pts, True, (0, 0, 180), 2)

        img = np.concatenate((img, img_o), axis=1)

        cv2.imwrite(file_name, img)
