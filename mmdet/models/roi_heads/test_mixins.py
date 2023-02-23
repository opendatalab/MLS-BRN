# Copyright (c) OpenMMLab. All rights reserved.
import sys
import warnings

import numpy as np
import torch

from mmdet.core import bbox2roi, bbox_mapping, merge_aug_bboxes, merge_aug_masks, multiclass_nms

if sys.version_info >= (3, 7):
    from mmdet.utils.contextmanagers import completed


class BBoxTestMixin:
    if sys.version_info >= (3, 7):

        async def async_test_bboxes(
            self, x, img_metas, proposals, rcnn_test_cfg, rescale=False, **kwargs
        ):
            """Asynchronized test for box head without augmentation."""
            rois = bbox2roi(proposals)
            roi_feats = self.bbox_roi_extractor(
                x[: len(self.bbox_roi_extractor.featmap_strides)], rois
            )
            if self.with_shared_head:
                roi_feats = self.shared_head(roi_feats)
            sleep_interval = rcnn_test_cfg.get("async_sleep_interval", 0.017)

            async with completed(__name__, "bbox_head_forward", sleep_interval=sleep_interval):
                cls_score, bbox_pred = self.bbox_head(roi_feats)

            img_shape = img_metas[0]["img_shape"]
            scale_factor = img_metas[0]["scale_factor"]
            det_bboxes, det_labels = self.bbox_head.get_bboxes(
                rois,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=rescale,
                cfg=rcnn_test_cfg,
            )
            return det_bboxes, det_labels

    def simple_test_bboxes(self, x, img_metas, proposals, rcnn_test_cfg, rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """

        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0,), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros((0, self.bbox_head.fc_cls.out_features))
            # There is no proposal in the whole batch
            return [det_bbox] * batch_size, [det_label] * batch_size

        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta["img_shape"] for meta in img_metas)
        scale_factors = tuple(meta["scale_factor"] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results["cls_score"]
        bbox_pred = bbox_results["bbox_pred"]
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None,) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0,), dtype=torch.long)
                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros((0, self.bbox_head.fc_cls.out_features))

            else:
                det_bbox, det_label = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg,
                )
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels

    def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
        """Test det bboxes with test time augmentation."""
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]["img_shape"]
            scale_factor = img_meta[0]["scale_factor"]
            flip = img_meta[0]["flip"]
            flip_direction = img_meta[0]["flip_direction"]
            # TODO more flexible
            proposals = bbox_mapping(
                proposal_list[0][:, :4], img_shape, scale_factor, flip, flip_direction
            )
            rois = bbox2roi([proposals])
            bbox_results = self._bbox_forward(x, rois)
            bboxes, scores = self.bbox_head.get_bboxes(
                rois,
                bbox_results["cls_score"],
                bbox_results["bbox_pred"],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None,
            )
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg
        )
        if merged_bboxes.shape[0] == 0:
            # There is no proposal in the single image
            det_bboxes = merged_bboxes.new_zeros(0, 5)
            det_labels = merged_bboxes.new_zeros((0,), dtype=torch.long)
        else:
            det_bboxes, det_labels = multiclass_nms(
                merged_bboxes,
                merged_scores,
                rcnn_test_cfg.score_thr,
                rcnn_test_cfg.nms,
                rcnn_test_cfg.max_per_img,
            )
        return det_bboxes, det_labels


class MaskTestMixin:
    if sys.version_info >= (3, 7):

        async def async_test_mask(
            self, x, img_metas, det_bboxes, det_labels, rescale=False, mask_test_cfg=None
        ):
            """Asynchronized test for mask head without augmentation."""
            # image shape of the first image in the batch (only one)
            ori_shape = img_metas[0]["ori_shape"]
            scale_factor = img_metas[0]["scale_factor"]
            if det_bboxes.shape[0] == 0:
                segm_result = [[] for _ in range(self.mask_head.num_classes)]
            else:
                if rescale and not isinstance(scale_factor, (float, torch.Tensor)):
                    scale_factor = det_bboxes.new_tensor(scale_factor)
                _bboxes = det_bboxes[:, :4] * scale_factor if rescale else det_bboxes
                mask_rois = bbox2roi([_bboxes])
                mask_feats = self.mask_roi_extractor(
                    x[: len(self.mask_roi_extractor.featmap_strides)], mask_rois
                )

                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
                if mask_test_cfg and mask_test_cfg.get("async_sleep_interval"):
                    sleep_interval = mask_test_cfg["async_sleep_interval"]
                else:
                    sleep_interval = 0.035
                async with completed(__name__, "mask_head_forward", sleep_interval=sleep_interval):
                    mask_pred = self.mask_head(mask_feats)
                segm_result = self.mask_head.get_seg_masks(
                    mask_pred, _bboxes, det_labels, self.test_cfg, ori_shape, scale_factor, rescale
                )
            return segm_result

    def simple_test_mask(self, x, img_metas, det_bboxes, det_labels, rescale=False):
        """Simple test for mask head without augmentation."""
        # image shapes of images in the batch
        ori_shapes = tuple(meta["ori_shape"] for meta in img_metas)
        scale_factors = tuple(meta["scale_factor"] for meta in img_metas)

        if isinstance(scale_factors[0], float):
            warnings.warn(
                "Scale factor in img_metas should be a "
                "ndarray with shape (4,) "
                "arrange as (factor_w, factor_h, factor_w, factor_h), "
                "The scale_factor with float type has been deprecated. "
            )
            scale_factors = np.array([scale_factors] * 4, dtype=np.float32)

        num_imgs = len(det_bboxes)
        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            segm_results = [
                [[] for _ in range(self.mask_head.num_classes)] for _ in range(num_imgs)
            ]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale:
                scale_factors = [
                    torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                    for scale_factor in scale_factors
                ]
            _bboxes = [
                det_bboxes[i][:, :4] * scale_factors[i] if rescale else det_bboxes[i][:, :4]
                for i in range(len(det_bboxes))
            ]
            ori_shapes = tuple(meta["ori_shape"] for meta in img_metas)
            mask_rois = bbox2roi(_bboxes)
            mask_results = self._mask_forward(x, mask_rois)
            mask_pred = mask_results["mask_pred"]
            # split batch mask prediction back to each image
            num_mask_roi_per_img = [len(det_bbox) for det_bbox in det_bboxes]
            mask_preds = mask_pred.split(num_mask_roi_per_img, 0)

            # apply mask post-processing to each image individually
            segm_results = []
            for i in range(num_imgs):
                if det_bboxes[i].shape[0] == 0:
                    segm_results.append([[] for _ in range(self.mask_head.num_classes)])
                else:
                    segm_result = self.mask_head.get_seg_masks(
                        mask_preds[i],
                        _bboxes[i],
                        det_labels[i],
                        self.test_cfg,
                        ori_shapes[i],
                        scale_factors[i],
                        rescale,
                    )
                    segm_results.append(segm_result)
        return segm_results, mask_pred

    def aug_test_mask(self, feats, img_metas, det_bboxes, det_labels):
        """Test for mask head with test time augmentation."""
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes)]
        else:
            aug_masks = []
            for x, img_meta in zip(feats, img_metas):
                img_shape = img_meta[0]["img_shape"]
                scale_factor = img_meta[0]["scale_factor"]
                flip = img_meta[0]["flip"]
                flip_direction = img_meta[0]["flip_direction"]
                _bboxes = bbox_mapping(
                    det_bboxes[:, :4], img_shape, scale_factor, flip, flip_direction
                )
                mask_rois = bbox2roi([_bboxes])
                mask_results = self._mask_forward(x, mask_rois)
                # convert to numpy array to save memory
                aug_masks.append(mask_results["mask_pred"].sigmoid().cpu().numpy())
            merged_masks = merge_aug_masks(aug_masks, img_metas, self.test_cfg)

            ori_shape = img_metas[0][0]["ori_shape"]
            scale_factor = det_bboxes.new_ones(4)
            segm_result = self.mask_head.get_seg_masks(
                merged_masks,
                det_bboxes,
                det_labels,
                self.test_cfg,
                ori_shape,
                scale_factor=scale_factor,
                rescale=False,
            )
        return segm_result


class FootprintMaskFromRoofOffsetTestMixin:
    def simple_test_footprint_mask_fro(
        self, img_metas, offsets, roof_masks, det_bboxes, det_labels, rescale=False
    ):
        footprint_mask_fro_results = self._footprint_mask_from_roof_offset_forward(
            offsets, roof_masks
        )
        footprint_mask_fro_pred = footprint_mask_fro_results["footprint_mask_from_roof_offset_pred"]
        # split batch mask prediction back to each image
        num_mask_roi_per_img = [len(det_bbox) for det_bbox in det_bboxes]
        footprint_mask_fro_preds = footprint_mask_fro_pred.split(num_mask_roi_per_img, 0)

        # apply mask post-processing to each image individually
        segm_results = []
        _bboxes = [det_bboxes[i][:, :4] for i in range(len(det_bboxes))]
        ori_shapes = tuple(meta["ori_shape"] for meta in img_metas)
        scale_factors = tuple(meta["scale_factor"] for meta in img_metas)
        for i in range(len(det_bboxes)):
            if det_bboxes[i].shape[0] == 0:
                segm_results.append([[] for _ in range(self.mask_head.num_classes)])
            else:
                segm_result = self.footprint_mask_from_roof_offset_head.get_seg_masks(
                    footprint_mask_fro_preds[i],
                    _bboxes[i],
                    det_labels[i],
                    self.test_cfg,
                    ori_shapes[i],
                    scale_factors[i],
                    rescale,
                )
                segm_results.append(segm_result)
        return segm_results


class FootprintMaskTestMixin:
    if sys.version_info >= (3, 7):

        async def async_test_footprint_mask(
            self, x, img_metas, det_bboxes, det_labels, rescale=False, mask_test_cfg=None
        ):
            """Asynchronized test for footprint mask head without augmentation."""
            # image shape of the first image in the batch (only one)
            ori_shape = img_metas[0]["ori_shape"]
            scale_factor = img_metas[0]["scale_factor"]
            if det_bboxes.shape[0] == 0:
                footprint_segm_result = [[] for _ in range(self.footprint_mask_head.num_classes)]
            else:
                if rescale and not isinstance(scale_factor, (float, torch.Tensor)):
                    scale_factor = det_bboxes.new_tensor(scale_factor)
                _bboxes = det_bboxes[:, :4] * scale_factor if rescale else det_bboxes
                footprint_mask_rois = bbox2roi([_bboxes])
                footprint_mask_feats = self.mask_roi_extractor(
                    x[: len(self.footprint_mask_roi_extractor.featmap_strides)], footprint_mask_rois
                )

                if self.with_shared_head:
                    footprint_mask_feats = self.shared_head(footprint_mask_feats)
                if mask_test_cfg and mask_test_cfg.get("async_sleep_interval"):
                    sleep_interval = mask_test_cfg["async_sleep_interval"]
                else:
                    sleep_interval = 0.035
                async with completed(
                    __name__, "footprint_mask_head_forward", sleep_interval=sleep_interval
                ):
                    footprint_mask_pred = self.footprint_mask_head(footprint_mask_feats)
                footprint_segm_result = self.footprint_mask_head.get_seg_masks(
                    footprint_mask_pred,
                    _bboxes,
                    det_labels,
                    self.test_cfg,
                    ori_shape,
                    scale_factor,
                    rescale,
                )
            return footprint_segm_result

    def simple_test_footprint_mask(self, x, img_metas, det_bboxes, det_labels, rescale=False):
        """Simple test for footprint mask head without augmentation."""
        # image shapes of images in the batch
        ori_shapes = tuple(meta["ori_shape"] for meta in img_metas)
        scale_factors = tuple(meta["scale_factor"] for meta in img_metas)

        if isinstance(scale_factors[0], float):
            warnings.warn(
                "Scale factor in img_metas should be a "
                "ndarray with shape (4,) "
                "arrange as (factor_w, factor_h, factor_w, factor_h), "
                "The scale_factor with float type has been deprecated. "
            )
            scale_factors = np.array([scale_factors] * 4, dtype=np.float32)

        num_imgs = len(det_bboxes)
        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            footprint_segm_results = [
                [[] for _ in range(self.footprint_mask_head.num_classes)] for _ in range(num_imgs)
            ]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale:
                scale_factors = [
                    torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                    for scale_factor in scale_factors
                ]
            _bboxes = [
                det_bboxes[i][:, :4] * scale_factors[i] if rescale else det_bboxes[i][:, :4]
                for i in range(len(det_bboxes))
            ]
            footprint_mask_rois = bbox2roi(_bboxes)
            footprint_mask_results = self._footprint_mask_forward(x, footprint_mask_rois)
            footprint_mask_pred = footprint_mask_results["footprint_mask_pred"]
            # split batch mask prediction back to each image
            num_mask_roi_per_img = [len(det_bbox) for det_bbox in det_bboxes]
            footprint_mask_preds = footprint_mask_pred.split(num_mask_roi_per_img, 0)

            # apply mask post-processing to each image individually
            footprint_segm_results = []
            for i in range(num_imgs):
                if det_bboxes[i].shape[0] == 0:
                    footprint_segm_results.append(
                        [[] for _ in range(self.footprint_mask_head.num_classes)]
                    )
                else:
                    footprint_segm_result = self.footprint_mask_head.get_seg_masks(
                        footprint_mask_preds[i],
                        _bboxes[i],
                        det_labels[i],
                        self.test_cfg,
                        ori_shapes[i],
                        scale_factors[i],
                        rescale,
                    )
                    footprint_segm_results.append(footprint_segm_result)
        return footprint_segm_results

    def aug_test_footprint_mask(self, feats, img_metas, det_bboxes, det_labels):
        """Test for footprint mask head with test time augmentation."""
        if det_bboxes.shape[0] == 0:
            footprint_segm_result = [[] for _ in range(self.footprint_mask_head.num_classes)]
        else:
            aug_footprint_masks = []
            for x, img_meta in zip(feats, img_metas):
                img_shape = img_meta[0]["img_shape"]
                scale_factor = img_meta[0]["scale_factor"]
                flip = img_meta[0]["flip"]
                flip_direction = img_meta[0]["flip_direction"]
                _bboxes = bbox_mapping(
                    det_bboxes[:, :4], img_shape, scale_factor, flip, flip_direction
                )
                footprint_mask_rois = bbox2roi([_bboxes])
                footprint_mask_results = self._footprint_mask_forward(x, footprint_mask_rois)
                # convert to numpy array to save memory
                aug_footprint_masks.append(
                    footprint_mask_results["mask_pred"].sigmoid().cpu().numpy()
                )
            merged_footprint_masks = merge_aug_masks(aug_footprint_masks, img_metas, self.test_cfg)

            ori_shape = img_metas[0][0]["ori_shape"]
            scale_factor = det_bboxes.new_ones(4)
            footprint_segm_result = self.footprint_mask_head.get_seg_masks(
                merged_footprint_masks,
                det_bboxes,
                det_labels,
                self.test_cfg,
                ori_shape,
                scale_factor=scale_factor,
                rescale=False,
            )
        return footprint_segm_result


class OffsetTestMixin(object):
    def simple_test_offset(self, x, img_metas, det_bboxes, det_labels, rescale=False):
        # image shape of the first image in the batch (only one)
        ori_shape = img_metas[0]["ori_shape"]
        scale_factor = img_metas[0]["scale_factor"]
        if det_bboxes.shape[0] == 0:
            offset_result = [[] for _ in range(2)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(det_bboxes.device)
            _bboxes = det_bboxes[:, :4] * scale_factor if rescale else det_bboxes
            offset_rois = bbox2roi([_bboxes])
            offset_feats = self.offset_roi_extractor(
                x[: len(self.offset_roi_extractor.featmap_strides)], offset_rois
            )

            offset_pred = self.offset_head(offset_feats)
            offset_result = self.offset_head.get_offsets(
                offset_pred, _bboxes, scale_factor, rescale
            )
        return offset_result, offset_pred

    def simple_test_offset_rotate_feature(
        self, x, img_metas, det_bboxes, det_labels, rescale=False, with_rotate=False, rotate_angle=0
    ):
        # image shape of the first image in the batch (only one)
        ori_shape = img_metas[0]["ori_shape"]
        scale_factor = img_metas[0]["scale_factor"]
        if det_bboxes.shape[0] == 0:
            offset_result = [[] for _ in range(2)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(det_bboxes.device)
            _bboxes = det_bboxes[:, :4] * scale_factor if rescale else det_bboxes
            offset_rois = bbox2roi([_bboxes])
            offset_feats = self.offset_roi_extractor(
                x[: len(self.offset_roi_extractor.featmap_strides)], offset_rois
            )

            # self._show_offset_feat(offset_rois, offset_feats)
            if with_rotate:
                theta = torch.empty((offset_feats.size()[0], 2, 3), device=offset_feats.device)

                angle = rotate_angle * np.pi / 180.0

                theta[:, 0, 0] = torch.tensor(np.cos(angle), device=offset_feats.device)
                theta[:, 0, 1] = torch.tensor(np.sin(-angle), device=offset_feats.device)
                theta[:, 1, 0] = torch.tensor(np.sin(angle), device=offset_feats.device)
                theta[:, 1, 1] = torch.tensor(np.cos(angle), device=offset_feats.device)

                grid = F.affine_grid(theta, offset_feats.size())
                offset_feats = F.grid_sample(offset_feats, grid)

            offset_pred = self.offset_head(offset_feats)

            offset_result = self.offset_head.get_offsets(
                offset_pred, _bboxes, scale_factor, rescale
            )

            self.vis_featuremap = self.offset_head.vis_featuremap

        return offset_result


class AngleTestMixin(object):
    def simple_test_angle(self, x):
        angle_pred = self.angle_head(x)

        return angle_pred


class OffsetHeightTestMixin(object):
    def simple_test_offset_height(self, x, img_metas, det_bboxes, det_labels, rescale=False):
        # image shape of the first image in the batch (only one)
        ori_shape = img_metas[0]["ori_shape"]
        scale_factor = img_metas[0]["scale_factor"]
        if det_bboxes.shape[0] == 0:
            offset_result = [[] for _ in range(2)]
            height_result = [[] for _ in range(1)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(det_bboxes.device)
            _bboxes = det_bboxes[:, :4] * scale_factor if rescale else det_bboxes
            offset_height_rois = bbox2roi([_bboxes])
            offset_height_feats = self.offset_height_roi_extractor(
                x[: len(self.offset_height_roi_extractor.featmap_strides)], offset_height_rois
            )

            offset_pred, height_pred = self.offset_height_head(offset_height_feats)
            offset_result = self.offset_height_head.get_offsets(
                offset_pred, _bboxes, scale_factor, rescale
            )
            height_result = self.offset_height_head.get_heights(
                height_pred, _bboxes, scale_factor, rescale
            )
        return offset_result, height_result


class HeightTestMixin(object):
    def simple_test_height(self, x, img_metas, det_bboxes, det_labels, rescale=False):
        # image shape of the first image in the batch (only one)
        ori_shape = img_metas[0]["ori_shape"]
        scale_factor = img_metas[0]["scale_factor"]
        if det_bboxes.shape[0] == 0:
            height_result = [[] for _ in range(1)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(det_bboxes.device)
            _bboxes = det_bboxes[:, :4] * scale_factor if rescale else det_bboxes
            height_rois = bbox2roi([_bboxes])
            height_result = self._height_forward(x, height_rois)

            height_result = self.height_head.get_heights(
                height_result["height_pred"], _bboxes, scale_factor, rescale
            )

        return height_result


class OffsetFieldTestMixin(object):
    def simple_test_offset_field(self, x, img_metas, det_bboxes, det_labels, rescale=False):
        """Simple test for mask head without augmentation."""
        # image shape of the first image in the batch (only one)
        ori_shape = img_metas[0]["ori_shape"]
        scale_factor = img_metas[0]["scale_factor"]
        if det_bboxes.shape[0] == 0:
            offset_result = np.zeros((0, 2))
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(det_bboxes.device)
            _bboxes = det_bboxes[:, :4] * scale_factor if rescale else det_bboxes
            offset_field_rois = bbox2roi([_bboxes])
            offset_field_results = self._offset_field_forward(x, offset_field_rois)
            mask_results = self._mask_forward(x, offset_field_rois)

            offset_result = self.offset_field_head.get_offset(
                mask_results["mask_pred"],
                offset_field_results["offset_field_pred"],
                _bboxes,
                det_labels,
                self.test_cfg,
                ori_shape,
                scale_factor,
                rescale,
            )

        return offset_result


class OffsetReweightTestMixin(object):
    def simple_test_offset_reweight(self, x, img_metas, det_bboxes, det_labels, rescale=False):
        # image shape of the first image in the batch (only one)
        ori_shape = img_metas[0]["ori_shape"]
        scale_factor = img_metas[0]["scale_factor"]
        if det_bboxes.shape[0] == 0:
            offset_result = [[] for _ in range(2)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(det_bboxes.device)
            _bboxes = det_bboxes[:, :4] * scale_factor if rescale else det_bboxes
            offset_rois = bbox2roi([_bboxes])

            mask_results = self._mask_forward(x, offset_rois)
            side_face_results = self._side_face_forward(x, offset_rois)

            feature_weight = side_face_results["side_face_pred"] + mask_results["mask_pred"]
            feature_weight = (torch.sigmoid(F.interpolate(feature_weight, size=[7, 7])) + 1) / 2.0

            offset_feats = self.offset_roi_extractor(
                x[: len(self.offset_roi_extractor.featmap_strides)], offset_rois
            )

            offset_feats = offset_feats * feature_weight

            offset_pred = self.offset_head(offset_feats)
            offset_result = self.offset_head.get_offsets(
                offset_pred, _bboxes, scale_factor, rescale
            )
        return offset_result
