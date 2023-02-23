# -*- encoding: utf-8 -*-

import torch

from mmdet.core import bbox2result, bbox2roi
from ..builder import HEADS, build_head, build_roi_extractor
from .standard_roi_head import StandardRoIHead
from .test_mixins import HeightTestMixin, OffsetTestMixin


@HEADS.register_module()
class LoftHRoIHead(StandardRoIHead, OffsetTestMixin, HeightTestMixin):
    def __init__(
        self,
        offset_roi_extractor=None,
        offset_head=None,
        height_roi_extractor=None,
        height_head=None,
        **kwargs,
    ):
        assert offset_head is not None
        assert height_head is not None
        super().__init__(**kwargs)

        self.init_offset_head(offset_roi_extractor, offset_head)
        self.init_height_head(height_roi_extractor, height_head)

        self.with_vis_feat = False

    def init_offset_head(self, offset_roi_extractor, offset_head):
        """Build offset roi extractor and offset head."""
        self.offset_roi_extractor = build_roi_extractor(offset_roi_extractor)
        self.offset_head = build_head(offset_head)

    def init_height_head(self, height_roi_extractor, height_head):
        """Build height roi extractor and height head."""
        self.height_roi_extractor = build_roi_extractor(height_roi_extractor)
        self.height_head = build_head(height_head)

    # def init_weights(self, pretrained):
    #     super(LoftRoIHead, self).init_weights(pretrained)
    #     self.offset_head.init_weights()
    def init_weights(self):
        super().init_weights()
        self.offset_head.init_weights()
        self.height_head.init_weights()

    def forward_train(
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
            gt_heights (None | Tensor): each item are truth heights for each
                image.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
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
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(
                x, sampling_results, gt_bboxes, gt_labels, img_metas
            )
            losses.update(bbox_results["loss_bbox"])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(
                x, sampling_results, bbox_results["bbox_feats"], gt_masks, img_metas
            )
            # TOD: Support empty tensor input. #2280
            if mask_results["loss_mask"] is not None:
                losses.update(mask_results["loss_mask"])

        # offset head forward and loss
        if self.with_offset:
            # print("mask_results['mask_pred']: ", mask_results['mask_pred'].shape)
            # print("mask_results['mask_targets']: ", mask_results['mask_targets'].shape)
            # print("bbox_results['bbox_feats']: ", bbox_results['bbox_feats'].shape)
            offset_results = self._offset_forward_train(
                x, sampling_results, bbox_results["bbox_feats"], gt_offsets, img_metas
            )
            # TOD: Support empty tensor input. #2280
            if offset_results["loss_offset"] is not None:
                losses.update(offset_results["loss_offset"])

        # height head forward and los
        if self.with_height:
            height_results = self._height_forward_train(
                x, sampling_results, bbox_results["bbox_feats"], gt_heights, img_metas
            )
            if height_results["loss_height"] is not None:
                losses.update(height_results["loss_height"])

        return losses

    def _offset_forward_train(self, x, sampling_results, bbox_feats, gt_offsets, img_metas):
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        # if pos_rois.shape[0] == 0:
        #     return dict(loss_offset=None)
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

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks, img_metas):
        """Run forward function and calculate loss for mask head in training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(res.pos_bboxes.shape[0], device=device, dtype=torch.uint8)
                )
                pos_inds.append(
                    torch.zeros(res.neg_bboxes.shape[0], device=device, dtype=torch.uint8)
                )
            pos_inds = torch.cat(pos_inds)
            mask_results = self._mask_forward(x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks, self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results["mask_pred"], mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def simple_test(self, x, proposal_list, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, "Bbox head must be implemented."

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale
        )

        bbox_results = bbox2result(det_bboxes[0], det_labels[0], self.bbox_head.num_classes)
        # bbox_results = bbox2result(det_bboxes, det_labels,
        #                            self.bbox_head.num_classes)

        height_results = self.simple_test_height(
            x, img_metas, det_bboxes[0], det_labels[0], rescale=rescale
        )

        if self.with_mask:
            seg_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale
            )
            if self.with_vis_feat:
                offset_results = self.simple_test_offset_rotate_feature(
                    x, img_metas, det_bboxes, det_labels, rescale=rescale
                )
                return (
                    bbox_results,
                    seg_results[0],
                    offset_results,
                    height_results,
                    self.vis_featuremap,
                )
            else:
                offset_results = self.simple_test_offset(
                    x, img_metas, det_bboxes[0], det_labels[0], rescale=rescale
                )
                return bbox_results, seg_results[0], offset_results, height_results
        else:
            offset_results = self.simple_test_offset(
                x, img_metas, det_bboxes[0], det_labels[0], rescale=rescale
            )
            # offset_results = self.simple_test_offset(
            #     x, img_metas, det_bboxes, det_labels, rescale=rescale)

            return bbox_results, None, offset_results, height_results
