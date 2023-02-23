"""This file defines the Footprint mask from roof and offset heads."""
import torch
from mmcv.runner import auto_fp16, force_fp32

from mmdet.models.builder import HEADS
from ..mask_heads import FCNMaskHead


@HEADS.register_module()
class FootprintMaskFromRoofOffsetHead(FCNMaskHead):
    """The FootprintMaskFromRoofOffsetHead."""

    @auto_fp16()
    def forward(self, offsets, roofs):
        x = self.concat_offsets_and_roofs(offsets, roofs)
        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == "deconv":
                x = self.relu(x)
        mask_pred = self.conv_logits(x)
        return mask_pred

    def concat_offsets_and_roofs(self, offsets, roofs):
        ox = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(offsets[:, 0], dim=1), dim=1), dim=1)
        oy = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(offsets[:, 1], dim=1), dim=1), dim=1)
        offset_x_mask = torch.ones_like(roofs) * ox
        offset_y_mask = torch.ones_like(roofs) * oy
        x = torch.cat([offset_x_mask, offset_y_mask, roofs], dim=1)
        return x

    @force_fp32(apply_to=("mask_pred",))
    def loss(self, mask_pred, mask_targets, labels):
        """
        Example:
            >>> from mmdet.models.roi_heads.mask_heads.fcn_mask_head import *  # NOQA
            >>> N = 7  # N = number of extracted ROIs
            >>> C, H, W = 11, 32, 32
            >>> # Create example instance of FCN Mask Head.
            >>> # There are lots of variations depending on the configuration
            >>> self = FCNMaskHead(num_classes=C, num_convs=1)
            >>> inputs = torch.rand(N, self.in_channels, H, W)
            >>> mask_pred = self.forward(inputs)
            >>> sf = self.scale_factor
            >>> labels = torch.randint(0, C, size=(N,))
            >>> # With the default properties the mask targets should indicate
            >>> # a (potentially soft) single-class label
            >>> mask_targets = torch.rand(N, H * sf, W * sf)
            >>> loss = self.loss(mask_pred, mask_targets, labels)
            >>> print('loss = {!r}'.format(loss))
        """
        loss = dict()
        if mask_pred.size(0) == 0:
            loss_mask = mask_pred.sum()
        else:
            if self.class_agnostic:
                loss_mask = self.loss_mask(mask_pred, mask_targets, torch.zeros_like(labels))
            else:
                loss_mask = self.loss_mask(mask_pred, mask_targets, labels)
        loss["loss_footprint_mask_from_roof_offset"] = loss_mask
        return loss
