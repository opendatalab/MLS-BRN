# -*- encoding: utf-8 -*-

import numpy as np
import torch
from mmcv.cnn import Conv2d, kaiming_init, normal_init
from torch import nn

from mmdet.core import build_bbox_coder, multi_apply
from mmdet.models.builder import HEADS, build_loss


@HEADS.register_module
class HeightHead(nn.Module):
    """This class defines the height head."""

    def __init__(
        self,
        roi_feat_size=7,
        in_channels=256,
        num_convs=4,
        num_fcs=2,
        reg_num=1,
        conv_out_channels=256,
        fc_out_channels=1024,
        height_coder=dict(type="DeltaHeightCoder", target_means=[0.0], target_stds=[0.5]),
        conv_cfg=None,
        norm_cfg=None,
        loss_height=dict(type="MSELoss", loss_weight=1.0),
    ):
        super().__init__()

        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.reg_num = reg_num

        self.height_coder = build_bbox_coder(height_coder)
        self.loss_height = build_loss(loss_height)

        # TODO: Confirm that whether conv_cfg and norm_cfg are used.
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # define the conv and fc operations
        self.convs = nn.ModuleList()
        for i in range(num_convs):
            self.convs.append(
                Conv2d(
                    in_channels=self.in_channels if i == 0 else self.conv_out_channels,
                    out_channels=self.conv_out_channels,
                    kernel_size=3,
                    padding=1,
                )
            )

        roi_feat_area = roi_feat_size * roi_feat_size

        self.fcs = nn.ModuleList()
        for i in range(num_fcs):
            in_channels = self.conv_out_channels * roi_feat_area if i == 0 else self.fc_out_channels
            self.fcs.append(
                nn.Linear(
                    in_features=in_channels,
                    out_features=self.fc_out_channels,
                )
            )

        self.fc_height = nn.Linear(self.fc_out_channels, self.reg_num)

        self.relu = nn.ReLU()

    def init_weights(self):
        """This method initializes the head's weights."""
        for conv in self.convs:
            kaiming_init(conv)

        for fc in self.fcs:
            kaiming_init(
                module=fc,
                a=1,
                mode="fan_in",
                nonlinearity="leaky_relu",
                distribution="uniform",
            )
        normal_init(self.fc_height, std=0.01)

    def forward(self, x):
        """This method defines the forward process."""
        if x.size(0) == 0:
            return x.new_empty(x.size(0), self.reg_num)

        for conv in self.convs:
            x = self.relu(conv(x))

        x = x.view(x.size(0), -1)

        for fc in self.fcs:
            x = self.relu(fc(x))

        height = self.fc_height(x)

        return height

    def loss(self, height_pred, height_targets):
        """This method defines the loss function."""
        if height_pred.size(0) == 0:
            loss_height = height_pred.sum() * 0
        else:
            loss_height = self.loss_height(height_pred, height_targets)
        return dict(loss_height=loss_height)

    def _height_target_single(self, pos_proposals, pos_assigned_gt_inds, gt_heights, cfg):
        device = pos_proposals.device
        num_pos = pos_proposals.size(0)
        height_targets = pos_proposals.new_zeros(pos_proposals.size(0), self.reg_num)

        pos_gt_heights = []

        if num_pos > 0:
            pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
            for i in range(num_pos):
                gt_height = gt_heights[pos_assigned_gt_inds[i]]
                pos_gt_heights.append(gt_height.tolist())

            pos_gt_heights = torch.from_numpy(np.stack(np.array(pos_gt_heights))).float().to(device)
            height_targets = self.height_coder.encode(pos_proposals, pos_gt_heights)
        else:
            height_targets = pos_proposals.new_zeros((0, self.reg_num))

        return height_targets, height_targets

    def get_targets(self, sampling_results, gt_heights, rcnn_train_cfg, concat=True):
        """get the targets of height in training stage

        Args:
            sampling_results (torch.Tensor): sampling results
            gt_heights (torch.Tensor): height ground truth
            rcnn_train_cfg (dict): rcnn training config
            concat (bool, optional): concat flag. Defaults to True.

        Returns:
            torch.Tensor: height targets
        """
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        height_targets, _ = multi_apply(
            self._height_target_single,
            pos_proposals,
            pos_assigned_gt_inds,
            gt_heights,
            cfg=rcnn_train_cfg,
        )

        if concat:
            height_targets = torch.cat(height_targets, 0)

        return height_targets

    def get_heights(self, height_pred, det_bboxes, scale_factor, rescale, img_shape=[1024, 1024]):
        # generate heights in inference stage
        if height_pred is not None:
            heights = self.height_coder.decode(det_bboxes, height_pred)
        else:
            heights = torch.zeros((det_bboxes.size()[0], self.reg_num))

        if isinstance(heights, torch.Tensor):
            heights = heights.cpu().numpy()
        assert isinstance(heights, np.ndarray)

        heights = heights.astype(np.float32)

        return heights
