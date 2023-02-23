# -*- encoding: utf-8 -*-

from functools import reduce

import torch
from mmcv.cnn import Conv2d, kaiming_init, normal_init
from torch import nn

from mmdet.models.builder import HEADS, build_loss


@HEADS.register_module
class OffsetAngleHead(nn.Module):
    """This class defines the offset angle head,
    which is used for constraining the offset and calculating pseudo gt_bboxes for RPN."""

    def __init__(
        self,
        in_size=1024,
        in_channels=3,
        num_convs=6,
        conv_out_channels=[16, 16, 32, 32, 64, 64],
        kernel_size=3,
        strides=[2, 2, 2, 2, 2, 2],
        num_fcs=2,
        fc_out_channels=[128, 32],
        reg_num=2,
        conv_cfg=None,
        norm_cfg=None,
        loss_angle=dict(type="MSELoss", loss_weight=1.0),
        regular_lambda=0.1,
        loss_method="loss",
        with_tanh=True,
    ):
        super().__init__()

        assert num_convs == len(conv_out_channels)
        assert num_convs > 0
        assert num_fcs == len(fc_out_channels)
        assert num_fcs > 0

        self.num_convs = num_convs
        self.num_fcs = num_fcs
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.reg_num = reg_num
        self.with_tanh = with_tanh

        self.loss_angle = build_loss(loss_angle)
        self.regular_lambda = regular_lambda
        self.loss_method = getattr(self, loss_method)

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # define the conv and fc operations
        self.convs = nn.ModuleList()
        conv_in_channels = [self.in_channels] + self.conv_out_channels[:-1]
        for i in range(self.num_convs):
            self.convs.append(
                Conv2d(
                    in_channels=conv_in_channels[i],
                    out_channels=self.conv_out_channels[i],
                    kernel_size=kernel_size,
                    stride=strides[i],
                    padding=1,
                )
            )
        fc_in_size = in_size / reduce(lambda x, y: x * y, strides)
        fc_first_in_channel = int(fc_in_size * fc_in_size * conv_out_channels[-1])
        fc_in_channels = [fc_first_in_channel] + self.fc_out_channels[:-1]

        self.fcs = nn.ModuleList()
        for i in range(num_fcs):
            self.fcs.append(
                nn.Linear(
                    in_features=fc_in_channels[i],
                    out_features=self.fc_out_channels[i],
                )
            )

        self.fc_angle = nn.Linear(self.fc_out_channels[-1], self.reg_num)

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
        normal_init(self.fc_angle, std=0.01)

    def forward(self, x):
        """This method defines the forward process."""
        if x.size(0) == 0:
            return x.new_empty(x.size(0), self.reg_num)

        for conv in self.convs:
            x = self.relu(conv(x))

        x = x.view(x.size(0), -1)

        for fc in self.fcs:
            x = self.relu(fc(x))

        angle_pred = self.fc_angle(x)

        if self.with_tanh:
            angle_pred = torch.tanh(angle_pred)

        return angle_pred

    def loss(self, offset_angle_pred, offset_angle_targets):
        """This method defines the loss function."""
        if offset_angle_pred.size(0) == 0:
            loss_offset_angle = offset_angle_pred.sum() * 0
            loss_regular = offset_angle_pred.sum() * 0
        else:
            loss_offset_angle = self.loss_angle(offset_angle_pred, offset_angle_targets)
            ones = torch.ones_like(
                offset_angle_pred[:, 0],
                dtype=offset_angle_pred.dtype,
                device=offset_angle_pred.device,
            )
            loss_regular = self.loss_angle(
                offset_angle_pred[:, 0] ** 2 + offset_angle_pred[:, 1] ** 2, ones
            )
        return loss_offset_angle + self.regular_lambda * loss_regular

    def forward_train(self, x, img_metas, gt_offset_angles):
        """Forward train process."""
        offset_angle_pred = self.forward(x)
        offset_angle_targets = torch.cat(gt_offset_angles, 0)
        loss_offset_angle = self.loss_method(offset_angle_pred, offset_angle_targets)
        return offset_angle_pred, dict(loss_offset_angle=loss_offset_angle)

    def simple_test(self, x):
        return self.forward(x)


@HEADS.register_module
class NadirAngleHead(OffsetAngleHead):
    """This class defines the nadir angle head,
    which is used for constraining the offset and calculating pseudo gt_bboxes for RPN."""

    def forward(self, x):
        """This method defines the forward process."""
        if x.size(0) == 0:
            return x.new_empty(x.size(0), self.reg_num)

        for conv in self.convs:
            x = self.relu(conv(x))

        x = x.view(x.size(0), -1)

        for fc in self.fcs:
            x = self.relu(fc(x))

        angle_pred = self.fc_angle(x)

        return angle_pred

    def loss(self, nadir_angle_pred, nadir_angle_targets):
        """This method defines the loss function."""
        if nadir_angle_pred.size(0) == 0:
            loss_angle = nadir_angle_pred.sum() * 0
        else:
            loss_angle = self.loss_angle(nadir_angle_pred, nadir_angle_targets)
        return loss_angle

    def forward_train(self, x, img_metas, gt_nadir_angles):
        """Forward train process."""
        nadir_angle_pred = self.forward(x)
        nadir_angle_targets = torch.unsqueeze(torch.cat(gt_nadir_angles, 0), dim=-1)
        loss_nadir_angle = self.loss(nadir_angle_pred, nadir_angle_targets)
        return nadir_angle_pred, dict(loss_nadir_angle=loss_nadir_angle)
