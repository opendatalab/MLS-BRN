import torch

from ..builder import BBOX_CODERS
from .base_bbox_coder import BaseBBoxCoder


@BBOX_CODERS.register_module()
class DeltaHeightCoder(BaseBBoxCoder):
    def __init__(self, target_means=(0.0), target_stds=(0.5)):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds

    def encode(self, bboxes, gt_heights):
        assert bboxes.size(0) == gt_heights.size(0)
        assert gt_heights.size(-1) == 1
        encoded_offsets = height2delta(bboxes, gt_heights, self.means, self.stds)
        return encoded_offsets

    def decode(self, bboxes, pred_heights):
        assert pred_heights.size(0) == bboxes.size(0)
        decoded_heights = delta2height(bboxes, pred_heights, self.means, self.stds)

        return decoded_heights


def height2delta(proposals, gt, means=(0.0), stds=(0.5)):
    assert proposals.size()[0] == gt.size()[0]

    proposals = proposals.float()
    gt = gt.float()
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]

    gh = gt[..., 0]

    pl = torch.sqrt(pw * pw + ph * ph)
    dh = gh / pl
    deltas = torch.stack([dh], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def delta2height(rois, deltas, means=(0.0), stds=(1.0)):
    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    dh = deltas * stds + means

    # Compute width/height of each roi
    pw = (rois[:, 2] - rois[:, 0]).unsqueeze(1).expand_as(dh)
    ph = (rois[:, 3] - rois[:, 1]).unsqueeze(1).expand_as(dh)

    pl = torch.sqrt(pw * pw + ph * ph)
    heights = dh * pl

    return heights
