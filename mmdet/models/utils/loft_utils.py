import copy
import math

import numpy as np


def offset_roof_to_footprint(offsets, roof_masks, reverse=False):
    """Used for offsetting the roof mask to footprint mask."""
    direction = -1 if reverse else 1
    h, w = roof_masks[0][0][0].shape
    footprint_masks = []
    for offsets_per_img, roof_masks_per_img in zip(offsets, roof_masks):
        footprint_masks_per_img = []
        for offset, roof_mask in zip(offsets_per_img, roof_masks_per_img[0]):
            border = max(math.ceil(abs(offset[0])), math.ceil(abs(offset[1])))
            canvas = np.full((h + 2 * border, w + 2 * border), False)
            canvas[border : h + border, border : w + border] = roof_mask
            canvas = np.roll(
                canvas, shift=(direction * int(offset[1]), direction * int(offset[0])), axis=(0, 1)
            )
            footprint_mask = canvas[border : h + border, border : w + border]
            footprint_masks_per_img.append(footprint_mask)
        footprint_masks.append([footprint_masks_per_img])

    return footprint_masks


def test_offset_roof_to_footprint(offsets, roof_masks, footprint_masks):
    import cv2

    save_path = "tmp/test_offset_roof_to_footprint/"
    canvas_origin = np.zeros((1024, 1024))
    idx = 0
    for offset, roof_mask, footprint_mask in zip(
        offsets[0], footprint_masks[0][0], roof_masks[0][0]
    ):
        if np.where(roof_mask is True)[0].shape[0] == 0:
            continue

        path = save_path + f"({offset[0]}_{offset[1]})_{str(idx).zfill(12)}.jpg"
        canvas = copy.deepcopy(canvas_origin)
        canvas[np.where(roof_mask is False)] = 0
        canvas[np.where(roof_mask is True)] = 255
        canvas[np.where(footprint_mask is True)] = 128
        cv2.imwrite(path, canvas)
        idx += 1
        # if idx > 4:
        #     break
