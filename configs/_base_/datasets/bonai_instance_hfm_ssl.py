dataset_type = "BONAI_SSL"
data_root = "data"
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="LoadAnnotations",
        with_bbox=True,
        with_mask=True,
        with_offset=True,
        with_height=True,
        with_height_mask=True,
        with_footprint_mask=True,
        with_image_scale_footprint_mask=True,
        with_footprint_bbox=True,
        with_offset_angle=True,
        with_nadir_angle=True,
        with_semi_supervised_learning=True,
        with_valid_height_flag=True,
    ),
    dict(type="Resize", img_scale=(1024, 1024), keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5, direction=["horizontal", "vertical"]),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="LOFTFormatBundle"),
    dict(
        type="Collect",
        keys=[
            "img",
            "gt_bboxes",
            "gt_labels",
            "gt_masks",
            "gt_offsets",
            "gt_heights",
            "gt_height_masks",
            "gt_footprint_masks",
            "gt_image_scale_footprint_masks",
            "gt_footprint_bboxes",
            "gt_is_semi_supervised_sample",
            "gt_is_valid_height_sample",
            "gt_offset_angles",
            "gt_nadir_angles",
            "height_mask_shape",
            "image_scale_footprint_mask_shape",
        ],
    ),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip", flip_ratio=0.5),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
train_ann_file = []
img_prefix = []

dataset_dirs = {
    "bonai_shanghai": "BONAI",
    "bonai_beijing": "BONAI",
    "bonai_jinan": "BONAI",
    "bonai_haerbin": "BONAI",
    "bonai_chengdu": "BONAI",
    "OmniCityView3WithOffset": "OmniCityView3WithOffset",
    "hongkong": "hongkong",
}

versions_to_ann_dirs = {
    "30oh": "coco_30oh",
    "30oh/30h": "coco_30oh_30h",
    "30oh/30h/40n": "coco_30oh_30h_40n",
    "30oh/70h": "coco_30oh_70h",
    "100oh": "coco",
    "30oh/70n": "coco_30oh_70n",
}

# ==================== control chosen datasets ====================
datasets = {
    # "bonai_shanghai": "30oh/70h",
    # "bonai_beijing": "30oh/70h",
    # "bonai_jinan": "30oh/70h",
    # "bonai_haerbin": "30oh/70h",
    # "bonai_chengdu": "30oh/70h",
    #
    # "bonai_shanghai": "30oh",
    # "bonai_beijing": "30oh",
    # "bonai_jinan": "30oh",
    # "bonai_haerbin": "30oh",
    # "bonai_chengdu": "30oh",
    #
    "bonai_shanghai": "100oh",
    "bonai_beijing": "100oh",
    "bonai_jinan": "100oh",
    "bonai_haerbin": "100oh",
    "bonai_chengdu": "100oh",
    #
    # "OmniCityView3WithOffset": "100oh",
    # "OmniCityView3WithOffset": "30oh",
    # "OmniCityView3WithOffset": "30oh/30h",
    # "OmniCityView3WithOffset": "30oh/70h",
    # "OmniCityView3WithOffset": "30oh/30h/40n",
    #
    # "hongkong": "100oh",
    # "hongkong": "30oh/70n",
    # "hongkong": "30oh",
}
# =================================================================

for dataset, version in datasets.items():
    dataset_dir = f"{data_root}/{dataset_dirs[dataset]}"
    ann_path = f"{dataset_dir}/{versions_to_ann_dirs[version]}/{dataset}_trainval.json"
    train_ann_file.append(ann_path)
    img_prefix.append(f"{dataset_dir}/trainval/images")

# Should be aligned with one output layer of FPN.
height_mask_shape = [256, 256]
image_scale_footprint_mask_shape = [256, 256]
resolution = 0.6
data = dict(
    train=dict(
        type=dataset_type,
        ann_file=train_ann_file,
        img_prefix=img_prefix,
        bbox_type="building",
        mask_type="roof",
        height_mask_shape=height_mask_shape,
        image_scale_footprint_mask_shape=image_scale_footprint_mask_shape,
        pipeline=train_pipeline,
    ),
    train_dataloader=dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
    ),
    val=dict(
        type=dataset_type,
        ann_file=train_ann_file,
        img_prefix=img_prefix,
        bbox_type="building",
        mask_type="footprint",
        height_mask_shape=height_mask_shape,
        image_scale_footprint_mask_shape=image_scale_footprint_mask_shape,
        gt_footprint_csv_file="",
        pipeline=test_pipeline,
    ),
    val_dataloader=dict(
        samples_per_gpu=1,
        workers_per_gpu=2,
    ),
    test=dict(
        type=dataset_type,
        ann_file=train_ann_file,
        img_prefix=img_prefix,
        bbox_type="building",
        mask_type="footprint",
        gt_footprint_csv_file="",
        height_mask_shape=height_mask_shape,
        image_scale_footprint_mask_shape=image_scale_footprint_mask_shape,
        pipeline=test_pipeline,
    ),
    test_dataloader=dict(
        samples_per_gpu=1,
        workers_per_gpu=2,
    ),
)
evaluation = dict(interval=1, metric=["segm"])
# evaluation = dict(start=20,interval=1, metric=["bbox", "segm"])
