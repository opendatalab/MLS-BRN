import argparse
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument("--merged-out", help="output merged result file in pickle format")
    parser.add_argument("--merge-iou-threshold", type=float, default=0.1, help="threshold of iou")
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase" "the inference speed",
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation. It is"
        "useful when you want to format the result to a specific format and "
        "submit it to the test server",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        default="segm",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
    )
    parser.add_argument(
        "--city", type=str, default="bonai_shanghai_xian", help="dataset for evaluation"
    )
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument("--show-dir", help="directory where painted images will be saved")
    parser.add_argument(
        "--show-score-thr", type=float, default=0.3, help="score threshold (default: 0.3)"
    )
    parser.add_argument(
        "--gpu-collect", action="store_true", help="whether to use gpu to collect results."
    )
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu-collect is not specified",
    )
    parser.add_argument("--options", nargs="+", action=DictAction, help="arguments in dict")
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--nms-score", type=float, default=0.5, help="nms threshold (default: 0.5)")
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def choose_test_dataset(cfg, args, mask_type, data_root):
    city_types_to_full = {
        "omnicity": "OmniCityView3WithOffset",
        "hk": "hongkong",
        "bonai_hk": "bonai_hongkong",
    }
    if "roof" in mask_type or "footprint" in mask_type:
        mask_short = "footprint" if "footprint" in mask_type else "roof"

        cfg.data.test.ann_file = f"{data_root}BONAI/coco/bonai_shanghai_xian_test_{mask_short}.json"
        cfg.data.test.img_prefix = data_root + "BONAI/test/images/"
        if args.city == "bonai":
            print("################ Use Default City BONAI shanghai_xian for TEST ################")
        elif args.city in ["bonai_hk"]:  # For City Group
            city_full_name = city_types_to_full[args.city]
            print(
                "################ Use City Group {} for {} TEST ################".format(
                    city_full_name, mask_short
                )
            )
            cfg.data.test.ann_file = (
                f"{data_root}combined_test/coco/{city_full_name}_test_{mask_short}.json"
            )
            cfg.data.test.img_prefix = f"{data_root}combined_test/images/{city_full_name}/"
        elif args.city in ["omnicity", "hk"]:
            city_full_name = city_types_to_full[args.city]
            print(
                "################ Use Single City {} for {} TEST ################".format(
                    city_full_name, mask_short
                )
            )
            cfg.data.test.ann_file = (
                f"{data_root}{city_full_name}/coco/{city_full_name}_test_{mask_short}.json"
            )
            cfg.data.test.img_prefix = f"{data_root}{city_full_name}/test/images/"
        else:
            print("################ No Such TEST City Type: {}! ################".format(args.city))
            print(
                "################ Use Default City BONAI shanghai_xian for {} TEST! ################".format(
                    mask_short
                )
            )
    else:
        raise ValueError(f"Wrong mask type for test: {mask_type}")


def eval_different_mask(cfg, args, kwargs, mask_type, outputs, data_root):
    choose_test_dataset(cfg, args, mask_type, data_root)

    print("Dataset for evaluation: ", cfg.data.test.ann_file)
    print("################", mask_type, " evaluate start ################")
    dataset = build_dataset(cfg.data.test)
    dataset.evaluate(outputs, args.eval, mask_type, **kwargs)
    print("################", mask_type, " evaluate end ################")


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show or args.show_dir, (
        "Please specify at least one operation (save/eval/format/show the "
        'results / save the results) with the argument "--out", "--eval"'
        ', "--format-only", "--show" or "--show-dir"'
    )

    if args.eval and args.format_only:
        raise ValueError("--eval and --format_only cannot be both specified")

    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    data_root = "./data/"

    choose_test_dataset(cfg, args, "roof", data_root)

    if cfg.test_cfg.get("rcnn", False):
        cfg.test_cfg.rcnn.nms.iou_threshold = args.nms_score
        print("NMS config for testing: {}".format(cfg.test_cfg.rcnn.nms))
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.test_dataloader.samples_per_gpu,
        workers_per_gpu=cfg.data.test_dataloader.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )
    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walk around is
    # for backward compatibility
    if "CLASSES" in checkpoint["meta"]:
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir, args.show_score_thr)
    else:
        model = MMDistributedDataParallel(
            model.cuda(), device_ids=[torch.cuda.current_device()], broadcast_buffers=False
        )
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f"\nwriting results to {args.out}")
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.options is None else args.options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            mask_types = ["roof", "offset_footprint", "direct_footprint"]
            eval_different_mask(cfg, args, kwargs, mask_types[0], outputs, data_root)
            eval_different_mask(cfg, args, kwargs, mask_types[1], outputs, data_root)
            eval_different_mask(cfg, args, kwargs, mask_types[2], outputs, data_root)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
    main()
