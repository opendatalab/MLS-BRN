import json
import random
from collections import defaultdict
from pathlib import Path

DATASETS_DIR = Path(__file__).parent.parent.parent / "data"
# DATASET = "hongkong"
# DATASET = "OmniCityView3WithOffset"
DATASET = "BONAI"
DATASET_DIR = DATASETS_DIR / DATASET


def check_ratios(ann_dir):
    total_cnt = 0
    oh_cnt = 0
    h_cnt = 0
    for ann_path in ann_dir.iterdir():
        str_path = str(ann_path)
        print(str_path)
        if str_path.startswith(".") or "test" in str_path:
            continue

        with open(ann_path, "r", encoding="UTF-8") as fp:
            content = json.load(fp)

        anns_per_image = defaultdict(list)
        for ann in content["annotations"]:
            anns_per_image[ann["image_id"]].append(ann)
        for anns in anns_per_image.values():
            total_cnt += 1
            if "offset" in anns[0] and "building_height" in anns[0]:
                oh_cnt += 1
            elif "building_height" in anns[0]:
                h_cnt += 1
    print("with offset&height anns:", oh_cnt / total_cnt)
    print("with height anns:", h_cnt / total_cnt)
    print("with footprint anns:", (total_cnt - oh_cnt - h_cnt) / total_cnt)


def let_segmentation_equal_to_roof():
    ann_path = DATASET_DIR / "coco" / "bonai_shanghai_xian_test_roof.json"
    with open(ann_path, "r", encoding="UTF-8") as fp:
        content = json.load(fp)
    for ann in content["annotations"]:
        ann["segmentation"] = [ann["roof_mask"]]
    with open(ann_path, "w", encoding="UTF-8") as fp:
        json.dump(content, fp, indent=4, separators=(",", ":"))


def create_wsl_dataset(oh_ratio, h_ratio, n_ratio):
    assert oh_ratio >= 0.0 and h_ratio >= 0.0 and n_ratio >= 0.0
    assert oh_ratio + h_ratio + n_ratio <= 1.0

    ann_dir = DATASET_DIR / "coco"
    oh_suffix = f"_{str(int(oh_ratio*100))}oh" if oh_ratio > 0.0 else ""
    h_suffix = f"_{str(int(h_ratio*100))}h" if h_ratio > 0.0 else ""
    n_suffix = f"_{str(int(n_ratio*100))}n" if n_ratio > 0.0 else ""
    new_ann_dir = DATASET_DIR / f"coco{oh_suffix}{h_suffix}{n_suffix}"

    if not new_ann_dir.exists():
        new_ann_dir.mkdir()

    for ann_path in ann_dir.iterdir():
        str_path = str(ann_path).rsplit("/", 1)[-1]
        print(str_path)
        if str_path.startswith(".") or "test" in str_path:
            continue

        with open(ann_path, "r", encoding="UTF-8") as fp:
            content = json.load(fp)

        anns_per_image = defaultdict(list)
        new_anns = []
        for ann in content["annotations"]:
            anns_per_image[ann["image_id"]].append(ann)

        for anns in anns_per_image.values():
            x = random.random()
            if x < oh_ratio:
                new_anns += anns
            elif x < oh_ratio + h_ratio:
                for ann in anns:
                    ann.pop("offset")
                new_anns += anns
            elif x < oh_ratio + h_ratio + n_ratio:
                for ann in anns:
                    ann.pop("offset")
                    ann.pop("building_height")
                new_anns += anns

        content["annotations"] = new_anns

        new_ann_path = new_ann_dir / str_path
        with open(new_ann_path, "w", encoding="UTF-8") as fp:
            json.dump(content, fp, indent=4, separators=(",", ":"))
        print(new_ann_path)


if __name__ == "__main__":
    # let_segmentation_equal_to_roof()
    # create_wsl_dataset(0.65, 0.15, 0.2)
    check_ratios(DATASET_DIR / "coco_30oh_30h_40n")
