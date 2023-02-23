# -*- encoding: utf-8 -*-
import argparse
import csv
import math
import os
import warnings

import bstool
import cv2
import geopandas
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import six
import tqdm
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from shapely import affinity
from terminaltables import AsciiTable


class Evaluation:
    def __init__(
        self,
        model=None,
        anno_file=None,
        pkl_file=None,
        resolution=0.6,
        gt_roof_csv_file=None,
        gt_footprint_csv_file=None,
        roof_csv_file=None,
        footprint_csv_file=None,
        footprint_direct_csv_file=None,
        json_prefix=None,
        iou_threshold=0.1,
        score_threshold=0.4,
        min_area=500,
        with_offset=False,
        with_height=False,
        output_dir=None,
        out_file_format="png",
        show=False,
        replace_pred_roof=False,
        replace_pred_offset=False,
        with_only_offset=False,
        offset_model="footprint2roof",
        save_merged_csv=True,
    ):
        self.anno_file = anno_file
        self.resolution = resolution
        self.gt_roof_csv_file = gt_roof_csv_file
        self.gt_footprint_csv_file = gt_footprint_csv_file
        self.roof_csv_file = roof_csv_file
        self.footprint_csv_file = footprint_csv_file
        self.footprint_direct_csv_file = footprint_direct_csv_file
        self.pkl_file = pkl_file
        self.json_prefix = json_prefix
        self.show = show
        self.classify_interval = [
            0,
            2,
            4,
            6,
            8,
            10,
            15,
            20,
            25,
            30,
            35,
            40,
            45,
            50,
            55,
            60,
            65,
            70,
            75,
            80,
            85,
            90,
            95,
            100,
            110,
            120,
            130,
            140,
            150,
            160,
            170,
            180,
            190,
            200,
            220,
            240,
            260,
            280,
            300,
            340,
            380,
        ]
        self.offset_class_num = len(self.classify_interval)
        self.with_only_offset = with_only_offset
        self.save_merged_csv = save_merged_csv

        self.out_file_format = out_file_format

        self.output_dir = output_dir
        if output_dir:
            mkdir_or_exist(self.output_dir)

        # 1. create the pkl parser which is used for parse the pkl file (detection result)
        if self.with_only_offset:
            # BSPklParser_Only_Offset is designed to evaluate the experimental model which only predicts the offsets
            pkl_parser = bstool.BSPklParser_Only_Offset(
                anno_file,
                pkl_file,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold,
                min_area=min_area,
                with_offset=with_offset,
                with_height=with_height,
                gt_roof_csv_file=gt_roof_csv_file,
                replace_pred_roof=replace_pred_roof,
                offset_model=offset_model,
            )
        else:
            if with_offset:
                # important
                # BSPklParser is the general class for evaluating the LOVE and S2LOVE models
                pkl_parser = bstool.BSPklParser(
                    anno_file,
                    pkl_file,
                    iou_threshold=iou_threshold,
                    score_threshold=score_threshold,
                    min_area=min_area,
                    with_offset=with_offset,
                    with_height=with_height,
                    gt_roof_csv_file=gt_roof_csv_file,
                    replace_pred_roof=replace_pred_roof,
                    replace_pred_offset=replace_pred_offset,
                    offset_model=offset_model,
                    merge_splitted=save_merged_csv,
                )
            else:
                # BSPklParser_Without_Offset is designed to evaluate the baseline models (Mask R-CNN, etc.)
                pkl_parser = bstool.BSPklParser_Without_Offset(
                    anno_file,
                    pkl_file,
                    iou_threshold=iou_threshold,
                    score_threshold=score_threshold,
                    min_area=min_area,
                    with_offset=with_offset,
                    with_height=with_height,
                    gt_roof_csv_file=gt_roof_csv_file,
                    replace_pred_roof=replace_pred_roof,
                    offset_model=offset_model,
                )

        # 2. merge the detection results, and generate the csv file (convert pkl to csv, the file file format for evaluating the F1 is CSV, pkl format is the pre format)
        # whether or not merge the results on the sub-images (1024 * 1024) to original image (2048 * 2048)
        if save_merged_csv:
            merged_objects = pkl_parser.merged_objects
            bstool.bs_csv_dump(merged_objects, roof_csv_file, footprint_csv_file)
            self.dump_result = True
        else:
            objects = pkl_parser.objects
            self.dump_result = bstool.bs_csv_dump(
                objects, roof_csv_file, footprint_csv_file, footprint_direct_csv_file
            )

    def _csv2json(self, csv_file, ann_file):
        """convert csv file to json which will be used to evaluate the results by COCO API

        Args:
            csv_file (str): csv file
            ann_file (str): annotation file of COCO format (.json)

        Returns:
            list: list for saving to json
        """
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids()
        self.img_ids = self.coco.get_img_ids()

        csv_parser = bstool.CSVParse(csv_file)

        bbox_json_results = []
        segm_json_results = []
        for idx in tqdm.tqdm(range(len(self.img_ids))):
            img_id = self.img_ids[idx]
            info = self.coco.load_imgs([img_id])[0]
            image_name = bstool.get_basename(info["file_name"])

            objects = csv_parser(image_name)

            masks = [obj["mask"] for obj in objects]
            bboxes = [bstool.mask2bbox(mask) for mask in masks]

            for bbox, mask in zip(bboxes, masks):
                data = dict()
                data["image_id"] = img_id
                data["bbox"] = bstool.xyxy2xywh(bbox)
                data["score"] = 1.0
                data["category_id"] = self.category_id

                rles = maskUtils.frPyObjects([mask], self.image_size[0], self.image_size[1])
                rle = maskUtils.merge(rles)
                if isinstance(rle["counts"], bytes):
                    rle["counts"] = rle["counts"].decode()
                data["segmentation"] = rle

                bbox_json_results.append(data)
                segm_json_results.append(data)

        return bbox_json_results, segm_json_results

    def _coco_eval(
        self,
        metric=["bbox", "segm"],
        classwise=False,
        proposal_nums=(100, 300, 1000),
        iou_thrs=np.arange(0.5, 0.96, 0.05),
    ):
        """Please reference to original code in mmdet"""
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ["bbox", "segm"]
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f"metric {metric} is not supported")

        result_files = self.dump_json_results()

        eval_results = {}
        cocoGt = self.coco
        for metric in metrics:
            msg = f"Evaluating {metric}..."
            print(msg)
            if metric not in result_files:
                raise KeyError(f"{metric} is not in results")
            try:
                cocoDt = cocoGt.loadRes(result_files[metric])
            except IndexError:
                print("The testing results of the whole dataset is empty.")
                break

            iou_type = "bbox" if metric == "proposal" else metric
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids

            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            if classwise:  # Compute per-category AP
                # Compute per-category AP
                # from https://github.com/facebookresearch/detectron2/
                precisions = cocoEval.eval["precision"]
                # precision: (iou, recall, cls, area range, max dets)
                assert len(self.cat_ids) == precisions.shape[2]

                results_per_category = []
                for idx, catId in enumerate(self.cat_ids):
                    # area range index 0: all area ranges
                    # max dets index -1: typically 100 per image
                    nm = self.coco.loadCats(catId)[0]
                    precision = precisions[:, :, idx, 0, -1]
                    precision = precision[precision > -1]
                    if precision.size:
                        ap = np.mean(precision)
                    else:
                        ap = float("nan")
                    results_per_category.append((f'{nm["name"]}', f"{float(ap):0.3f}"))

                num_columns = min(6, len(results_per_category) * 2)
                results_flatten = list(itertools.chain(*results_per_category))
                headers = ["category", "AP"] * (num_columns // 2)
                results_2d = itertools.zip_longest(
                    *[results_flatten[i::num_columns] for i in range(num_columns)]
                )
                table_data = [headers]
                table_data += [result for result in results_2d]
                table = AsciiTable(table_data)

            metric_items = ["mAP", "mAP_50", "mAP_75", "mAP_s", "mAP_m", "mAP_l"]
            for i in range(len(metric_items)):
                key = f"{metric}_{metric_items[i]}"
                val = float(f"{cocoEval.stats[i]:.3f}")
                eval_results[key] = val
            ap = cocoEval.stats[:6]
            eval_results[f"{metric}_mAP_copypaste"] = (
                f"{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} " f"{ap[4]:.3f} {ap[5]:.3f}"
            )

        return eval_results

    def cosine_distance(self, a, b):
        """calculate the cos distance of two vectors

        Args:
            a (list): a vector
            b (list): b vector

        Returns:
            int: cos distance
        """
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)

        similiarity = (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]) / (a_norm * b_norm)
        dist = 1.0 - similiarity
        return dist

    def height_calculate(self):
        objects = self.get_confusion_matrix_indexes_json_gt(mask_type="footprint")

        dataset_gt_heights, dataset_pred_heights = [], []
        for ori_image_name in self.ori_image_name_list:
            if ori_image_name not in objects.keys():
                continue

            dataset_gt_heights += objects[ori_image_name]["gt_heights"]
            dataset_pred_heights += objects[ori_image_name]["pred_heights"]

        dataset_gt_heights = np.array(dataset_gt_heights)
        dataset_pred_heights = np.array(dataset_pred_heights)

        rmse = np.sqrt(np.sum((dataset_gt_heights - dataset_pred_heights) ** 2) / len(dataset_gt_heights))
        mae = np.sum(np.absolute(dataset_gt_heights - dataset_pred_heights)) / len(dataset_gt_heights) 

        return mae, rmse

    def parse_ann_offset(self, gt_data_path):
        import json

        gt = []
        json_file = open(gt_data_path, "r")
        content = json_file.read()
        json_content = json.loads(content)
        annotations = json_content["annotations"]
        images = json_content["images"]
        images_ids = []
        images_filenames = []
        for item in images:
            images_ids.append(item["id"])
            images_filenames.append(item["file_name"])
        id_2_filename = dict(zip(images_ids, images_filenames))
        for image_id in images_ids:
            gt_offset_angles = []
            for ann in annotations:
                if (
                    "offset" in ann.keys()
                    and ann["offset"] != [0, 0]
                    and ann["image_id"] == image_id
                ):
                    offset = ann["offset"]
                    z = math.sqrt(offset[0] ** 2 + offset[1] ** 2)
                    gt_offset_angles.append([float(offset[1]) / z, float(offset[0]) / z])

            if len(gt_offset_angles) > 0:
                gt.append(
                    dict(
                        filename=id_2_filename[image_id],
                        offset_angle=np.array(gt_offset_angles, dtype=np.float32).mean(axis=0),
                    )
                )
            else:
                gt.append(
                    dict(
                        filename=id_2_filename[image_id],
                        offset_angle=[1.0, 0.0],
                    )
                )

        return gt

    def parse_ann_nadir(self, gt_data_path):
        import json

        gt = []
        json_file = open(gt_data_path, "r")
        content = json_file.read()
        json_content = json.loads(content)
        annotations = json_content["annotations"]
        images = json_content["images"]
        images_ids = []
        images_filenames = []
        for item in images:
            images_ids.append(item["id"])
            images_filenames.append(item["file_name"])
        id_2_filename = dict(zip(images_ids, images_filenames))
        for image_id in images_ids:
            gt_nadir_angles = []
            for ann in annotations:
                if (
                    "offset" in ann.keys()
                    and "building_height" in ann.keys()
                    and ann["image_id"] == image_id
                ):
                    offset_x, offset_y = ann["offset"]
                    norm = offset_x**2 + offset_y**2
                    height = ann["building_height"]
                    if height != 0 and norm != 0:
                        angle = math.sqrt(norm) * self.resolution / float(height)
                    gt_nadir_angles.append(angle)

            if len(gt_nadir_angles) > 0:
                gt.append(
                    dict(
                        filename=id_2_filename[image_id],
                        nadir_angle=np.array(gt_nadir_angles, dtype=np.float32).mean(axis=0),
                    )
                )
            else:
                gt.append(
                    dict(
                        filename=id_2_filename[image_id],
                        nadir_angle=1,
                    )
                )

        return gt

    def vector2angle(self, vector):
        length = np.sqrt(vector[0] ** 2 + vector[1] ** 2)
        sin = vector[1] / length
        cos = vector[0] / length
        angle = math.atan2(sin, cos)  # 返回弧度值
        angle = math.degrees(angle)
        # 转换为0-360°
        if angle < 0:
            angle += 360
        return angle

    def offset_angle_evaluate(self):
        ann = self.anno_file
        gt = self.parse_ann_offset(ann)
        pkl = mmcv.load(self.pkl_file)
        gts = []
        preds = []
        for i in range(len(pkl)):
            vector_pred = pkl[i][6]
            vector_gt = gt[i]["offset_angle"]
            angle_pred = self.vector2angle(vector_pred)
            angle_gt = self.vector2angle(vector_gt)
            gts.append(angle_gt)
            preds.append(angle_pred)
        gts = np.array(gts)
        preds = np.array(preds)
        error = np.abs(np.subtract(gts, preds))
        return np.mean(error)

    def nadir_angle_evaluate(self):
        ann = self.anno_file
        gt = self.parse_ann_nadir(ann)
        pkl = mmcv.load(self.pkl_file)
        gts = []
        preds = []
        for i in range(len(pkl)):
            angle_pred = pkl[i][7]
            angle_gt = gt[i]["nadir_angle"]
            gts.append(angle_gt)
            preds.append(angle_pred)
        gts = np.array(gts)
        preds = np.array(preds)
        error = np.abs(np.subtract(gts, preds))
        return np.mean(error)

    def offset_error_vector(self, title="demo", show_polar=False):
        objects = self.get_confusion_matrix_indexes_json_gt(mask_type="footprint")

        dataset_gt_offsets, dataset_pred_offsets = [], []
        for ori_image_name in self.ori_image_name_list:
            if ori_image_name not in objects.keys():
                continue

            dataset_gt_offsets += objects[ori_image_name]["gt_offsets"]
            dataset_pred_offsets += objects[ori_image_name]["pred_offsets"]

        dataset_gt_offsets = np.array(dataset_gt_offsets)
        dataset_pred_offsets = np.array(dataset_pred_offsets)

        error_vectors = dataset_gt_offsets - dataset_pred_offsets

        EPE = np.sqrt(error_vectors[..., 0] ** 2 + error_vectors[..., 1] ** 2)
        gt_angle = np.arctan2(dataset_gt_offsets[..., 1], dataset_gt_offsets[..., 0])
        gt_length = np.sqrt(dataset_gt_offsets[..., 1] ** 2 + dataset_gt_offsets[..., 0] ** 2)

        pred_angle = np.arctan2(dataset_pred_offsets[..., 1], dataset_pred_offsets[..., 0])
        pred_length = np.sqrt(dataset_pred_offsets[..., 1] ** 2 + dataset_pred_offsets[..., 0] ** 2)

        AE = np.abs(gt_angle - pred_angle)

        aEPE = EPE.mean()
        aAE = AE.mean()

        cos_distance = self.cosine_distance(dataset_gt_offsets, dataset_pred_offsets)
        average_cos_distance = cos_distance.mean()

        eval_results = {"aEPE": aEPE, "aAE": aAE}

        if self.show:
            r = gt_length - pred_length
            angle = np.abs((gt_angle - pred_angle))
            max_r = np.percentile(r, 95)
            min_r = np.percentile(r, 0.01)

            fig = plt.figure(figsize=(7, 7))
            ax = plt.gca(projection="polar")
            ax.set_thetagrids(np.arange(0.0, 360.0, 15.0))
            ax.set_thetamin(0.0)
            ax.set_thetamax(360.0)
            ax.set_rgrids(np.arange(min_r, max_r, max_r / 10))
            ax.set_rlabel_position(0.0)
            ax.set_rlim(0, max_r)
            plt.setp(ax.get_yticklabels(), fontsize=6)
            ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
            ax.set_axisbelow("True")

            plt.scatter(angle, r, s=2.0)
            plt.title(title + " offset error distribution", fontsize=10)

            plt.savefig(
                os.path.join(
                    self.output_dir,
                    "{}_offset_error_polar_evaluation.{}".format(title, self.out_file_format),
                ),
                bbox_inches="tight",
                dpi=600,
                pad_inches=0.1,
            )

            plt.clf()

            max_r = np.percentile(r, 99.99)
            min_r = np.percentile(r, 0.01)
            plt.hist(
                r,
                bins=np.arange(min_r, max_r, (int(max_r) - int(min_r)) // 40),
                histtype="bar",
                facecolor="dodgerblue",
                alpha=0.75,
                rwidth=0.9,
            )
            plt.title(title + " Length Error Distribution", fontsize=10)
            plt.xlim([min_r - 5, max_r + 5])
            plt.xlabel("Error")
            plt.ylabel("Num")
            plt.yscale("log")
            plt.savefig(
                os.path.join(
                    self.output_dir,
                    "{}_offset_error_length_hist_evaluation.{}".format(title, self.out_file_format),
                ),
                bbox_inches="tight",
                dpi=600,
                pad_inches=0.1,
            )

            plt.clf()

            max_angle = angle.max() * 180.0 / np.pi
            min_angle = angle.min() * 180.0 / np.pi
            plt.hist(
                r,
                bins=np.arange(min_angle, max_angle, (max_angle - min_angle) // 80),
                histtype="bar",
                facecolor="dodgerblue",
                alpha=0.75,
                rwidth=0.9,
            )
            plt.title(title + " Angle Error Distribution", fontsize=10)
            plt.xlim([min_angle - 20, max_angle])
            plt.xlabel("Error")
            plt.ylabel("Num")
            plt.yscale("log")
            plt.savefig(
                os.path.join(
                    self.output_dir,
                    "{}_offset_error_angle_hist_evaluation.{}".format(title, self.out_file_format),
                ),
                bbox_inches="tight",
                dpi=600,
                pad_inches=0.1,
            )

            plt.clf()

        return eval_results

    def direct_footprint_evaluate(self):
        objects = self.get_confusion_matrix_indexes_direct_footprint()
        (
            dataset_gt_TP_indexes,
            dataset_pred_TP_indexes,
            dataset_gt_FN_indexes,
            dataset_pred_FP_indexes,
        ) = ([], [], [], [])
        for ori_image_name in self.ori_image_name_list:
            if ori_image_name not in objects.keys():
                continue

            gt_TP_indexes = objects[ori_image_name]["gt_TP_indexes"]
            pred_TP_indexes = objects[ori_image_name]["pred_TP_indexes"]
            gt_FN_indexes = objects[ori_image_name]["gt_FN_indexes"]
            pred_FP_indexes = objects[ori_image_name]["pred_FP_indexes"]

            dataset_gt_TP_indexes += gt_TP_indexes
            dataset_pred_TP_indexes += pred_TP_indexes
            dataset_gt_FN_indexes += gt_FN_indexes
            dataset_pred_FP_indexes += pred_FP_indexes

        TP = len(dataset_gt_TP_indexes)
        FN = len(dataset_gt_FN_indexes)
        FP = len(dataset_pred_FP_indexes)
        Precision = float(TP) / (float(TP) + float(FP))
        Recall = float(TP) / (float(TP) + float(FN))

        F1_score = (2 * Precision * Recall) / (Precision + Recall)
        eval_results = {
            "F1_score": F1_score,
            "Precision": Precision,
            "Recall": Recall,
            "TP": TP,
            "FN": FN,
            "FP": FP,
        }
        return eval_results

    def segmentation_evaluate(self, mask_types=["roof", "footprint"]):
        """evaluation for segmentation (F1 Score, Precision, Recall)

        Args:
            mask_types (list, optional): evaluate which object (roof or footprint). Defaults to ['roof', 'footprint'].

        Returns:
            dict: evaluation results
        """
        eval_results = dict()
        for mask_type in mask_types:
            print(f"========== Processing {mask_type} segmentation ==========")
            objects = self.get_confusion_matrix_indexes(mask_type=mask_type)

            (
                dataset_gt_TP_indexes,
                dataset_pred_TP_indexes,
                dataset_gt_FN_indexes,
                dataset_pred_FP_indexes,
            ) = ([], [], [], [])
            for ori_image_name in self.ori_image_name_list:
                if ori_image_name not in objects.keys():
                    continue

                gt_TP_indexes = objects[ori_image_name]["gt_TP_indexes"]
                pred_TP_indexes = objects[ori_image_name]["pred_TP_indexes"]
                gt_FN_indexes = objects[ori_image_name]["gt_FN_indexes"]
                pred_FP_indexes = objects[ori_image_name]["pred_FP_indexes"]

                dataset_gt_TP_indexes += gt_TP_indexes
                dataset_pred_TP_indexes += pred_TP_indexes
                dataset_gt_FN_indexes += gt_FN_indexes
                dataset_pred_FP_indexes += pred_FP_indexes

            TP = len(dataset_gt_TP_indexes)
            FN = len(dataset_gt_FN_indexes)
            FP = len(dataset_pred_FP_indexes)

            Precision = float(TP) / (float(TP) + float(FP)) if 0 != (float(TP) + float(FP)) else 0.0
            Recall = float(TP) / (float(TP) + float(FN)) if 0 != (float(TP) + float(FN)) else 0.0

            F1_score = (2 * Precision * Recall) / (Precision + Recall) if 0 != (Precision + Recall) else 0.0

            eval_results[mask_type] = {
                "F1_score": F1_score,
                "Precision": Precision,
                "Recall": Recall,
                "TP": TP,
                "FN": FN,
                "FP": FP,
            }

        return eval_results

    def get_confusion_matrix_indexes(self, mask_type="footprint"):
        if mask_type == "footprint":
            gt_csv_parser = bstool.CSVParse(self.gt_footprint_csv_file)
            pred_csv_parser = bstool.CSVParse(self.footprint_csv_file)
        else:
            gt_csv_parser = bstool.CSVParse(self.gt_roof_csv_file)
            pred_csv_parser = bstool.CSVParse(self.roof_csv_file)

        self.ori_image_name_list = gt_csv_parser.image_name_list

        gt_objects = gt_csv_parser.objects
        pred_objects = pred_csv_parser.objects

        objects = dict()

        for ori_image_name in self.ori_image_name_list:
            buildings = dict()

            gt_buildings = gt_objects[ori_image_name]
            pred_buildings = pred_objects[ori_image_name]

            gt_polygons = [gt_building["polygon"] for gt_building in gt_buildings]
            pred_polygons = [pred_building["polygon"] for pred_building in pred_buildings]

            gt_polygons_origin = gt_polygons[:]
            pred_polygons_origin = pred_polygons[:]

            if len(gt_polygons) == 0 or len(pred_polygons) == 0:
                print(
                    f"Skip this image: {ori_image_name}, because length gt_polygons or length pred_polygons is zero"
                )
                continue

            gt_offsets = [gt_building["offset"] for gt_building in gt_buildings]
            pred_offsets = [pred_building["offset"] for pred_building in pred_buildings]

            gt_heights = [gt_building["height"] for gt_building in gt_buildings]
            pred_heights = [pred_building["height"] for pred_building in pred_buildings]

            angles = []
            for gt_offset, gt_height in zip(gt_offsets, gt_heights):
                offset_x, offset_y = gt_offset
                angle = math.atan2(math.sqrt(offset_x**2 + offset_y**2) * 0.6, gt_height)
                angles.append(angle)

            height_angle = np.array(angles).mean()

            gt_polygons = geopandas.GeoSeries(gt_polygons)
            pred_polygons = geopandas.GeoSeries(pred_polygons)

            gt_df = geopandas.GeoDataFrame(
                {"geometry": gt_polygons, "gt_df": range(len(gt_polygons))}
            )
            pred_df = geopandas.GeoDataFrame(
                {"geometry": pred_polygons, "pred_df": range(len(pred_polygons))}
            )

            gt_df = gt_df.loc[~gt_df.geometry.is_empty]
            pred_df = pred_df.loc[~pred_df.geometry.is_empty]

            res_intersection = geopandas.overlay(gt_df, pred_df, how="intersection")

            iou = np.zeros((len(pred_polygons), len(gt_polygons)))
            for idx, row in res_intersection.iterrows():
                gt_idx = row.gt_df
                pred_idx = row.pred_df

                inter = row.geometry.area
                union = pred_polygons[pred_idx].area + gt_polygons[gt_idx].area

                iou[pred_idx, gt_idx] = inter / (union - inter + 1.0)

            iou_indexes = np.argwhere(iou >= 0.5)

            gt_TP_indexes = list(iou_indexes[:, 1])
            pred_TP_indexes = list(iou_indexes[:, 0])

            gt_FN_indexes = list(set(range(len(gt_polygons))) - set(gt_TP_indexes))
            pred_FP_indexes = list(set(range(len(pred_polygons))) - set(pred_TP_indexes))

            buildings["gt_iou"] = np.max(iou, axis=0)

            buildings["gt_TP_indexes"] = gt_TP_indexes
            buildings["pred_TP_indexes"] = pred_TP_indexes
            buildings["gt_FN_indexes"] = gt_FN_indexes
            buildings["pred_FP_indexes"] = pred_FP_indexes

            buildings["gt_offsets"] = np.array(gt_offsets)[gt_TP_indexes].tolist()
            buildings["pred_offsets"] = np.array(pred_offsets)[pred_TP_indexes].tolist()

            buildings["gt_heights"] = np.array(gt_heights)[gt_TP_indexes].tolist()
            buildings["pred_heights"] = np.array(pred_heights)[pred_TP_indexes].tolist()

            buildings["gt_polygons"] = gt_polygons
            buildings["pred_polygons"] = pred_polygons

            buildings["gt_polygons_matched"] = np.array(gt_polygons_origin)[gt_TP_indexes].tolist()
            buildings["pred_polygons_matched"] = np.array(pred_polygons_origin)[
                pred_TP_indexes
            ].tolist()

            buildings["height_angle"] = height_angle

            objects[ori_image_name] = buildings

        return objects

    def get_confusion_matrix_indexes_direct_footprint(self):
        gt_csv_parser = bstool.CSVParse(self.gt_footprint_csv_file)
        pred_csv_parser = bstool.CSVParse(self.footprint_direct_csv_file)

        self.ori_image_name_list = gt_csv_parser.image_name_list

        gt_objects = gt_csv_parser.objects
        pred_objects = pred_csv_parser.objects

        objects = dict()

        for ori_image_name in self.ori_image_name_list:
            buildings = dict()

            gt_buildings = gt_objects[ori_image_name]
            pred_buildings = pred_objects[ori_image_name]

            gt_polygons = [gt_building["polygon"] for gt_building in gt_buildings]
            pred_polygons = [pred_building["polygon"] for pred_building in pred_buildings]

            gt_polygons_origin = gt_polygons[:]
            pred_polygons_origin = pred_polygons[:]

            if len(gt_polygons) == 0 or len(pred_polygons) == 0:
                print(
                    f"Skip this image: {ori_image_name}, because length gt_polygons or length pred_polygons is zero"
                )
                continue

            gt_offsets = [gt_building["offset"] for gt_building in gt_buildings]
            pred_offsets = [pred_building["offset"] for pred_building in pred_buildings]

            gt_heights = [gt_building["height"] for gt_building in gt_buildings]
            pred_heights = [pred_building["height"] for pred_building in pred_buildings]

            angles = []
            for gt_offset, gt_height in zip(gt_offsets, gt_heights):
                offset_x, offset_y = gt_offset
                angle = math.atan2(math.sqrt(offset_x**2 + offset_y**2) * 0.6, gt_height)
                angles.append(angle)

            height_angle = np.array(angles).mean()

            gt_polygons = geopandas.GeoSeries(gt_polygons)
            pred_polygons = geopandas.GeoSeries(pred_polygons)

            gt_df = geopandas.GeoDataFrame(
                {"geometry": gt_polygons, "gt_df": range(len(gt_polygons))}
            )
            pred_df = geopandas.GeoDataFrame(
                {"geometry": pred_polygons, "pred_df": range(len(pred_polygons))}
            )

            gt_df = gt_df.loc[~gt_df.geometry.is_empty]
            pred_df = pred_df.loc[~pred_df.geometry.is_empty]

            res_intersection = geopandas.overlay(gt_df, pred_df, how="intersection")

            iou = np.zeros((len(pred_polygons), len(gt_polygons)))
            for idx, row in res_intersection.iterrows():
                gt_idx = row.gt_df
                pred_idx = row.pred_df

                inter = row.geometry.area
                union = pred_polygons[pred_idx].area + gt_polygons[gt_idx].area

                iou[pred_idx, gt_idx] = inter / (union - inter + 1.0)

            iou_indexes = np.argwhere(iou >= 0.5)

            gt_TP_indexes = list(iou_indexes[:, 1])
            pred_TP_indexes = list(iou_indexes[:, 0])

            gt_FN_indexes = list(set(range(len(gt_polygons))) - set(gt_TP_indexes))
            pred_FP_indexes = list(set(range(len(pred_polygons))) - set(pred_TP_indexes))

            buildings["gt_iou"] = np.max(iou, axis=0)

            buildings["gt_TP_indexes"] = gt_TP_indexes
            buildings["pred_TP_indexes"] = pred_TP_indexes
            buildings["gt_FN_indexes"] = gt_FN_indexes
            buildings["pred_FP_indexes"] = pred_FP_indexes

            buildings["gt_offsets"] = np.array(gt_offsets)[gt_TP_indexes].tolist()
            buildings["pred_offsets"] = np.array(pred_offsets)[pred_TP_indexes].tolist()

            buildings["gt_heights"] = np.array(gt_heights)[gt_TP_indexes].tolist()
            buildings["pred_heights"] = np.array(pred_heights)[pred_TP_indexes].tolist()

            buildings["gt_polygons"] = gt_polygons
            buildings["pred_polygons"] = pred_polygons

            buildings["gt_polygons_matched"] = np.array(gt_polygons_origin)[gt_TP_indexes].tolist()
            buildings["pred_polygons_matched"] = np.array(pred_polygons_origin)[
                pred_TP_indexes
            ].tolist()

            buildings["height_angle"] = height_angle

            objects[ori_image_name] = buildings

        return objects

    # use
    def get_confusion_matrix_indexes_json_gt(self, mask_type="footprint"):
        if mask_type == "footprint":
            gt_coco_parser = bstool.COCOParse(self.anno_file)
            pred_csv_parser = bstool.CSVParse(self.footprint_csv_file)
        else:
            raise (NotImplementedError)

        self.ori_image_name_list = pred_csv_parser.image_name_list

        # gt_objects = gt_csv_parser.objects
        pred_objects = pred_csv_parser.objects

        objects = dict()

        for ori_image_name in self.ori_image_name_list:
            buildings = dict()
            try:
                gt_buildings = gt_coco_parser(ori_image_name + ".png")
            except:
                gt_buildings = gt_coco_parser(ori_image_name + ".jpg")
            pred_buildings = pred_objects[ori_image_name]

            gt_polygons = [
                bstool.mask2polygon(gt_building["footprint_mask"]).buffer(0)
                for gt_building in gt_buildings
            ]
            pred_polygons = [pred_building["polygon"] for pred_building in pred_buildings]

            gt_polygons_origin = gt_polygons[:]
            pred_polygons_origin = pred_polygons[:]

            if len(gt_polygons) == 0 or len(pred_polygons) == 0:
                print(
                    f"Skip this image: {ori_image_name}, because length gt_polygons or length pred_polygons is zero"
                )
                continue

            gt_offsets = [gt_building["offset"] for gt_building in gt_buildings]
            pred_offsets = [pred_building["offset"] for pred_building in pred_buildings]

            gt_heights = [gt_building["building_height"] for gt_building in gt_buildings]
            pred_heights = [pred_building["height"] for pred_building in pred_buildings]

            gt_polygons = geopandas.GeoSeries(gt_polygons)
            pred_polygons = geopandas.GeoSeries(pred_polygons)

            gt_df = geopandas.GeoDataFrame(
                {"geometry": gt_polygons, "gt_df": range(len(gt_polygons))}
            )
            pred_df = geopandas.GeoDataFrame(
                {"geometry": pred_polygons, "pred_df": range(len(pred_polygons))}
            )

            gt_df = gt_df.loc[~gt_df.geometry.is_empty]
            pred_df = pred_df.loc[~pred_df.geometry.is_empty]

            res_intersection = geopandas.overlay(gt_df, pred_df, how="intersection")

            iou = np.zeros((len(pred_polygons), len(gt_polygons)))
            for idx, row in res_intersection.iterrows():
                gt_idx = row.gt_df
                pred_idx = row.pred_df

                inter = row.geometry.area
                union = pred_polygons[pred_idx].area + gt_polygons[gt_idx].area

                iou[pred_idx, gt_idx] = inter / (union - inter + 1.0)

            iou_indexes = np.argwhere(iou >= 0.5)

            gt_TP_indexes = list(iou_indexes[:, 1])
            pred_TP_indexes = list(iou_indexes[:, 0])

            gt_FN_indexes = list(set(range(len(gt_polygons))) - set(gt_TP_indexes))
            pred_FP_indexes = list(set(range(len(pred_polygons))) - set(pred_TP_indexes))

            buildings["gt_iou"] = np.max(iou, axis=0)

            buildings["gt_TP_indexes"] = gt_TP_indexes
            buildings["pred_TP_indexes"] = pred_TP_indexes
            buildings["gt_FN_indexes"] = gt_FN_indexes
            buildings["pred_FP_indexes"] = pred_FP_indexes

            buildings["gt_offsets"] = np.array(gt_offsets)[gt_TP_indexes].tolist()
            buildings["pred_offsets"] = np.array(pred_offsets)[pred_TP_indexes].tolist()

            buildings["gt_heights"] = np.array(gt_heights)[gt_TP_indexes].tolist()
            buildings["pred_heights"] = np.array(pred_heights)[pred_TP_indexes].tolist()

            buildings["gt_polygons"] = gt_polygons
            buildings["pred_polygons"] = pred_polygons

            buildings["gt_polygons_matched"] = np.array(gt_polygons_origin)[gt_TP_indexes].tolist()
            buildings["pred_polygons_matched"] = np.array(pred_polygons_origin)[
                pred_TP_indexes
            ].tolist()

            objects[ori_image_name] = buildings

        return objects

    def visualization_boundary(
        self,
        image_dir,
        vis_dir,
        mask_types=["roof", "footprint","direct_footprint"],
        with_iou=False,
        with_gt=True,
        with_only_pred=False,
        with_image=True,
    ):
        colors = {
            # "gt_TP": (0, 255, 0),
            "pred_TP": (255, 255, 0),
            "FP": (0, 255, 255),
            "FN": (255, 0, 0),
        }
        for mask_type in mask_types:
            if mask_type == 'direct_footprint':
                objects = self.get_confusion_matrix_indexes_direct_footprint()
            else:
                objects = self.get_confusion_matrix_indexes(mask_type=mask_type)
            for image_name in os.listdir(image_dir):
                image_basename = bstool.get_basename(image_name)
                image_file = os.path.join(image_dir, image_name)

                output_file = os.path.join(vis_dir, mask_type, image_name)
                bstool.mkdir_or_exist(os.path.join(vis_dir, mask_type))

                if with_image:
                    img = cv2.imread(image_file)
                else:
                    img = bstool.generate_image(1024, 1024, color=(255, 255, 255))

                if image_basename not in objects:
                    continue

                building = objects[image_basename]

                if with_only_pred == False:
                    for idx, gt_polygon in enumerate(building["gt_polygons"]):
                        iou = building["gt_iou"][idx]
                        if idx in building["gt_TP_indexes"]:
                            # color = colors["gt_TP"][::-1]
                            continue
                            if not with_gt:
                                continue
                        else:
                            color = colors["FN"][::-1]

                        if gt_polygon.geom_type != "Polygon":
                            continue

                        img = bstool.draw_mask_boundary(
                            img, bstool.polygon2mask(gt_polygon), color=color
                        )
                        if with_iou:
                            img = bstool.draw_iou(img, gt_polygon, iou, color=color)

                for idx, pred_polygon in enumerate(building["pred_polygons"]):
                    if with_only_pred == False:
                        if idx in building["pred_TP_indexes"]:
                            color = colors["pred_TP"][::-1]
                        else:
                            color = colors["FP"][::-1]
                    else:
                        if with_image:
                            color = colors["pred_TP"][::-1]
                        else:
                            color = (0, 0, 255)

                    if pred_polygon.geom_type != "Polygon":
                        continue

                    img = bstool.draw_mask_boundary(
                        img, bstool.polygon2mask(pred_polygon), color=color
                    )
                cv2.imwrite(output_file, img)

    def visualization_offset(self, image_dir, vis_dir, with_footprint=True):
        print("========== generation vis images with offset ==========")
        if with_footprint:
            image_dir = os.path.join(vis_dir, "..", "boundary", "footprint")
            vis_dir = vis_dir + "_with_footprint"

        bstool.mkdir_or_exist(vis_dir)

        colors = {
            "gt_matched": (0, 255, 0),
            "pred_matched": (255, 255, 0),
            "pred_un_matched": (0, 255, 255),
            "gt_un_matched": (255, 0, 0),
        }
        objects = self.get_confusion_matrix_indexes(mask_type="roof")

        for image_name in os.listdir(image_dir):
            image_basename = bstool.get_basename(image_name)
            image_file = os.path.join(image_dir, image_name)

            output_file = os.path.join(vis_dir, image_name)

            img = cv2.imread(image_file)

            if image_basename not in objects:
                continue

            building = objects[image_basename]

            height_angle = building["height_angle"]

            img = bstool.draw_height_angle(img, height_angle)

            for gt_polygon, gt_offset, pred_polygon, pred_offset, gt_height in zip(
                building["gt_polygons_matched"],
                building["gt_offsets"],
                building["pred_polygons_matched"],
                building["pred_offsets"],
                building["gt_heights"],
            ):
                gt_roof_centroid = list(gt_polygon.centroid.coords)[0]
                pred_roof_centroid = list(pred_polygon.centroid.coords)[0]

                gt_footprint_centroid = [
                    coordinate - offset for coordinate, offset in zip(gt_roof_centroid, gt_offset)
                ]
                pred_footprint_centroid = [
                    coordinate - offset
                    for coordinate, offset in zip(pred_roof_centroid, pred_offset)
                ]

                xoffset, yoffset = gt_offset
                transform_matrix = [1, 0, 0, 1, -xoffset, -yoffset]
                gt_footprint_polygon = affinity.affine_transform(gt_polygon, transform_matrix)

                xoffset, yoffset = pred_offset
                transform_matrix = [1, 0, 0, 1, -xoffset, -yoffset]
                pred_footprint_polygon = affinity.affine_transform(pred_polygon, transform_matrix)

                intersection = gt_footprint_polygon.intersection(pred_footprint_polygon).area
                union = gt_footprint_polygon.union(pred_footprint_polygon).area

                iou = intersection / (union - intersection + 1.0)

                if iou >= 0.5:
                    gt_color = colors["gt_matched"][::-1]
                    pred_color = colors["pred_matched"][::-1]
                else:
                    gt_color = colors["gt_un_matched"][::-1]
                    pred_color = colors["pred_un_matched"][::-1]

                img = bstool.draw_offset_arrow(
                    img, gt_roof_centroid, gt_footprint_centroid, color=gt_color
                )
                img = bstool.draw_offset_arrow(
                    img, pred_roof_centroid, pred_footprint_centroid, color=pred_color
                )

            cv2.imwrite(output_file, img)


def mkdir_or_exist(dir_name, mode=0o777):
    """make of check the dir

    Args:
        dir_name (str): directory name
        mode (str, optional): authority of mkdir. Defaults to 0o777.
    """
    if dir_name == "":
        return
    dir_name = os.path.expanduser(dir_name)
    if six.PY3:
        os.makedirs(dir_name, mode=mode, exist_ok=True)
    else:
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name, mode=mode)


def write_results2csv(results, meta_info=None):
    """Write the evaluation results to csv file

    Args:
        results (list): list of result
        meta_info (dict, optional): The meta info about the evaluation (file path of ground truth etc.). Defaults to None.
    """
    # print("meta_info: ", meta_info)
    segmentation_eval_results = results[0]
    with open(meta_info["summary_file"], "w") as summary:
        csv_writer = csv.writer(summary, delimiter=",")
        csv_writer.writerow(["Meta Info"])
        csv_writer.writerow(["model", meta_info["model"]])
        csv_writer.writerow(["anno_file", meta_info["anno_file"]])
        csv_writer.writerow(["gt_roof_csv_file", meta_info["gt_roof_csv_file"]])
        csv_writer.writerow(["gt_footprint_csv_file", meta_info["gt_footprint_csv_file"]])
        # csv_writer.writerow(['vis_dir', meta_info['vis_dir']])
        csv_writer.writerow([""])
        for mask_type in ["roof", "footprint"]:
            csv_writer.writerow([mask_type])
            csv_writer.writerow([segmentation_eval_results[mask_type]])
            csv_writer.writerow(["F1 Score", segmentation_eval_results[mask_type]["F1_score"]])
            csv_writer.writerow(["Precision", segmentation_eval_results[mask_type]["Precision"]])
            csv_writer.writerow(["Recall", segmentation_eval_results[mask_type]["Recall"]])
            csv_writer.writerow(["True Positive", segmentation_eval_results[mask_type]["TP"]])
            csv_writer.writerow(["False Positive", segmentation_eval_results[mask_type]["FP"]])
            csv_writer.writerow(["False Negative", segmentation_eval_results[mask_type]["FN"]])
            csv_writer.writerow([""])

        csv_writer.writerow([""])


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet eval on semantic segmentation")
    parser.add_argument("pkl_file_path", help="pkl file for eval")
    parser.add_argument(
        "csv_save_path", help="root to save csv file, if path not exists, will create"
    )
    parser.add_argument(
        "--version", type=str, default="bc_v100.01.09", help="model name (version) for evaluation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="loft-foa-oa-na-fro-h",
        help="full model name for evaluation",
    )
    parser.add_argument(
        "--city", type=str, default="BONAI", help="dataset city for evaluation"
    )

    args = parser.parse_args()

    return args


def get_model_shortname(model_name):
    return "bonai" + "_" + model_name.split("_")[1]


class EvaluationParameters:
    def __init__(self, city, model, pkl, csv_root):
        # flags
        self.with_vis = False
        self.with_only_vis = False
        self.with_only_pred = False
        self.with_image = True
        self.with_offset = True
        self.save_merged_csv = False

        city_types_to_full = {
            "omnicity": "OmniCityView3WithOffset",
            "hk": "hongkong",
            "bonai_hk": "bonai_hongkong",
        }

        # basic info
        self.city = city
        self.model = model
        self.score_threshold = 0.4

        self.dataset_root = "./data"
        
        # self.with_vis = True # whether draw when eval
        # self.with_only_vis = True # only draw 

        # Default dataset
        # dataset file
        self.anno_file = f"{self.dataset_root}/BONAI/coco/bonai_shanghai_xian_test_roof.json"
        self.test_image_dir = f"{self.dataset_root}/BONAI/test/images"
        # csv ground truth files
        self.gt_roof_csv_file = f"{self.dataset_root}/BONAI/csv/shanghai_xian_v3_merge_val_roof_crop1024_gt_minarea500.csv"
        self.gt_footprint_csv_file = f"{self.dataset_root}/BONAI/csv/shanghai_xian_v3_merge_val_footprint_crop1024_gt_minarea500.csv"

        if city == 'bonai':
            print('################ Use Default City BONAI shanghai_xian for Eval ################')
        elif city in ['bonai_hk']: # For City Group
            city_full_name = city_types_to_full[city]
            print('################ Use City Group {} for Eval ################'.format(city_full_name))
            self.anno_file = f"{self.dataset_root}/combined_test/coco/{city_full_name}_test_roof.json"
            self.test_image_dir = f"{self.dataset_root}/combined_test/images/{city_full_name}/"
            self.gt_roof_csv_file = f"{self.dataset_root}/combined_test/csv/{city_full_name}_roof_gt_minarea500.csv"
            self.gt_footprint_csv_file = f"{self.dataset_root}/combined_test/csv/{city_full_name}_footprint_gt_minarea500.csv"
        elif city in ['omnicity','hk']:
            city_full_name = city_types_to_full[city]
            print('################ Use City {} for Eval ################'.format(city_full_name))
            self.test_image_dir = f"{self.dataset_root}/{city_full_name}/test/images/"
            self.anno_file = f"{self.dataset_root}/{city_full_name}/coco/{city_full_name}_test_roof.json"
            self.gt_roof_csv_file = f"{self.dataset_root}/{city_full_name}/csv/{city_full_name}_roof_gt_minarea500.csv"
            self.gt_footprint_csv_file = f"{self.dataset_root}/{city_full_name}/csv/{city_full_name}_footprint_gt_minarea500.csv"
        else:
            print('################ No Such TEST City Type: {}! ################'.format(city))
            print('################ Use Default City BONAI shanghai_xian for Eval! ################')

        # detection result files
        self.mmdetection_pkl_file = pkl
        self.save_root = csv_root

        self.csv_info = "merged" if self.save_merged_csv else "splitted"

        if not os.path.exists(self.save_root):
            os.makedirs(self.save_root)
        self.pred_roof_csv_file = os.path.join(self.save_root, "roof_pred.csv")
        self.pred_footprint_csv_file = os.path.join(self.save_root, "footprint_offset.csv")
        self.direct_footprint_csv_file = os.path.join(self.save_root, "footprint_direct.csv")

        # vis
        if self.with_vis or self.with_only_vis:
            self.vis_boundary_dir = f'{self.save_root}/vis/boundary' + ("_pred" if self.with_only_pred else "")
            self.vis_offset_dir = f'{self.save_root}/vis/offset'

        self.summary_file = self.save_root + "eval_summary_{self.csv_info}.csv"

    def post_processing(self):
        mkdir_or_exist(self.vis_boundary_dir)
        mkdir_or_exist(self.vis_offset_dir)


if __name__ == "__main__":
    args = parse_args()
    warnings.filterwarnings("ignore")
    eval_parameters = EvaluationParameters(
        city=args.city, model=args.model, pkl=args.pkl_file_path, csv_root=args.csv_save_path
    )
    # eval_parameters.post_processing()
    print(f"========== {args.model} ========== {args.city} ==========")

    pkl_file = eval_parameters.mmdetection_pkl_file
    pkl_test = mmcv.load(pkl_file)
    eval_offset = False if pkl_test[0][2] is None else True
    eval_height = False if pkl_test[0][3] is None else True
    eval_direct_footprint = False if pkl_test[0][5] is None else True
    eval_offset_angle = False if pkl_test[0][6] is None else True
    eval_naidr_angle = False if pkl_test[0][7] is None else True

    evaluation = Evaluation(
        model=eval_parameters.model,
        anno_file=eval_parameters.anno_file,
        pkl_file=pkl_file,
        gt_roof_csv_file=eval_parameters.gt_roof_csv_file,
        gt_footprint_csv_file=eval_parameters.gt_footprint_csv_file,
        roof_csv_file=eval_parameters.pred_roof_csv_file,
        footprint_csv_file=eval_parameters.pred_footprint_csv_file,
        footprint_direct_csv_file=eval_parameters.direct_footprint_csv_file,
        iou_threshold=0.1,
        score_threshold=eval_parameters.score_threshold,
        with_offset=eval_offset,
        show=False,
        save_merged_csv=eval_parameters.save_merged_csv,
    )

    if eval_parameters.with_only_vis is False:
        # evaluation
        if evaluation.dump_result:
            # calculate the F1 score
            offset = {"aEPE": []}
            offset_angle = []
            nadir_angle = []
            mae = []
            rmse = []

            if eval_offset:
                offset = evaluation.offset_error_vector()
                roof_and_cal_footprint = evaluation.segmentation_evaluate()
            else:
                roof_and_cal_footprint = evaluation.segmentation_evaluate(mask_types=["roof"])
                roof_and_cal_footprint["footprint"] = {}
                roof_and_cal_footprint["footprint"]["F1_score"] = []
                roof_and_cal_footprint["footprint"]["Precision"] = []
                roof_and_cal_footprint["footprint"]["Recall"] = []
            if eval_naidr_angle:
                nadir_angle = evaluation.nadir_angle_evaluate()
            if eval_offset_angle:
                offset_angle = evaluation.offset_angle_evaluate()
            if eval_height:
                mae, rmse = evaluation.height_calculate()
            if eval_direct_footprint:
                direct_footprint = evaluation.direct_footprint_evaluate()
            else:
                direct_footprint = {"F1_score":[],"Precision":[],"Recall":[]}

            print("roof_F1:                 ", roof_and_cal_footprint["roof"]["F1_score"])
            print("calculated_footprint_F1: ", roof_and_cal_footprint["footprint"]["F1_score"])

            print("inferenced_footprint_F1: ", direct_footprint['F1_score'])
            print("inferenced_footprint_Precision: ", direct_footprint['Precision'])
            print("inferenced_footprint_Recall: ", direct_footprint['Recall'])
        
            print("offset_EPE:              ", offset["aEPE"])
            print("height_MAE:              ", mae)
            print("height_RMSE:              ", rmse)
            print("offset_angle_mae:        ", offset_angle)
            print("nadir_angle_mae:         ", nadir_angle)

            with open(os.path.join(eval_parameters.save_root, "results.txt"), "w") as w:
                w.write("roof_F1: {}\n".format(roof_and_cal_footprint["roof"]["F1_score"]))
                w.write("roof_Precision: {}\n".format(roof_and_cal_footprint["roof"]["Precision"]))
                w.write("roof_Recall: {}\n".format(roof_and_cal_footprint["roof"]["Recall"]))

                w.write("calculated_footprint_F1: {}\n".format(roof_and_cal_footprint["footprint"]["F1_score"]))
                w.write("calculated_footprint_Precision: {}\n".format(roof_and_cal_footprint["footprint"]["Precision"]))
                w.write("calculated_footprint_Recall: {}\n".format(roof_and_cal_footprint["footprint"]["Recall"]))

                w.write("inferenced_footprint_F1: {}\n".format(direct_footprint['F1_score']))
                w.write("inferenced_footprint_Precision: {}\n".format(direct_footprint['Precision']))
                w.write("inferenced_footprint_Recall: {}\n".format(direct_footprint['Recall']))

                w.write("offset_EPE: {}\n".format(offset["aEPE"]))
                w.write("height_MAE: {}\n".format(mae))
                w.write("height_RMSE: {}\n".format(rmse))
                w.write("offset_angle_mae: {}\n".format(offset_angle))
                w.write("nadir_angle_mae: {}\n".format(nadir_angle))

        else:
            print(
                "!!!!!!!!!!!!!!!!!!!!!! ALl the results of images are empty !!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )

        # vis
        if eval_parameters.with_vis:
            # generate the vis results
            evaluation.visualization_boundary(
                image_dir=eval_parameters.test_image_dir,
                vis_dir=eval_parameters.vis_boundary_dir,
                with_gt=True,
            )
            # draw offset in the image (not used in this file)
            # for with_footprint in [True, False]:
            #     evaluation.visualization_offset(image_dir=image_dir, vis_dir=vis_offset_dir, with_footprint=with_footprint)
    else:
        # generate the vis results
        evaluation.visualization_boundary(
            image_dir=eval_parameters.test_image_dir,
            vis_dir=eval_parameters.vis_boundary_dir,
            with_gt=True,
            with_only_pred=eval_parameters.with_only_pred,
            with_image=eval_parameters.with_image,
        )
        # draw offset in the image (not used in this file)
        # for with_footprint in [False, True]:
        #     evaluation.visualization_offset(image_dir=eval_parameters.test_image_dir, vis_dir=eval_parameters.vis_offset_dir, with_footprint=with_footprint)