# this file is use for validation purposes
"""
Run this file by : 
python val.py --model_weights weights/best.pt --test_path ../data/val/images
"""

import os
import time
import json
import torch
import argparse
from collections import defaultdict
from tqdm import tqdm
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils.wbf import *
from utils.minority_optimizer import minority_optimizer_func
from test import create_models, get_models_predictions


if __name__ == "__main__":
    # Sử dụng argparse để truyền tham số từ dòng lệnh
    parser = argparse.ArgumentParser(
        description="Run object detection on a set of images with multiple models."
    )
    parser.add_argument(
        "--model_weights_list",
        nargs="+",  # Cho phép truyền nhiều tham số
        required=True,
        help="List of paths to the model weights files.",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="../data/val/images",
        help="Path to the directory containing images to process.",
    )
    args = parser.parse_args()

    model_weights_list = args.model_weights_list
    test_path = args.test_path

    predictions = get_models_predictions(
        args.model_weights_list, args.test_path, single_image=False
    )

    models_names_list = [os.path.basename(path) for path in model_weights_list]

    results = fuse(
        models_names_list,
        args.test_path,
        predictions,
        single_image=False,
        iou_thr=0.5,
        skip_box_thr=0.05,
    )

    results = minority_optimizer_func(results)

    ground_truth = []
    predictions = []
    images = []
    annotation_id = 1  # Khởi tạo ID cho annotation
    image_id_map = {}  # Để lưu trữ `image_id` cho từng ảnh

    label_path = test_path.replace("images", "labels")

    # Xử lý dự đoán (predictions)
    for image_name, result in results.items():

        image_id = image_id_map.setdefault(image_name, len(image_id_map) + 1)

        for pred_box in result:
            parts = pred_box.strip().split(",")
            x1, y1, x2, y2 = map(float, [parts[0], parts[1], parts[2], parts[3]])
            label, conf = int(float(parts[6])), float(parts[7])
            width, height = x2 - x1, y2 - y1
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": label,
                    "bbox": [x1, y1, width, height],
                    "score": conf,
                }
            )

    # Xử lý ground truth
    image_files = [f for f in os.listdir(label_path) if f.endswith(".txt")]

    for image_file in image_files:

        img_name = image_file.replace(".txt", ".jpg")

        if img_name not in image_id_map:
            print(f"{img_name} not in image_id_map")
            continue
        else:
            image_id = image_id_map[img_name]

        images.append({"id": image_id, "file_name": img_name})
        with open(os.path.join(label_path, image_file), "r") as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])

                # calculate and denormalize
                x1 = (x_center - width / 2) * 1920
                y1 = (y_center - height / 2) * 1080
                width *= 1920
                height *= 1080

                area = width * height  # Tính diện tích bbox
                ground_truth.append(
                    {
                        "id": annotation_id,  # Thêm ID cho annotation
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": [x1, y1, width, height],
                        "area": area,  # Thêm diện tích vào annotation
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1  # Tăng ID cho annotation tiếp theo

    # Tạo JSON định dạng COCO
    ground_truth_coco = {
        "images": images,
        "annotations": ground_truth,
        "categories": [
            {"id": 0, "name": "motorbike"},
            {"id": 1, "name": "DHelmet"},
            {"id": 2, "name": "DNoHelmet"},
            {"id": 3, "name": "P1Helmet"},
            {"id": 4, "name": "P1NoHelmet"},
            {"id": 5, "name": "P2Helmet"},
            {"id": 6, "name": "P2NoHelmet"},
            {"id": 7, "name": "P0Helmet"},
            {"id": 8, "name": "P0NoHelmet"},
        ],
    }

    with open("predictions.json", "w") as f:
        json.dump(predictions, f)

    with open("ground_truth_coco.json", "w") as f:
        json.dump(ground_truth_coco, f)

    # Đánh giá COCO
    coco_gt = COCO("ground_truth_coco.json")
    coco_dt = coco_gt.loadRes("predictions.json")
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Lấy mAP trực tiếp từ coco_eval.stats (chỉ số mAP tại ngưỡng IoU trung bình)
    mAP_50 = coco_eval.stats[1]  # mAP@[IoU=0.5]
    print(f"mAP at IoU=0.5: {mAP_50}")
