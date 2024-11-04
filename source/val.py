# this file is use for validation purposes
"""
Run this file by : 
python val.py --model_weights weights/best.pt --test_path ../data/val/images
"""

from utils.wbf import *
from utils.minority_optimizer import minority_optimizer_func
import os
from collections import defaultdict
import torch
import time
import argparse
from tqdm import tqdm
from ultralytics import YOLO


def calculate_iou(boxA, boxB):
    # boxA và boxB có định dạng [x1, y1, x2, y2]
    x1_A, y1_A, x2_A, y2_A = boxA
    x1_B, y1_B, x2_B, y2_B = boxB

    # Tính diện tích của các box
    area_A = (x2_A - x1_A) * (y2_A - y1_A)
    area_B = (x2_B - x1_B) * (y2_B - y1_B)

    # Tính tọa độ của hộp giới hạn chung
    x1_intersection = max(x1_A, x1_B)
    y1_intersection = max(y1_A, y1_B)
    x2_intersection = min(x2_A, x2_B)
    y2_intersection = min(y2_A, y2_B)

    # Tính diện tích của hộp giới hạn chung
    width_intersection = max(0, x2_intersection - x1_intersection)
    height_intersection = max(0, y2_intersection - y1_intersection)
    area_intersection = width_intersection * height_intersection

    # Tính IoU
    total_area = area_A + area_B - area_intersection
    iou = area_intersection / total_area if total_area > 0 else 0
    return iou


def my_normalize(box):
    img_w, img_h = 1920, 1080
    x1, y1, x2, y2, label = box

    return (
        float(float(x1) / img_w),
        float(float(y1) / img_h),
        float(float(x2) / img_w),
        float(float(y2) / img_h),
        label,
    )


if __name__ == "__main__":
    # Sử dụng argparse để truyền tham số từ dòng lệnh
    parser = argparse.ArgumentParser(
        description="Run object detection on a set of images with multiple models."
    )
    parser.add_argument(
        "--model_weights",
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

    # Tạo danh sách tên của mô hình từ đường dẫn trọng số
    model_list = [
        os.path.basename(model_weights) for model_weights in args.model_weights
    ]

    # Lưu kết quả từ từng mô hình
    predictions = {}

    # Dự đoán cho từng mô hình
    for model_weights in args.model_weights:
        model = YOLO(model_weights)  # Load model từ trọng số
        model_name = os.path.basename(model_weights)  # Lấy tên mô hình từ đường dẫn
        predictions[model_name] = {}  # Tạo mục cho từng mô hình trong predictions
        model.overrides["verbose"] = False  # thu gọn terminal khi predict

        print(f"Processing with model: {model_name}")

        # Bắt đầu tính thời gian
        start_time = time.time()

        # Dự đoán trên từng ảnh trong thư mục test
        for image_name in tqdm(
            os.listdir(args.test_path), desc=f"Processing {model_name}"
        ):
            image_path = os.path.join(args.test_path, image_name)
            results = detect_image(
                image_path, model
            )  # Hàm thực hiện dự đoán trên 1 ảnh

            # Lưu kết quả dự đoán của từng ảnh cho mô hình hiện tại
            predictions[model_name][image_name] = results

        # In thời gian xử lý của từng mô hình (Chỉ in thời gian)
        elapsed_time = time.time() - start_time
        print(f"Completed {model_name} in {elapsed_time:.2f} seconds")

    print("Start Fuse")
    fused_preds_dict = fuse(
        model_list, args.test_path, predictions, iou_thr=0.7, skip_box_thr=0.07
    )
    print("End Fuse")

    results = minority_optimizer_func(fused_preds_dict)

    # Danh sách các lớp
    classes = [
        "motorbike",
        "DHelmet",
        "DNoHelmet",
        "P1Helmet",
        "P1NoHelmet",
        "P2Helmet",
        "P2NoHelmet",
        "P0Helmet",
        "P0NoHelmet",
    ]

    # Khởi tạo biến để lưu trữ số liệu cho từng lớp
    true_positive = defaultdict(int)
    false_positive = defaultdict(int)
    false_negative = defaultdict(int)

    iou_thres = 0.7

    # Danh sách các file image
    labels_path = args.test_path.replace("/images", "/labels")

    image_files = [f for f in os.listdir(labels_path) if f.endswith(".txt")]

    for image_file in image_files:
        annotations = []
        with open(
            os.path.join(
                labels_path,
                image_file,
            ),
            "r",
        ) as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])  # Lớp (class ID)
                x_center, y_center, width, height = map(float, parts[1:])  # Tọa độ
                annotations.append((class_id, x_center, y_center, width, height))

        normalized_boxes = []

        img_name = image_file.replace(".txt", ".jpg")

        if img_name not in results:
            print(f"Error: {img_name} not in results")
            continue

        for r in results[img_name]:
            parts = r.strip().split(",")
            box = [parts[0], parts[1], parts[2], parts[3], parts[6]]
            normalized_box = my_normalize(box)
            # normalized_box = x1y1wh_2_x1y1x2y2(box)  # chuyển x1,y1,w,h --> x1,y1,x2,y2 và normalize
            normalized_boxes.append(normalized_box)

        # Tạo dictionary để tăng tốc độ tìm kiếm
        annotations_dict = {
            (x_center_ann, y_center_ann, width_ann, height_ann): class_id
            for (
                class_id,
                x_center_ann,
                y_center_ann,
                width_ann,
                height_ann,
            ) in annotations
        }

        matched_annotations = set()  # Để theo dõi nhãn nào đã được dự đoán đúng

        # So sánh từng dự đoán với nhãn thực tế
        for box in normalized_boxes:

            x1_pred, y1_pred, x2_pred, y2_pred, class_pred = box
            class_pred = int(float(class_pred))

            pred_box = float(x1_pred), float(y1_pred), float(x2_pred), float(y2_pred)

            match_found = False

            for (
                x_center_ann,
                y_center_ann,
                width_ann,
                height_ann,
            ), class_id in annotations_dict.items():
                # print(class_id, class_pred)
                if class_pred == class_id:  # Kiểm tra lớp

                    true_box = [
                        x_center_ann - width_ann / 2,
                        y_center_ann - height_ann / 2,
                        x_center_ann + width_ann / 2,
                        y_center_ann + height_ann / 2,
                    ]

                    iou = calculate_iou(pred_box, true_box)

                    if iou > iou_thres:  # Ngưỡng IoU
                        true_positive[class_pred] += 1  # Dự đoán đúng
                        matched_annotations.add(
                            (x_center_ann, y_center_ann, width_ann, height_ann)
                        )
                        match_found = True
                        break

            if not match_found:
                false_positive[class_pred] += 1  # Dự đoán sai

        for class_id, _, _, _, _ in annotations:
            if (
                x_center_ann,
                y_center_ann,
                width_ann,
                height_ann,
            ) not in matched_annotations:
                false_negative[class_id] += 1  # Nhãn thực tế không được dự đoán đúng

    # Tính toán độ chính xác và độ hồi tưởng
    precision = {}
    recall = {}

    for class_id in range(len(classes)):
        tp = true_positive[class_id]
        fp = false_positive[class_id]
        fn = false_negative[class_id]

        precision[class_id] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[class_id] = tp / (tp + fn) if (tp + fn) > 0 else 0

        print(
            f"{classes[class_id]} - Precision: {precision[class_id]:.4f}, Recall: {recall[class_id]:.4f}"
        )
