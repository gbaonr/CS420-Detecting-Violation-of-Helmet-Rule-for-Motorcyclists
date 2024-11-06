import os
import cv2
from ensemble_boxes import *
import sys


# FUNCTION TO DETECT AND FUSE
def detect_image(image_path, model):
    """
    Detect 1 image with specified model
    """
    img = cv2.imread(image_path)
    img_h, img_w = img.shape[:2]
    results = model.predict(
        source=img, save=False, stream=True, batch=8, conf=0.00001, device="cpu"
    )
    # detect with gpu, use this if gpu is available to speed up detection
    # results = model.predict(
    #     source=img, save=False, stream=True, batch=8, device="cuda:0"
    # )
    lines = []
    for result in results:
        for box in result.boxes:
            bbox = box.xyxy[0]
            score = box.conf[0].item()
            label = int(box.cls[0].item())
            # NOT normalized xyxy and h,w are image_h and image_w
            lines.append(
                f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{img_w},{img_h},{label},{score}\n"
            )
    return lines


def fuse(
    models_names_list,
    test_path,
    predictions,
    single_image=False,
    iou_thr=0.5,
    skip_box_thr=0.001,
):
    """
    - Fuse results from multiple models of all images or single image

    - Params: (model_names_list, test_path, predicitons of all models, iou_thres, skip_box_thr)
    """
    results = []
    results_dict = {}

    # Iterate through all images
    if not single_image:
        for image_name in os.listdir(test_path):
            img_boxes = []
            img_labels = []
            img_scores = []
            i_h, i_w = cv2.imread(os.path.join(test_path, image_name)).shape[
                :2
            ]  # store the image size

            for model in models_names_list:
                model_prediction = predictions[model].get(image_name, [])
                for line in model_prediction:
                    # Extract information from the line and normalize bbox
                    try:
                        x1, y1, x2, y2, img_w, img_h, label, score = line.strip().split(
                            ","
                        )
                        img_w = float(img_w)
                        img_h = float(img_h)
                        box = (
                            float(x1) / img_w,
                            float(y1) / img_h,
                            float(x2) / img_w,
                            float(y2) / img_h,
                        )
                        img_boxes.append(box)
                        img_labels.append(int(label))
                        img_scores.append(float(score))
                    except Exception as e:
                        print(
                            f"Error processing line {line} in image {image_name}: {e}"
                        )
                        continue

            # fuse boxes
            if img_boxes:
                # weighted_boxes_fusion expects lists of lists
                boxes, scores, labels = weighted_boxes_fusion(
                    [img_boxes],  # Make it a list of lists
                    [img_scores],  # Make it a list of lists
                    [img_labels],  # Make it a list of lists
                    iou_thr=iou_thr,
                    skip_box_thr=skip_box_thr,
                )

                # save results with de-normalized bounding box
                for i in range(len(boxes)):
                    if image_name not in results_dict:
                        results_dict[image_name] = []

                    results_dict[image_name].append(
                        f"{boxes[i][0]*i_w},{boxes[i][1]*i_h},{boxes[i][2]*i_w},{boxes[i][3]*i_h},{i_w},{i_h},{labels[i]},{scores[i]}\n"
                    )

    # fuse on single image
    if single_image:
        image_name = os.path.basename(test_path)
        img_boxes = []
        img_labels = []
        img_scores = []
        i_h, i_w = cv2.imread(test_path).shape[:2]  # store the image size
        for model in models_names_list:
            model_prediction = predictions[model][image_name]
            for line in model_prediction:
                # Extract information from the line and normalize bbox
                try:
                    x1, y1, x2, y2, img_w, img_h, label, score = line.strip().split(",")
                    img_w = float(img_w)
                    img_h = float(img_h)
                    box = (
                        float(x1) / img_w,
                        float(y1) / img_h,
                        float(x2) / img_w,
                        float(y2) / img_h,
                    )
                    img_boxes.append(box)
                    img_labels.append(int(label))
                    img_scores.append(float(score))
                except Exception as e:
                    print(f"Error processing line {line} in image: {e}")
                    continue
        # fuse boxes
        if img_boxes:
            # weighted_boxes_fusion expects lists of lists
            boxes, scores, labels = weighted_boxes_fusion(
                [img_boxes],  # Make it a list of lists
                [img_scores],  # Make it a list of lists
                [img_labels],  # Make it a list of lists
                iou_thr=iou_thr,
                skip_box_thr=skip_box_thr,
            )
            # save results with de-normalized bounding box
            for i in range(len(boxes)):
                if image_name not in results_dict:
                    results_dict[image_name] = []

                results_dict[image_name].append(
                    f"{boxes[i][0]*i_w},{boxes[i][1]*i_h},{boxes[i][2]*i_w},{boxes[i][3]*i_h},{i_w},{i_h},{labels [i]},{scores[i]}\n"
                )

    # results_dict = {image_name : [ [xyxy, i_w, i_h, label, score], [..] ]} with de-normalized xywh

    return results_dict
