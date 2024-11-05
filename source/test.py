# CREATE MODELS AND PREDICT
"""
Run this file by :
python test.py --model_weights path1_to_your_model1.pt path2_to_your_model2.pt --test_path data/small_val/images

"""
import argparse
from ultralytics import YOLO
import os
import time
from tqdm import tqdm

from utils.wbf import *
from utils.visualize import *
from utils.minority_optimizer import *

names = [
    "motorbike",  # 0
    "DHelmet",  # 1
    "DNoHelmet",  # 2
    "P1Helmet",  # 3
    "P1NoHelmet",  # 4
    "P2Helmet",  # 5
    "P2NoHelmet",  # 6
    "P0Helmet",  # 7
    "P0NoHelmet",  # 8
]  # Danh sách tên class


# create model lists and namesa
def create_models(model_weights_list):
    """
    - Create a list of models used to predict
    - Params:
        model_weights_list: list of paths to weights files
    """
    models_list = []
    model_names_list = []
    for model_weights in model_weights_list:
        model = YOLO(model_weights)
        model_name = os.path.basename(model_weights)

        models_list.append(model)
        model_names_list.append(model_name)

    return models_list, model_names_list


def get_models_predictions(model_weights_list, test_path, single_image=False):
    """
    Get predictions of models for all images in test_path or just single image

    Params:
    - model_weights_list: list of paths to model weights
    - test_path: path to test_folder or single image
    - single_image: predict for single image or not, (default = False)
    """
    # check valid path
    for path in model_weights_list:
        if not os.path.exists(path):
            print(f"\nModel with path {path} does not exist!")
            return None
    if not os.path.exists(test_path):
        print("Test path does not exist!")
        return None

    # create models list and get model names
    models_list, model_names_list = create_models(model_weights_list)

    # store models predictions
    predictions = {}

    for idx, model in enumerate(models_list):
        predictions[model_names_list[idx]] = {}
        model.overrides["verbose"] = False  # make terminal shorter

        # Predict for every image in test_path
        if not single_image:
            print(f"Processing with model: {model_names_list[idx]} ")

            # start counting time
            start_time = time.time()

            for image_name in tqdm(
                os.listdir(test_path), desc=f"Processing {model_names_list[idx]}"
            ):
                image_path = os.path.join(test_path, image_name)
                results = detect_image(image_path, model)

                predictions[model_names_list[idx]][image_name] = results

            # Print process time for current model
            elapsed_time = time.time() - start_time
            print(f"Completed {model_names_list[idx]} in {elapsed_time:.2f} seconds")

        # Predict for single image
        else:
            results = detect_image(test_path, model)
            predictions[model_names_list[idx]][os.path.basename(test_path)] = results

    # predictions[model_name][image_name] = [ [x1,y1,x2,y2,img_w, img_h, label, score], [..] ] non-normalized
    return predictions


def run_test(model_weights_list, test_path, plot=True):
    predictions = get_models_predictions(
        model_weights_list, test_path, single_image=False
    )

    models_names_list = [os.path.basename(path) for path in model_weights_list]

    # fuse results
    results = fuse(
        models_names_list,
        test_path,
        predictions,
        single_image=False,
        iou_thr=0.5,
        skip_box_thr=0.05,
    )

    # apply minority optimizer
    p = 0.001
    results = minority_optimizer_func(results, p)

    # visualize
    if plot:
        images = os.listdir(test_path)
        sample_image = images[10]
        visualize(os.path.join(test_path, sample_image), results[sample_image])

    return results


def run_test_on_single_image(model_weights_list, image_path, plot=True):
    predictions = get_models_predictions(
        model_weights_list, image_path, single_image=True
    )
    models_names_list = [os.path.basename(path) for path in model_weights_list]

    # fuse results
    results = fuse(
        models_names_list,
        image_path,
        predictions,
        single_image=True,
        iou_thr=0.5,
        skip_box_thr=0.001,
    )

    # apply minority optimizer (p: conf thres for rare classes, common_p: conf thres for common classes)
    """this step is to filter out the predictions of common classes with low confidence and preserve the predictions of rare classes with higher confidence than minority_score"""
    results = minority_optimizer_func(results, p=0.09, common_p=0.3)

    for image_name, preds in results.items():
        print(f"Image: {image_name}")
        for pred in preds:
            parts = pred.strip().split(",")
            print(
                f"\tLabel: {parts[6]}: {names[int(float(parts[6]))]}, Score: {parts[7]}"
            )

    # visualize
    if plot:
        visualize(image_path, results[os.path.basename(image_path)])

    return results


if __name__ == "__main__":

    # PARSE ARGUMENTS
    parser = argparse.ArgumentParser(description="YOLO Model Predictions")

    parser.add_argument(
        "--model_weights",
        nargs="+",
        default=["weights/v10s/v10s_best5.pt", "weights/v11s/v11s_best5.pt"],
        help="List of paths to model weight files",
    )
    parser.add_argument(
        "--test_path",
        default="../data/small_val/images",
        help="Path to test images folder or a single image",
    )
    parser.add_argument(
        "--single_image",
        type=bool,
        default=False,
        help="Flag to run prediction on a single image",
    )
    parser.add_argument(
        "--image_index",
        default=10,
        type=int,
        help="Flag to choose an image from test_path to run prediction on",
    )

    args = parser.parse_args()

    # CHECK INPUTS
    if not os.path.exists(args.test_path):
        print("Test path does not exist!")
        print(args.test_path)
        exit()
    if not all([os.path.exists(path) for path in args.model_weights]):
        print("Model weights do not exist!")
        print(args.model_weights)
        exit()

    # RUN TEST
    if not args.single_image:
        run_test(args.model_weights, args.test_path)

    # RUN TEST ON SINGLE IMAGE
    if args.single_image:
        sample_image = os.path.join(
            args.test_path, os.listdir(args.test_path)[args.image_index]
        )
        run_test_on_single_image(args.model_weights, sample_image)

    # sample_images = os.listdir(test_path)
    # for sample_image in sample_images[:10]:
    #     run_test_on_single_image(model_weights_list, os.path.join(test_path, sample_image))


"""
Chỉnh sửa lại cách chạy file test.py: 
- có thể truyền p và common_p vào hàm minority_optimizer_func
- Bỏ argument model_weights 
- Chỉnh lại tham số cho hàm count_samples_per_class_on_train: nên count_samples_per_class trong hàm run_test, rồi truyền vào các hàm tính minority scores
- Cân nhắc tính minority score cho từng class riêng biệt
- 
"""
