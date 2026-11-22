# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import os
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import get_real_score, parse_model_name
warnings.filterwarnings('ignore')


SAMPLE_IMAGE_PATH = "./images/sample/"


# 因为安卓端APK获取的视频流宽高比为3:4,为了与之一致，所以将宽高比限制为3:4
def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True


def test(image_name, model_dir, device_id, real_threshold):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    image = cv2.imread(SAMPLE_IMAGE_PATH + image_name)
    result = check_image(image)
    if result is False:
        return
    image_bbox = model_test.get_bbox(image)
    if image_bbox is None:
        print("No face detected in '{}'.".format(image_name))
        return
    prediction = np.zeros((1, 3))
    test_speed = 0
    # sum the prediction from single model's result
    model_names = [
        name for name in sorted(os.listdir(model_dir)) if name.endswith(".pth")
    ]
    if not model_names:
        raise ValueError("No .pth models found in {}".format(model_dir))
    for model_name in model_names:
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time()-start

    # draw result of prediction
    value = get_real_score(prediction, len(model_names))
    if value >= real_threshold:
        print(
            "Image '{}' is Real Face. Class-1 score: {:.2f}.".format(
                image_name, value
            )
        )
        result_text = "RealFace Score: {:.2f}".format(value)
        color = (255, 0, 0)
    else:
        print(
            "Image '{}' is NOT VERIFIED (treated as fake). "
            "Class-1 score: {:.2f}.".format(image_name, value)
        )
        result_text = "Not verified (fake): {:.2f}".format(value)
        color = (0, 0, 255)
    print("Prediction cost {:.2f} s".format(test_speed))
    cv2.rectangle(
        image,
        (image_bbox[0], image_bbox[1]),
        (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
        color, 2)
    cv2.putText(
        image,
        result_text,
        (image_bbox[0], image_bbox[1] - 5),
        cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)

    format_ = os.path.splitext(image_name)[-1]
    result_image_name = image_name.replace(format_, "_result" + format_)
    cv2.imwrite(SAMPLE_IMAGE_PATH + result_image_name, image)


if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to test")
    parser.add_argument(
        "--image_name",
        type=str,
        default="image_F1.jpg",
        help="image used to test")
    parser.add_argument(
        "--real_threshold",
        type=float,
        default=0.92,
        help="Minimum averaged class-1 score required to show Real (default: 0.92)",
    )
    args = parser.parse_args()
    if not 0 <= args.real_threshold <= 1:
        parser.error("--real_threshold must be between 0 and 1")
    test(args.image_name, args.model_dir, args.device_id, args.real_threshold)
