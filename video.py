import argparse
import os
import time

import cv2
import numpy as np

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name


def process_video(video_path, output_path, model_dir, device_id):
    if os.path.abspath(video_path) == os.path.abspath(output_path):
        raise ValueError("Output video must be different from the input video")

    model_paths = [
        os.path.join(model_dir, name)
        for name in sorted(os.listdir(model_dir))
        if name.endswith(".pth")
    ]
    if not model_paths:
        raise ValueError("No .pth models found in {}".format(model_dir))

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError("Could not open video: {}".format(video_path))

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("Could not read video dimensions: {}".format(video_path))
    if fps <= 0:
        fps = 30.0

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError("Could not create output video: {}".format(output_path))

    frame_count = 0
    start_time = time.perf_counter()

    try:
        predictor = AntiSpoofPredict(device_id)
        cropper = CropImage()
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            bbox = predictor.get_bbox(frame)
            prediction = np.zeros((1, 3))
            for model_path in model_paths:
                h_input, w_input, _, scale = parse_model_name(
                    os.path.basename(model_path)
                )
                crop = cropper.crop(
                    org_img=frame,
                    bbox=bbox,
                    scale=scale,
                    out_w=w_input,
                    out_h=h_input,
                    crop=scale is not None,
                )
                prediction += predictor.predict(crop, model_path)

            label = int(np.argmax(prediction))
            confidence = prediction[0][label] / len(model_paths)
            if label == 1:
                result_text = "Real Face: {:.2f}".format(confidence)
                color = (255, 0, 0)
            else:
                result_text = "Fake Face: {:.2f}".format(confidence)
                color = (0, 0, 255)

            x, y, box_width, box_height = bbox
            cv2.rectangle(
                frame, (x, y), (x + box_width, y + box_height), color, 2
            )
            cv2.putText(
                frame,
                result_text,
                (x, max(y - 5, 0)),
                cv2.FONT_HERSHEY_COMPLEX,
                0.6,
                color,
            )
            writer.write(frame)
            frame_count += 1
            if frame_count % 30 == 0:
                print("Processed {} frames".format(frame_count))
    finally:
        capture.release()
        writer.release()

    elapsed = time.perf_counter() - start_time
    print(
        "Processed {} frames in {:.1f}s ({:.2f} FPS). Output: {}".format(
            frame_count, elapsed, frame_count / elapsed if elapsed else 0, output_path
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run anti-spoofing on a video")
    parser.add_argument("video", help="Path to the input video")
    parser.add_argument(
        "--output",
        help="Path for the annotated MP4 (defaults to <video>_result.mp4)",
    )
    parser.add_argument(
        "--model_dir",
        default="./resources/anti_spoof_models",
        help="Directory containing anti-spoofing .pth model files",
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="GPU device index (CPU is used when CUDA is unavailable)",
    )
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        video_stem = os.path.splitext(args.video)[0]
        output_path = video_stem + "_result.mp4"
    process_video(args.video, output_path, args.model_dir, args.device_id)
