import argparse
import os
import time

import cv2
import numpy as np

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import get_real_score, parse_model_name


class ORBFrameSampler:
    def __init__(self, similarity_threshold):
        self.orb = cv2.ORB_create(nfeatures=400)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.similarity_threshold = similarity_threshold
        self.reference_descriptors = None

    def is_redundant(self, frame):
        small_frame = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        _, descriptors = self.orb.detectAndCompute(gray, None)

        if descriptors is None or self.reference_descriptors is None:
            self.reference_descriptors = descriptors
            return False

        matches = self.matcher.match(self.reference_descriptors, descriptors)
        denominator = min(
            len(self.reference_descriptors),
            len(descriptors),
        )
        similarity = len(matches) / denominator if denominator else 0
        if similarity >= self.similarity_threshold:
            return True

        self.reference_descriptors = descriptors
        return False


def _annotate(frame, bbox, label, real_score):
    if bbox is None:
        cv2.putText(
            frame,
            "No face detected",
            (20, 35),
            cv2.FONT_HERSHEY_COMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        return frame

    if label is not None:
        if label == 1:
            result_text = "Real Face: {:.2f}".format(real_score)
            color = (255, 0, 0)
        else:
            result_text = "Not verified (treated as fake): {:.2f}".format(
                real_score
            )
            color = (0, 0, 255)
    else:
        result_text = "Analyzing..."
        color = (0, 255, 255)

    x, y, box_width, box_height = bbox
    cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), color, 2)
    cv2.putText(
        frame,
        result_text,
        (x, max(y - 5, 0)),
        cv2.FONT_HERSHEY_COMPLEX,
        0.6,
        color,
    )
    return frame


def process_video(
    video_path, output_path, model_dir, device_id, orb_threshold, real_threshold
):
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

    sampler = ORBFrameSampler(orb_threshold)
    frame_count = 0
    inference_count = 0
    bbox = None
    label = None
    real_score = 0.0
    start_time = time.perf_counter()

    try:
        predictor = AntiSpoofPredict(device_id)
        cropper = CropImage()
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if not sampler.is_redundant(frame):
                bbox = predictor.get_bbox(frame)
                label = None
                real_score = 0.0
                if bbox is not None:
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
                    real_score = get_real_score(prediction, len(model_paths))
                    label = int(real_score >= real_threshold)
                inference_count += 1

            writer.write(_annotate(frame, bbox, label, real_score))
            frame_count += 1
            if frame_count % 30 == 0:
                print(
                    "Processed {} frames; ran inference {} times".format(
                        frame_count, inference_count
                    )
                )
    finally:
        capture.release()
        writer.release()

    elapsed = time.perf_counter() - start_time
    print(
        "Processed {} frames; ran inference {} times in {:.1f}s. Output: {}".format(
            frame_count, inference_count, elapsed, output_path
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run anti-spoofing on a video and skip similar-frame inference with ORB"
    )
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
    parser.add_argument(
        "--orb_threshold",
        type=float,
        default=0.8,
        help="ORB descriptor similarity threshold for skipping inference (0-1)",
    )
    parser.add_argument(
        "--real_threshold",
        type=float,
        default=0.92,
        help="Minimum averaged class-1 score required to show Real (default: 0.92)",
    )
    args = parser.parse_args()
    if not 0 <= args.orb_threshold <= 1:
        parser.error("--orb_threshold must be between 0 and 1")
    if not 0 <= args.real_threshold <= 1:
        parser.error("--real_threshold must be between 0 and 1")

    output_path = args.output
    if output_path is None:
        video_stem = os.path.splitext(args.video)[0]
        output_path = video_stem + "_result.mp4"
    process_video(
        args.video,
        output_path,
        args.model_dir,
        args.device_id,
        args.orb_threshold,
        args.real_threshold,
    )
