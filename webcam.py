import argparse
import os
import queue
import sys
import threading
import tkinter as tk
import traceback
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import get_real_score, parse_model_name


class WebcamApp:
    def __init__(self, camera_index, model_dir, device_id, real_threshold):
        self.camera_index = camera_index
        self.real_threshold = real_threshold
        self.model_paths = [
            os.path.join(model_dir, name)
            for name in sorted(os.listdir(model_dir))
            if name.endswith(".pth")
        ]
        if not self.model_paths:
            raise ValueError("No .pth models found in {}".format(model_dir))

        self.predictor = AntiSpoofPredict(device_id)
        self.cropper = CropImage()
        self.capture = None
        self.writer = None
        self.recording_path = None
        self.paused = False
        self.inference_busy = False
        self.inference_disabled = False
        self.results = queue.Queue()
        self.last_frame = None
        self.last_bbox = None
        self.last_label = None
        self.last_confidence = 0.0
        self.closed = False
        self.read_failures = 0
        self.camera_error_shown = False

        self.root = tk.Tk()
        self.root.title("Silent Face Anti-Spoofing - Webcam")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<KeyPress>", self._on_key)

        self.image_label = tk.Label(
            self.root,
            text="Waiting for camera frame...",
            width=80,
            height=24,
            bg="#202020",
            fg="white",
        )
        self.image_label.pack(padx=10, pady=10)
        self.status = tk.StringVar(value="Starting camera...")
        ttk.Label(self.root, textvariable=self.status).pack(pady=(0, 8))
        self.model_status = tk.StringVar(value="Waiting for first inference...")
        ttk.Label(self.root, textvariable=self.model_status).pack(pady=(0, 8))

        controls = ttk.Frame(self.root)
        controls.pack(pady=(0, 10))
        self.pause_button = ttk.Button(
            controls, text="Pause", command=self.toggle_pause
        )
        self.pause_button.pack(side=tk.LEFT, padx=4)
        self.record_button = ttk.Button(
            controls, text="Record (r)", command=self.toggle_recording
        )
        self.record_button.pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Quit (q)", command=self.close).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Label(
            self.root,
            text=(
                "Real threshold: {:.0%}   Space: pause/resume   "
                "r: start/stop recording   q or Esc: quit"
            ).format(self.real_threshold),
        ).pack(pady=(0, 10))

        if not self._open_camera():
            self.root.destroy()
            raise RuntimeError(
                "Could not open camera {}. Check camera permissions.".format(
                    camera_index
                )
            )

        self.root.after(20, self._update)

    def _open_camera(self):
        backends = []
        if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
            backends.append(("AVFoundation", cv2.CAP_AVFOUNDATION))
        backends.append(("default", cv2.CAP_ANY))

        for backend_name, backend in backends:
            capture = cv2.VideoCapture(self.camera_index, backend)
            if capture.isOpened():
                self.capture = capture
                self.camera_backend = backend_name
                print(
                    "Opened camera {} using {} backend".format(
                        self.camera_index, backend_name
                    )
                )
                return True
            capture.release()

        self.capture = None
        return False

    def _show_camera_error(self):
        if self.camera_error_shown:
            return
        self.camera_error_shown = True
        guidance = (
            "The camera opened, but OpenCV is not receiving frames.\n\n"
            "On macOS, allow camera access for the app running this script "
            "(Terminal or VS Code) in System Settings > Privacy & Security > "
            "Camera, then restart this script. Also close other apps using "
            "the camera. You can try another camera with --camera 1."
        )
        self.status.set("Camera opened but no frames received. See the error dialog.")
        self.image_label.configure(
            image="",
            text="No camera frames received\nCheck camera permissions and terminal output",
        )
        print(
            "Camera {} opened using {} backend, but read() returned no frames. "
            "Check camera permissions and whether another app is using it.".format(
                self.camera_index, self.camera_backend
            ),
            file=sys.stderr,
        )
        messagebox.showerror("No camera frames", guidance, parent=self.root)

    def _on_key(self, event):
        if event.keysym in ("Escape", "q", "Q"):
            self.close()
        elif event.keysym == "space":
            self.toggle_pause()
        elif event.char in ("r", "R"):
            self.toggle_recording()

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.configure(text="Resume" if self.paused else "Pause")
        self.status.set("Paused" if self.paused else "Live")

    def toggle_recording(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            saved_path = self.recording_path
            self.recording_path = None
            self.record_button.configure(text="Record (r)")
            self.status.set("Recording saved: {}".format(saved_path))
            return

        was_paused = self.paused
        if self.capture is not None:
            self.capture.release()
            self.capture = None

        output_path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save webcam recording",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4"), ("All files", "*")],
        )
        camera_opened = self._open_camera()
        if not camera_opened:
            messagebox.showerror(
                "Camera unavailable",
                "Could not restart camera {} after the save dialog.".format(
                    self.camera_index
                ),
                parent=self.root,
            )
            self.close()
            return

        self.paused = was_paused
        self.pause_button.configure(text="Resume" if self.paused else "Pause")
        if not output_path:
            self.status.set("Paused" if self.paused else "Live")
            return

        if self.last_frame is None:
            messagebox.showerror(
                "No camera frame",
                "Wait for the camera preview before starting a recording.",
                parent=self.root,
            )
            self.status.set("Paused" if self.paused else "Live")
            return

        height, width = self.last_frame.shape[:2]
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 20.0
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            writer.release()
            messagebox.showerror(
                "Could not save recording",
                "Could not create video file:\n{}".format(output_path),
                parent=self.root,
            )
            self.status.set("Paused" if self.paused else "Live")
            return

        self.writer = writer
        self.recording_path = output_path
        self.record_button.configure(text="Stop recording (r)")
        self.status.set("Recording: {}".format(output_path))

    def _run_inference(self, frame):
        try:
            bbox = self.predictor.get_bbox(frame)
            if bbox is None:
                self.results.put((None, None, 0.0, None))
                return

            prediction = np.zeros((1, 3))
            for model_path in self.model_paths:
                h_input, w_input, _, scale = parse_model_name(
                    os.path.basename(model_path)
                )
                crop = self.cropper.crop(
                    org_img=frame,
                    bbox=bbox,
                    scale=scale,
                    out_w=w_input,
                    out_h=h_input,
                    crop=scale is not None,
                )
                prediction += self.predictor.predict(crop, model_path)

            confidence = get_real_score(prediction, len(self.model_paths))
            label = int(confidence >= self.real_threshold)
            self.results.put((bbox, label, confidence, None))
        except Exception as error:
            self.results.put((None, None, 0.0, error))

    def _collect_results(self):
        try:
            bbox, label, confidence, error = self.results.get_nowait()
        except queue.Empty:
            return

        self.inference_busy = False
        if error is not None:
            self.inference_disabled = True
            self.model_status.set("Inference failed: {}".format(error))
            messagebox.showerror("Inference failed", str(error), parent=self.root)
            return

        self.last_bbox = bbox
        self.last_label = label
        self.last_confidence = confidence
        if bbox is None:
            self.model_status.set("No face detected")
        elif label == 1:
            self.model_status.set(
                "REAL — class-1 score {:.1%} (threshold {:.0%})".format(
                    confidence, self.real_threshold
                )
            )
        else:
            self.model_status.set(
                "NOT VERIFIED — class-1 score {:.1%} (threshold {:.0%})".format(
                    confidence, self.real_threshold
                )
            )

    def _annotated_frame(self, frame):
        if self.last_bbox is None:
            text = "No face detected" if self.last_label is None else "Detecting face..."
            cv2.putText(
                frame,
                text,
                (20, 35),
                cv2.FONT_HERSHEY_COMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            return frame

        if self.last_label is None:
            text = "Analyzing..."
            color = (0, 255, 255)
        elif self.last_label == 1:
            text = "Real Face: {:.2f}".format(self.last_confidence)
            color = (255, 0, 0)
        else:
            text = "Not verified (treated as fake): {:.2f}".format(
                self.last_confidence
            )
            color = (0, 0, 255)

        x, y, width, height = self.last_bbox
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
        cv2.putText(
            frame,
            text,
            (x, max(y - 5, 0)),
            cv2.FONT_HERSHEY_COMPLEX,
            0.6,
            color,
        )
        return frame

    def _show_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb_frame))
        self.image_label.configure(image=photo, text="")
        self.image_label.image = photo

    def _update(self):
        if self.closed:
            return

        try:
            self._collect_results()
            if not self.paused and self.capture is not None:
                ok, frame = self.capture.read()
                if ok and frame is not None and frame.size:
                    self.read_failures = 0
                    self.last_frame = frame.copy()
                    if not self.inference_busy and not self.inference_disabled:
                        self.inference_busy = True
                        threading.Thread(
                            target=self._run_inference,
                            args=(frame.copy(),),
                            daemon=True,
                        ).start()

                    display_frame = self._annotated_frame(frame.copy())
                    if self.writer is not None:
                        self.writer.write(display_frame)
                    self._show_frame(display_frame)
                    if self.writer is None:
                        self.status.set("Live")
                else:
                    self.read_failures += 1
                    self.status.set(
                        "Waiting for camera frames ({})...".format(
                            self.read_failures
                        )
                    )
                    if self.read_failures == 25:
                        self._show_camera_error()
            elif self.paused and self.last_frame is not None:
                self._show_frame(self._annotated_frame(self.last_frame.copy()))
        except Exception as error:
            traceback.print_exc()
            self.status.set("Display/capture error: {}".format(error))
            self.image_label.configure(
                image="",
                text="Display/capture error:\n{}".format(error),
            )
        finally:
            if not self.closed:
                self.root.after(20, self._update)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Live anti-spoofing webcam preview and optional recording"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0)",
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
        "--real_threshold",
        type=float,
        default=0.96,
        help=(
            "Minimum averaged class-1 score required to show Real "
            "(default: 0.96; score is not calibrated)"
        ),
    )
    args = parser.parse_args()
    if not 0 <= args.real_threshold <= 1:
        parser.error("--real_threshold must be between 0 and 1")

    try:
        WebcamApp(
            args.camera, args.model_dir, args.device_id, args.real_threshold
        ).run()
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, "Error: {}\n".format(error))
