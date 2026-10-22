#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Keep only the reference person unblurred in a video. "
            "Everyone else gets Gaussian blurred."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Input video path.")
    parser.add_argument("--output", required=True, type=Path, help="Output video path.")
    parser.add_argument(
        "--reference",
        nargs="+",
        type=Path,
        help="One or more reference photos of the person to keep sharp.",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        help="Saved face profile (.npz) created with enroll_face_profile.py.",
    )
    parser.add_argument(
        "--models-dir",
        default=script_dir / "models",
        type=Path,
        help="Where YuNet/SFace ONNX models are stored.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="DNN backend for YuNet/SFace. 'auto' prefers CUDA when available.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Downscale factor for face detection. Lower is faster, higher is more accurate.",
    )
    parser.add_argument(
        "--keep-threshold",
        type=float,
        default=0.38,
        help="Cosine similarity needed to keep a face unblurred.",
    )
    parser.add_argument(
        "--detect-threshold",
        type=float,
        default=0.65,
        help="YuNet detector confidence threshold.",
    )
    parser.add_argument(
        "--blur-padding",
        type=float,
        default=0.2,
        help="Extra blur padding around each rejected face box.",
    )
    parser.add_argument(
        "--blur-strength",
        type=float,
        default=0.35,
        help="Relative Gaussian blur kernel size.",
    )
    parser.add_argument(
        "--effect",
        choices=("mosaic", "blur"),
        default="mosaic",
        help="How to censor non-target faces.",
    )
    parser.add_argument(
        "--mosaic-scale",
        type=float,
        default=0.06,
        help="Relative downscale factor for mosaic pixelation. Lower is chunkier.",
    )
    parser.add_argument(
        "--encoder",
        choices=("auto", "h264_nvenc", "libx264"),
        default="auto",
        help="Video encoder for ffmpeg output.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame cap for quick testing.",
    )
    args = parser.parse_args()

    if not 0 < args.scale <= 1:
        parser.error("--scale must be in the range (0, 1].")
    if args.keep_threshold <= 0:
        parser.error("--keep-threshold must be positive.")
    if not args.reference and not args.profile:
        parser.error("provide at least one of --reference or --profile")

    return args


def ensure_file(path: Path, url: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path

    print(f"[setup] downloading {path.name}", file=sys.stderr)
    urllib.request.urlretrieve(url, path)
    return path


def resolve_backend(choice: str) -> tuple[int, int, str]:
    can_use_cuda = False
    if hasattr(cv2, "cuda"):
        try:
            can_use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
        except Exception:
            can_use_cuda = False

    if choice == "cpu":
        return cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU, "cpu"

    if choice == "cuda" and not can_use_cuda:
        raise RuntimeError("CUDA backend requested, but no CUDA-capable OpenCV device is available.")

    if choice == "cuda" or (choice == "auto" and can_use_cuda):
        return cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA_FP16, "cuda"

    return cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU, "cpu"


def create_models(
    detector_model: Path,
    recognizer_model: Path,
    frame_size: tuple[int, int],
    detect_threshold: float,
    backend_choice: str,
) -> tuple[cv2.FaceDetectorYN, cv2.FaceRecognizerSF, str]:
    backend_id, target_id, backend_name = resolve_backend(backend_choice)

    try:
        detector = cv2.FaceDetectorYN.create(
            str(detector_model),
            "",
            frame_size,
            detect_threshold,
            0.3,
            5000,
            backend_id,
            target_id,
        )
        recognizer = cv2.FaceRecognizerSF.create(
            str(recognizer_model),
            "",
            backend_id,
            target_id,
        )
        return detector, recognizer, backend_name
    except cv2.error as exc:
        if backend_name != "cuda":
            raise

        print(
            "[setup] CUDA OpenCV backend failed, falling back to CPU DNN backend.",
            file=sys.stderr,
        )
        detector = cv2.FaceDetectorYN.create(
            str(detector_model),
            "",
            frame_size,
            detect_threshold,
            0.3,
            5000,
            cv2.dnn.DNN_BACKEND_OPENCV,
            cv2.dnn.DNN_TARGET_CPU,
        )
        recognizer = cv2.FaceRecognizerSF.create(
            str(recognizer_model),
            "",
            cv2.dnn.DNN_BACKEND_OPENCV,
            cv2.dnn.DNN_TARGET_CPU,
        )
        return detector, recognizer, "cpu"


def detect_faces(detector: cv2.FaceDetectorYN, frame: np.ndarray, scale: float) -> list[np.ndarray]:
    if scale != 1.0:
        scaled = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    else:
        scaled = frame

    height, width = scaled.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(scaled)
    if faces is None or len(faces) == 0:
        return []

    detections = [face.copy() for face in faces]
    if scale != 1.0:
        inv = 1.0 / scale
        for face in detections:
            face[:14] *= inv
    return detections


def largest_face(faces: list[np.ndarray]) -> np.ndarray:
    return max(faces, key=lambda face: float(face[2] * face[3]))


def extract_feature(
    frame: np.ndarray,
    face: np.ndarray,
    recognizer: cv2.FaceRecognizerSF,
) -> np.ndarray:
    aligned = recognizer.alignCrop(frame, face)
    feature = recognizer.feature(aligned)
    return feature.reshape(-1).astype(np.float32)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0:
        return 0.0
    return float(np.dot(left, right) / denom)


def load_reference_features(
    image_paths: list[Path],
    detector: cv2.FaceDetectorYN,
    recognizer: cv2.FaceRecognizerSF,
    scale: float,
) -> list[np.ndarray]:
    features: list[np.ndarray] = []

    for image_path in image_paths:
        if not image_path.exists():
            raise FileNotFoundError(f"Reference image not found: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read reference image: {image_path}")

        faces = detect_faces(detector, image, scale)
        if not faces:
            raise RuntimeError(f"No face found in reference image: {image_path}")

        try:
            feature = extract_feature(image, largest_face(faces), recognizer)
        except cv2.error as exc:
            raise RuntimeError(f"Failed to extract face embedding from {image_path}") from exc

        features.append(feature)

    return features


def load_profile_features(profile_path: Path) -> list[np.ndarray]:
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")

    try:
        data = np.load(profile_path, allow_pickle=False)
    except Exception as exc:
        raise RuntimeError(f"Could not read profile: {profile_path}") from exc

    if "embeddings" not in data:
        raise RuntimeError(f"Profile is missing embeddings: {profile_path}")

    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise RuntimeError(f"Profile has no usable embeddings: {profile_path}")

    return [row.astype(np.float32, copy=False) for row in embeddings]


def best_similarity(feature: np.ndarray, references: list[np.ndarray]) -> float:
    return max(cosine_similarity(feature, ref) for ref in references)


def face_box(face: np.ndarray) -> tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in face[:4]]
    return x, y, x + w, y + h


def bbox_iou(left: np.ndarray, right: np.ndarray) -> float:
    lx1, ly1, lx2, ly2 = face_box(left)
    rx1, ry1, rx2, ry2 = face_box(right)
    ix1 = max(lx1, rx1)
    iy1 = max(ly1, ry1)
    ix2 = min(lx2, rx2)
    iy2 = min(ly2, ry2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    left_area = max(1.0, (lx2 - lx1) * (ly2 - ly1))
    right_area = max(1.0, (rx2 - rx1) * (ry2 - ry1))
    return inter / (left_area + right_area - inter)


def temporal_score(face: np.ndarray, previous_face: np.ndarray | None) -> float:
    if previous_face is None:
        return 0.0

    x, y, w, h = [float(v) for v in face[:4]]
    px, py, pw, ph = [float(v) for v in previous_face[:4]]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
    prev_center = np.array([px + pw * 0.5, py + ph * 0.5], dtype=np.float32)
    center_distance = float(np.linalg.norm(center - prev_center))
    max_distance = max(w, h, pw, ph) * 2.5 + 1.0
    proximity = max(0.0, 1.0 - center_distance / max_distance)
    size_ratio = min(w * h, pw * ph) / max(w * h, pw * ph, 1.0)
    return 0.5 * bbox_iou(face, previous_face) + 0.35 * proximity + 0.15 * size_ratio


def select_target_face(
    faces: list[np.ndarray],
    features: list[np.ndarray],
    references: list[np.ndarray],
    previous_face: np.ndarray | None,
    previous_feature: np.ndarray | None,
    keep_threshold: float,
) -> int | None:
    if not faces:
        return None

    best_idx: int | None = None
    best_score = -1.0
    best_ref = 0.0
    best_prev = 0.0
    best_motion = 0.0

    for idx, (face, feature) in enumerate(zip(faces, features)):
        ref_similarity = best_similarity(feature, references)
        prev_similarity = (
            cosine_similarity(feature, previous_feature) if previous_feature is not None else 0.0
        )
        motion = temporal_score(face, previous_face)
        if previous_face is None:
            score = ref_similarity
        else:
            score = ref_similarity * 0.6 + prev_similarity * 0.25 + motion * 0.15

        if score > best_score:
            best_idx = idx
            best_score = score
            best_ref = ref_similarity
            best_prev = prev_similarity
            best_motion = motion

    if best_idx is None:
        return None

    if best_ref >= keep_threshold:
        return best_idx
    if previous_face is None:
        return None
    if best_ref >= keep_threshold - 0.08 and best_motion >= 0.25:
        return best_idx
    if best_prev >= keep_threshold - 0.10 and best_motion >= 0.12:
        return best_idx
    if best_score >= keep_threshold - 0.06 and best_motion >= 0.35:
        return best_idx
    return None


def blur_face(frame: np.ndarray, face: np.ndarray, padding: float, strength: float) -> None:
    x, y, w, h = face[:4]
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    x1 = max(0, int(round(x - pad_x)))
    y1 = max(0, int(round(y - pad_y)))
    x2 = min(frame.shape[1], int(round(x + w + pad_x)))
    y2 = min(frame.shape[0], int(round(y + h + pad_y)))

    if x2 <= x1 or y2 <= y1:
        return

    roi = frame[y1:y2, x1:x2]
    min_dim = min(roi.shape[0], roi.shape[1])
    if min_dim < 3:
        return

    kernel = max(3, int(round(min_dim * strength)))
    if kernel % 2 == 0:
        kernel += 1
    if kernel > min_dim:
        kernel = min_dim if min_dim % 2 == 1 else min_dim - 1
    if kernel < 3:
        kernel = 3

    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kernel, kernel), 0)


def mosaic_face(frame: np.ndarray, face: np.ndarray, padding: float, mosaic_scale: float) -> None:
    x, y, w, h = face[:4]
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    x1 = max(0, int(round(x - pad_x)))
    y1 = max(0, int(round(y - pad_y)))
    x2 = min(frame.shape[1], int(round(x + w + pad_x)))
    y2 = min(frame.shape[0], int(round(y + h + pad_y)))

    if x2 <= x1 or y2 <= y1:
        return

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return

    down_w = max(1, int(round(roi.shape[1] * mosaic_scale)))
    down_h = max(1, int(round(roi.shape[0] * mosaic_scale)))
    reduced = cv2.resize(roi, (down_w, down_h), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(
        reduced,
        (roi.shape[1], roi.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    frame[y1:y2, x1:x2] = pixelated


def censor_face(
    frame: np.ndarray,
    face: np.ndarray,
    effect: str,
    padding: float,
    blur_strength: float,
    mosaic_scale: float,
) -> None:
    if effect == "blur":
        blur_face(frame, face, padding, blur_strength)
        return
    mosaic_face(frame, face, padding, mosaic_scale)


def pick_encoder(choice: str) -> str:
    if choice != "auto":
        return choice

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found in PATH.")

    probe = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=True,
    )
    if "h264_nvenc" in probe.stdout:
        return "h264_nvenc"
    return "libx264"


def start_ffmpeg_writer(
    input_path: Path,
    output_path: Path,
    frame_size: tuple[int, int],
    fps: float,
    encoder: str,
) -> subprocess.Popen[bytes]:
    width, height = frame_size
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        f"{fps:.06f}",
        "-i",
        "-",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        encoder,
    ]

    if encoder == "h264_nvenc":
        command.extend(["-preset", "p5", "-cq", "23", "-rc", "vbr", "-b:v", "0"])
    else:
        command.extend(["-preset", "medium", "-crf", "20"])

    command.extend(
        [
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-shortest",
            str(output_path),
        ]
    )

    return subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def print_progress(processed: int, total: int | None, started_at: float) -> None:
    elapsed = max(time.time() - started_at, 0.001)
    fps = processed / elapsed
    if total:
        pct = processed / total * 100
        text = f"\r[run] {processed}/{total} frames ({pct:5.1f}%) at {fps:5.1f} fps"
    else:
        text = f"\r[run] {processed} frames at {fps:5.1f} fps"
    print(text, end="", file=sys.stderr, flush=True)


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input video not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    detector_model = ensure_file(args.models_dir / "face_detection_yunet_2023mar.onnx", YUNET_URL)
    recognizer_model = ensure_file(
        args.models_dir / "face_recognition_sface_2021dec.onnx",
        SFACE_URL,
    )

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {args.input}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_frames = min(frame_count, args.max_frames) if args.max_frames and frame_count else args.max_frames or frame_count

    detector, recognizer, backend_name = create_models(
        detector_model=detector_model,
        recognizer_model=recognizer_model,
        frame_size=(max(1, int(round(width * args.scale))), max(1, int(round(height * args.scale)))),
        detect_threshold=args.detect_threshold,
        backend_choice=args.backend,
    )
    reference_features: list[np.ndarray] = []
    if args.profile:
        reference_features.extend(load_profile_features(args.profile))
    if args.reference:
        reference_features.extend(load_reference_features(args.reference, detector, recognizer, args.scale))
    if not reference_features:
        raise RuntimeError("No reference embeddings were loaded.")

    encoder = pick_encoder(args.encoder)
    print(f"[setup] backend={backend_name} encoder={encoder} fps={fps:.3f}", file=sys.stderr)
    writer = start_ffmpeg_writer(args.input, args.output, (width, height), fps, encoder)
    if writer.stdin is None or writer.stderr is None:
        raise RuntimeError("ffmpeg pipe setup failed.")

    processed = 0
    faces_seen = 0
    faces_blurred = 0
    started_at = time.time()
    previous_target_face: np.ndarray | None = None
    previous_target_feature: np.ndarray | None = None
    target_miss_streak = 0

    try:
        while True:
            if args.max_frames is not None and processed >= args.max_frames:
                break

            ok, frame = capture.read()
            if not ok:
                break

            faces = detect_faces(detector, frame, args.scale)
            faces_seen += len(faces)
            face_features: list[np.ndarray | None] = []

            for face in faces:
                try:
                    face_features.append(extract_feature(frame, face, recognizer))
                except cv2.error:
                    face_features.append(None)

            valid_faces: list[np.ndarray] = []
            valid_features: list[np.ndarray] = []
            valid_to_original: list[int] = []
            for idx, (face, feature) in enumerate(zip(faces, face_features)):
                if feature is None:
                    continue
                valid_faces.append(face)
                valid_features.append(feature)
                valid_to_original.append(idx)

            target_original_idx: int | None = None
            if valid_faces:
                selected_valid_idx = select_target_face(
                    valid_faces,
                    valid_features,
                    reference_features,
                    previous_target_face if target_miss_streak <= 12 else None,
                    previous_target_feature if target_miss_streak <= 12 else None,
                    args.keep_threshold,
                )
                if selected_valid_idx is not None:
                    target_original_idx = valid_to_original[selected_valid_idx]
                    previous_target_face = valid_faces[selected_valid_idx]
                    previous_target_feature = valid_features[selected_valid_idx]
                    target_miss_streak = 0
                else:
                    target_miss_streak += 1
            else:
                target_miss_streak += 1

            if target_miss_streak > 24:
                previous_target_face = None
                previous_target_feature = None

            for idx, face in enumerate(faces):
                if idx == target_original_idx:
                    continue
                censor_face(
                    frame,
                    face,
                    args.effect,
                    args.blur_padding,
                    args.blur_strength,
                    args.mosaic_scale,
                )
                faces_blurred += 1

            writer.stdin.write(frame.tobytes())
            processed += 1
            if processed == 1 or processed % 30 == 0:
                print_progress(processed, total_frames or None, started_at)
    finally:
        capture.release()
        try:
            writer.stdin.close()
        except Exception:
            pass

    ffmpeg_stderr = writer.stderr.read().decode("utf-8", errors="replace")
    exit_code = writer.wait()
    print(file=sys.stderr)

    if exit_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {exit_code}:\n{ffmpeg_stderr}")

    elapsed = max(time.time() - started_at, 0.001)
    print(
        (
            f"[done] wrote {args.output} | frames={processed} | "
            f"faces_seen={faces_seen} | faces_blurred={faces_blurred} | "
            f"elapsed={elapsed:.1f}s"
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[abort] interrupted by user", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
