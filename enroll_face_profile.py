#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from blur_not_me import (
    SFACE_URL,
    YUNET_URL,
    create_models,
    ensure_file,
    load_reference_features,
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Create a reusable face profile from one or more photos."
    )
    parser.add_argument(
        "--images",
        required=True,
        nargs="+",
        type=Path,
        help="Reference photos for the same person.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output .npz profile path.",
    )
    parser.add_argument(
        "--name",
        default="main-face",
        help="Short profile name stored in metadata.",
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
        help="DNN backend for feature extraction.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Downscale factor used while detecting faces in the reference images.",
    )
    parser.add_argument(
        "--detect-threshold",
        type=float,
        default=0.65,
        help="YuNet detector confidence threshold.",
    )
    args = parser.parse_args()

    if not 0 < args.scale <= 1:
        parser.error("--scale must be in the range (0, 1].")

    return args


def main() -> int:
    args = parse_args()

    detector_model = ensure_file(
        args.models_dir / "face_detection_yunet_2023mar.onnx",
        YUNET_URL,
    )
    recognizer_model = ensure_file(
        args.models_dir / "face_recognition_sface_2021dec.onnx",
        SFACE_URL,
    )
    detector, recognizer, backend_name = create_models(
        detector_model=detector_model,
        recognizer_model=recognizer_model,
        frame_size=(320, 320),
        detect_threshold=args.detect_threshold,
        backend_choice=args.backend,
    )
    features = load_reference_features(args.images, detector, recognizer, args.scale)
    embeddings = np.stack(features).astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        embeddings=embeddings,
        profile_name=np.array(args.name),
        created_at=np.array(datetime.now(timezone.utc).isoformat()),
        image_count=np.array(len(args.images), dtype=np.int32),
    )

    print(
        f"[done] wrote {args.output} | embeddings={embeddings.shape[0]} "
        f"| backend={backend_name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
