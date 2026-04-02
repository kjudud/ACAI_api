"""
Detection 모듈 - DetectionProcessor를 사용하여 이미지 내 객체 검출
"""

import argparse
import os
from pathlib import Path

from ocr_processor import DetectionProcessor, DeepSeekOCRConfig


def run_detection(
    image_path: str | Path,
    output_dir: str,
    ref: str,
    config: DeepSeekOCRConfig | None = None,
) -> dict:
    """
    단일 이미지에 대해 DetectionProcessor로 검출 수행

    Args:
        image_path: 입력 이미지 파일 경로
        output_dir: 출력 디렉토리 경로
        ref: 검출 대상 (Locate <|ref|>ref<|/ref|>에 사용)
        config: DeepSeekOCRConfig (None이면 기본값 사용)

    Returns:
        detect_from_image 반환값 (또는 빈 dict)
    """
    path = Path(image_path) if isinstance(image_path, str) else image_path
    if not path.exists():
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {path}")

    processor = DetectionProcessor(config=config)
    return processor.detect_from_image(
        image_path=path,
        output_dir=output_dir,
        ref=ref,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="이미지 내 특정 대상을 검출합니다 (DetectionProcessor 사용)."
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="입력 이미지 파일 경로",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="detection_output",
        help="출력 디렉토리 (기본: detection_output)",
    )
    parser.add_argument(
        "-r",
        "--ref",
        type=str,
        required=True,
        help="검출 대상 (예: 'chart', 'table', 'signature')",
    )
    parser.add_argument(
        "--cuda-device",
        type=str,
        default="0",
        help="CUDA_VISIBLE_DEVICES (기본: 0)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = DeepSeekOCRConfig(cuda_visible_devices=args.cuda_device)
    result = run_detection(
        image_path=args.image_path,
        output_dir=args.output_dir,
        ref=args.ref,
        config=config,
    )

    if result:
        print("검출 완료:", result)
    else:
        print("검출 수행됨. (반환값 없음)")


if __name__ == "__main__":
    main()
