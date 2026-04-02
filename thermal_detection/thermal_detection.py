"""
열화상(thermal) 이미지 사람 검출 - YOLOv11n 기반.
CLI로 실행하거나 thermal_detection_api에서 run_thermal_detection()을 호출해 사용.
"""
from pathlib import Path
from typing import TypedDict

from ultralytics import YOLO

# 로컬 가중치 경로 (원하는 경로로 수정)
WEIGHTS_PATH = Path(__file__).resolve().parent / "thermal_detection_weights" / "best.pt"

_model: YOLO | None = None


class BoundingBox(TypedDict):
    """바운딩 박스 좌표 (xyxy) 및 메타정보."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int


class ThermalDetectionResult(TypedDict):
    """run_thermal_detection 반환값."""

    result_image_path: str
    boxes: list[BoundingBox]


def get_model() -> YOLO:
    """모델 싱글톤 (API에서 재사용)."""
    global _model
    if _model is None:
        _model = YOLO(str(WEIGHTS_PATH))
    return _model


def _results_to_boxes(results) -> list[BoundingBox]:
    """Results 객체에서 바운딩 박스 리스트 추출 (xyxy, confidence, class_id)."""
    boxes_out: list[BoundingBox] = []
    if not results or not hasattr(results[0], "boxes") or results[0].boxes is None:
        return boxes_out
    for box in results[0].boxes:
        xyxy = box.xyxy[0]
        x1, y1, x2, y2 = xyxy.tolist()
        conf = float(box.conf.item()) if box.conf.numel() else 0.0
        cls_id = int(box.cls.item()) if box.cls.numel() else 0
        boxes_out.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": conf,
                "class_id": cls_id,
            }
        )
    return boxes_out


def run_thermal_detection(image_path: str | Path, output_dir: str | Path) -> ThermalDetectionResult:
    """
    열화상 이미지에서 사람을 검출하고 결과 이미지 경로와 바운딩 박스 좌표를 반환합니다.

    Args:
        image_path: 입력 이미지 파일 경로
        output_dir: 결과 이미지를 저장할 디렉터리

    Returns:
        {
            "result_image_path": 검출 결과 이미지의 절대 경로,
            "boxes": [{"x1", "y1", "x2", "y2", "confidence", "class_id"}, ...]
        }

    Raises:
        FileNotFoundError: 이미지가 없을 때
        RuntimeError: 결과 이미지를 찾지 못했을 때
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {path}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = get_model()
    results = model.predict(
        str(path),
        save=True,
        project=str(out),
        name=".",
        exist_ok=True,
    )

    if not results:
        raise RuntimeError("검출 결과가 없습니다.")

    boxes = _results_to_boxes(results)
    print(boxes)
    save_dir = Path(results[0].save_dir)
    result_image = save_dir / path.name
    if not result_image.exists():
        candidates = list(save_dir.glob("*.*"))
        if not candidates:
            raise RuntimeError(f"결과 이미지를 찾을 수 없습니다: {save_dir}")
        result_image = candidates[0]

    return {
        "result_image_path": str(result_image.resolve()),
        "boxes": boxes,
    }


if __name__ == "__main__":
    import json
    import sys

    default_image = Path(__file__).resolve().parent / "data" / "thermal_ex1.jpg"
    image = sys.argv[1] if len(sys.argv) > 1 else str(default_image)
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "runs/thermal_detect"

    result = run_thermal_detection(image, out_dir)
    print("저장 경로:", result["result_image_path"])
    print("바운딩 박스 개수:", len(result["boxes"]))
    if result["boxes"]:
        print("좌표 (xyxy, confidence, class_id):")
        print(json.dumps(result["boxes"], indent=2, ensure_ascii=False))