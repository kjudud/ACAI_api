"""
conda 가상환경 thermal-detection 환경에서 실행
conda activate thermal-detection
Thermal Detection API - 열화상 이미지 파일을 받아 YOLO로 사람 검출 후
결과 이미지(body) + bbox 좌표(X-Detection-Results 헤더) 한 번에 반환

  curl -X POST "http://host:port/thermal/detect" -F "file=@thermal.jpg" -o result.jpg -D -
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from thermal_detection import run_thermal_detection

ALLOWED_IMAGE = {".jpg", ".jpeg", ".png", ".webp"}

app = FastAPI(
    title="Thermal Detection API",
    description="열화상 이미지(file)를 받아 YOLO로 사람을 검출하고 인식 결과 이미지를 반환합니다.",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/thermal/detect")
def thermal_detect(
    file: UploadFile = File(..., description="열화상 이미지 파일"),
):
    """
    열화상 이미지를 받아 사람 검출을 수행합니다.
    - Body: 바운딩 박스가 그려진 결과 이미지 바이너리
    - Header X-Detection-Results: bbox 배열 JSON (각 항목: x1, y1, x2, y2, confidence, class_id)
    """
    if not file or not file.filename or not (file.filename or "").strip():
        raise HTTPException(400, "이미지 파일(file)을 보내주세요.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_IMAGE:
        raise HTTPException(
            400,
            f"이미지 허용 형식: {', '.join(ALLOWED_IMAGE)}",
        )

    tmpdir = None
    try:
        tmpdir = tempfile.mkdtemp(prefix="thermal_api_")
        image_path = os.path.join(tmpdir, f"img{suffix}")
        with open(image_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        if not os.path.isfile(image_path) or os.path.getsize(image_path) == 0:
            raise HTTPException(400, "이미지 파일이 비어 있습니다.")

        out_dir = os.path.join(tmpdir, "out")
        detection_result = run_thermal_detection(image_path, out_dir)
        result_path = detection_result["result_image_path"]

        if not os.path.isfile(result_path):
            raise HTTPException(500, "검출 결과 이미지를 생성하지 못했습니다.")

        with open(result_path, "rb") as f:
            content = f.read()

        result_filename = Path(result_path).name
        boxes = detection_result.get("boxes", [])

        ext = Path(result_path).suffix.lower()
        media_type = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
        headers = {
            "X-Detection-Results": json.dumps(boxes, ensure_ascii=False),
        }

        return Response(
            content=content,
            media_type=media_type,
            headers=headers,
        )

    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Thermal detection 오류: {e}")
    finally:
        if tmpdir and os.path.isdir(tmpdir):
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 20901))
    uvicorn.run(app, host="0.0.0.0", port=port)
