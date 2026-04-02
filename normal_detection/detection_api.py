"""
conda 가상환경 deepseek-ocr 환경에서 실행
conda activate deepseek-ocr
Detection API - 이미지 파일(file)을 받아 DetectionProcessor로 검출하고
result_with_boxes.jpg(인식 결과 사진)를 반환

  curl -X POST "http://host:port/detect" -F "file=@tank_ex2.jpg" -o result_with_boxes.jpg
  (멀티라인 시 \\ 뒤에 공백 없이 줄 끝에 두세요. 공백이 있으면 curl: (3) Malformed URL 오류가 납니다.)
"""

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from ocr_processor import DeepSeekOCRConfig, DetectionProcessor

ALLOWED_IMAGE = {".jpg", ".jpeg", ".png", ".webp"}
RESULT_IMAGE = "result_with_boxes.jpg"

app = FastAPI(
    title="Detection API",
    description="이미지(file)를 받아 DetectionProcessor로 검출하고 인식 결과 이미지(result_with_boxes.jpg)를 반환합니다.",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect")
def detect(
    file: UploadFile = File(..., description="이미지 파일"),
    ref: str | None = Form(None, description="검출 대상 (기본: tank, war plane, war ship, helicopter and soldier)"),
    cuda_device: str = Form("0", description="CUDA_VISIBLE_DEVICES"),
):
    """
    이미지를 받아 검출을 수행하고 바운딩 박스가 그려진 결과 이미지를 반환합니다.
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
    out_dir = None

    try:
        tmpdir = tempfile.mkdtemp(prefix="detection_api_")
        image_path = os.path.join(tmpdir, f"img{suffix}")
        with open(image_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        if not os.path.isfile(image_path) or os.path.getsize(image_path) == 0:
            raise HTTPException(400, "이미지 파일이 비어 있습니다.")

        out_dir = tempfile.mkdtemp(prefix="detection_out_")
        config = DeepSeekOCRConfig(cuda_visible_devices=cuda_device)
        processor = DetectionProcessor(config=config)
        result = processor.detect_from_image(
            image_path=image_path,
            output_dir=out_dir,
            ref=(ref.strip() if ref and ref.strip() else None),
        )

        # model.infer(save_results=True)가 output_path 디렉터리 안에 result_with_boxes.jpg 생성
        output_path = result.get("output_path", "")
        result_image = os.path.join(output_path, RESULT_IMAGE)

        if not os.path.isfile(result_image):
            raise HTTPException(500, f"검출 결과 이미지를 생성하지 못했습니다: {RESULT_IMAGE}")

        with open(result_image, "rb") as f:
            content = f.read()
        return Response(
            content=content,
            media_type="image/jpeg",
            headers={"Content-Disposition": f'attachment; filename="{RESULT_IMAGE}"'},
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Detection 오류: {e}")
    finally:
        for d in (tmpdir, out_dir):
            if d and os.path.isdir(d):
                try:
                    shutil.rmtree(d, ignore_errors=True)
                except Exception:
                    pass


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 20900))
    uvicorn.run(app, host="0.0.0.0", port=port)
