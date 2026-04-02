"""
conda 가상환경 qwen3-vl 환경에서 실행
conda activate qwen3-vl
Chat API - 이미지·텍스트를 옵션으로 받아 Qwen3-VL로 답변을 반환

  - 텍스트만: text
  - 이미지만: file (기본 질문 "이 이미지에 대해 자세히 설명해주세요." 사용)
  - 둘 다: file + text (이미지를 보며 text 질문에 답변)

  curl -X POST "http://host:port/chat" -F "text=이 이미지에 뭐가 있어?" -F "file=@img.jpg"
  curl -X POST "http://host:port/chat" -F "text=인공지능이란?"
  curl -X POST "http://host:port/chat" -F "file=@img.jpg"
"""

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from qa_generator import QAGenerator, Qwen3vlQaConfig

ALLOWED_IMAGE = {".jpg", ".jpeg", ".png"}
DEFAULT_IMAGE_PROMPT = "이 이미지에 대해 자세히 설명해주세요."

_chat_generator: QAGenerator | None = None


def _get_generator() -> QAGenerator:
    global _chat_generator
    if _chat_generator is None:
        model_name = os.environ.get("CHAT_MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct")
        max_new_tokens = int(os.environ.get("CHAT_MAX_NEW_TOKENS", "512"))
        config = Qwen3vlQaConfig(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
        )
        _chat_generator = QAGenerator(config)
    return _chat_generator


app = FastAPI(
    title="Chat API",
    description="이미지와 텍스트를 옵션으로 받아 Qwen3-VL 기반 답변을 반환합니다. 텍스트·이미지 중 최소 하나는 필요합니다.",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(
    text: str | None = Form(None, description="질문 또는 지시 (선택)"),
    file: UploadFile | None = File(None, description="이미지 파일 (선택)"),
):
    """
    텍스트(text)와 이미지(file) 중 최소 하나를 받아 답변을 반환합니다.
    - 텍스트만: text로 질문/지시에 답변
    - 이미지만: file만 보내면 기본 질문으로 이미지 설명
    - 둘 다: 이미지를 보며 text 질문에 답변
    """
    has_text = text is not None and (text or "").strip()
    has_file = file is not None and getattr(file, "filename", None) and (file.filename or "").strip()

    if not has_text and not has_file:
        raise HTTPException(
            400,
            "텍스트(text) 또는 이미지(file) 중 최소 하나를 보내주세요.",
        )

    prompt = (text or "").strip() if has_text else DEFAULT_IMAGE_PROMPT
    image_path: str | None = None
    tmpdir = None

    try:
        if has_file:
            suffix = Path(file.filename).suffix.lower()
            if suffix not in ALLOWED_IMAGE:
                raise HTTPException(
                    400,
                    f"이미지 허용 형식: {', '.join(ALLOWED_IMAGE)}",
                )
            tmpdir = tempfile.mkdtemp(prefix="chat_api_")
            image_path = os.path.join(tmpdir, f"img{suffix}")
            with open(image_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            if not os.path.isfile(image_path) or os.path.getsize(image_path) == 0:
                raise HTTPException(400, "이미지 파일이 비어 있습니다.")

        gen = _get_generator()
        answer = gen.get_model_response(prompt, image_path, None)

        if answer is None:
            raise HTTPException(500, "모델 답변 생성에 실패했습니다.")

        return {"answer": answer.strip()}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Chat 오류: {e}")
    finally:
        if tmpdir and os.path.isdir(tmpdir):
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
