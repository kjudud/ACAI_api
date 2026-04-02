"""
Chat - Qwen3-VL로 request(텍스트)와 image(경로)를 옵션으로 받아 응답 문자열 반환
"""

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_IMAGE_PROMPT = "이 이미지를 자세히 설명해주세요."

_model = None
_processor = None


def _get_model():
    global _model, _processor
    if _model is None:
        _model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_NAME, dtype="auto", device_map="auto"
        )
        _processor = AutoProcessor.from_pretrained(MODEL_NAME)
    return _model, _processor


def chat(
    request: str | None = None,
    image_path: str | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> str:
    """
    request(텍스트)와 image_path(이미지 파일 경로)를 옵션으로 받아 모델 응답을 반환합니다.
    둘 다 없으면 ValueError. image만 있으면 기본 프롬프트로 설명 요청.

    Args:
        request: 질문 또는 지시 (None 가능)
        image_path: 이미지 파일 경로 (None 가능)
        max_new_tokens: 최대 생성 토큰 수

    Returns:
        생성된 응답 문자열
    """
    if request is None and not image_path:
        raise ValueError("request와 image_path 중 최소 하나는 있어야 합니다.")

    text = (request or "").strip() or DEFAULT_IMAGE_PROMPT
    model, processor = _get_model()

    if image_path:
        content = [
            {"type": "image", "image": image_path},
            {"type": "text", "text": text},
        ]
    else:
        content = [{"type": "text", "text": text}]

    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return (output_text[0] or "").strip()
