# ACAI apis

전장 정보 micro model 예시 API 입니다. 일반 객체 검출(Normal detection), 열화상 카메라 객체 검출(Thermal detection), 전장 정보 수집 채팅(Chat)을 위한 REST API입니다.

- **Normal-Detection**: 이미지 내 지정 대상(tank, chart 등) 검출 및 바운딩 박스 결과 이미지 생성
- **Thermal-Detection**: 열화상 카메라 이미지 내 지정 사람 검출 및 바운딩 박스 결과 이미지 생성
- **Chat**: Qwen3-VL로 전장 정보에 대한 이미지·텍스트 질의 답변

---

## 요구 사항

- Python 3.10+
- CUDA (Detection/OCR, Chat/QA용)
- GPU 메모리: Detection(DeepSeek-OCR), Chat/QA(Qwen3-VL) 각각 권장

---

## 설치

```bash
git clone <repository-url>
cd ACAI_api
```

각 API는 **서로 다른 conda 환경**을 사용합니다. 아래 둘 중 하나로 맞추면 됩니다.

### Conda 환경 (`*.yml`)

저장소에 포함된 YAML로 한 번에 생성합니다. 이미 같은 이름의 환경이 있으면 `conda env remove -n <이름>` 후 다시 만들거나, 다른 이름으로 만들려면 `conda env create -f deepseek-ocr.yml -n my-deepseek`처럼 `-n`을 지정합니다.

```bash
conda env create -f deepseek-ocr.yml
conda env create -f qwen3-vl.yml
conda env create -f yolov11n-thermal.yml
```

| 파일 | 생성되는 환경 이름 | 용도 |
|------|-------------------|------|
| `deepseek-ocr.yml` | `deepseek-ocr` | Normal detection / OCR |
| `qwen3-vl.yml` | `qwen3-vl` | Chat |
| `yolov11n-thermal.yml` | `yolov11n-thermal` | Thermal detection |

YAML은 특정 OS·채널에서 export된 경우가 있어, 다른 머신에서 실패하면 `requirements-*.txt` 경로를 사용하세요.

### pip requirements (`requirements-*.txt`)

이미 Python/conda 베이스 환경을 쓰는 경우, 활성화한 뒤 해당 API용 목록만 설치합니다.

```bash
conda activate deepseek-ocr   # 또는 venv 등
pip install -r requirements-deepseek-ocr.txt

conda activate qwen3-vl
pip install -r requirements-qwen3-vl.txt

conda activate yolov11n-thermal
pip install -r requirements-yolov11n-thermal.txt
```

`pip freeze`로 뽑은 목록이라 패키지 수가 많을 수 있습니다. `flash-attn` 등 빌드가 필요한 패키지는 [공식 문서](https://github.com/Dao-AILab/flash-attention)에 맞게 별도 설치가 필요할 수 있습니다.

---

## 프로젝트 구조

| 경로 | 설명 |
|------|------|
| `normal_detection/ocr_processor.py` | DeepSeek-OCR 기반 OCR, Detection 처리 |
| `normal_detection/detection.py` | CLI: 이미지 검출 (`ref` 지정) |
| `normal_detection/detection_api.py` | REST: 이미지 업로드 → 검출 결과 이미지 반환 |
| `chat/qa_generator.py` | Qwen3-VL 기반 Q&A 쌍 생성 |
| `chat/chat.py` | Qwen3-VL 로컬/스크립트용 진입점 |
| `chat/chat_api.py` | REST: 텍스트·이미지 옵션 → 답변 JSON |
| `thermal_detection/thermal_detection.py` | 열화상 YOLO 검출 로직 |
| `thermal_detection/thermal_detection_api.py` | REST: 열화상 이미지 → 검출 결과 이미지 + 헤더 |

---

## 사용법

API는 **해당 패키지 디렉터리에서** 실행해야 로컬 import(`ocr_processor`, `qa_generator` 등)가 동작합니다.

### 1. Normal detection API

이미지를 업로드하면 검출 결과 이미지(`result_with_boxes.jpg`)를 반환합니다.

```bash
conda activate deepseek-ocr
# 기본 포트 20900
PORT=20900 python normal_detection/detection_api.py
```

```bash
# 호출 예시 (ref 생략 시 기본: tank, war plane, war ship, helicopter and soldier)
curl -X POST "http://localhost:20900/detect" \
  -F "file=@image.jpg" \
  -F "ref=tank" \
  -o result_with_boxes.jpg
```

- **환경 변수**: `PORT` (기본 `20900`), 폼 필드 `cuda_device`는 요청 시 `CUDA_VISIBLE_DEVICES`로 전달 (기본 `"0"`).
- **curl 멀티라인**: 줄 끝의 `\` 바로 뒤에 공백이 있으면 셸이 줄을 이어 붙이지 못해 `curl: (3) URL rejected` 등이 납니다. 한 줄로 쓰거나 `\`를 줄의 마지막 문자로 두세요.

### 2. Chat API

텍스트와 이미지를 옵션으로 받아 Qwen3-VL로 답변을 생성합니다.

```bash
conda activate qwen3-vl
# 기본 포트 PORT=20901
PORT=20901 python chat/chat_api.py
```

```bash
# 텍스트 + 이미지
curl -X POST "http://localhost:20901/chat" \
  -F "text=이 이미지에 뭐가 있어?" \
  -F "file=@image.jpg"

# 텍스트만
curl -X POST "http://localhost:20901/chat" -F "text=인공지능이란?"

# 이미지만 (기본 질문으로 이미지 설명)
curl -X POST "http://localhost:20901/chat" -F "file=@image.jpg"
```

- **환경 변수**: `PORT`, `CHAT_MODEL_NAME`, `CHAT_MAX_NEW_TOKENS`

### 3. Thermal detection API

열화상 이미지에서 사람 등 객체를 검출합니다.

```bash
conda activate yolov11n-thermal
# 기본 포트 20901 (다른 API와 동시에 띄울 때는 PORT로 분리)
PORT=20901 python thermal_detection/thermal_detection_api.py
```

```bash
curl -X POST "http://localhost:20901/thermal/detect" \
  -F "file=@/path/to/img.jpg" \
  -o result.jpg \
  -D headers.txt
```

응답 헤더 `X-Detection-Results`에 bbox JSON이 포함됩니다(자세한 형식은 `thermal_detection_api.py` 참고).

---

## API 요약

| API | 경로 | 기본 포트 | 설명 |
|-----|------|-----------|------|
| Normal Detection | `POST /detect` | 20900 | 이미지 + 선택 `ref` → `result_with_boxes.jpg` |
| Normal Detection | `GET /health` | 동일 | 서버 상태 |
| Thermal Detection | `POST /thermal/detect` | 20901 | 열화상 이미지 → 결과 이미지 + 검출 헤더 |
| Thermal Detection | `GET /health` | 동일 | 서버 상태 |
| Chat | `POST /chat` | `8000` (`PORT`로 변경) | `text` / `file`(이미지) 옵션 → `{"answer": "..."}` |
| Chat | `GET /health` | 동일 | 서버 상태 |

---

## 라이선스

(프로젝트에 맞게 추가하세요.)
