from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import os
import random
import json
import time
import concurrent.futures
import threading
import uuid
from loguru import logger
from tqdm import tqdm

# 환경 변수는 QAGenerator.__init__에서 Config 값으로 설정됨


@dataclass
class Qwen3vlQaConfig:
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    input_path: str = "ocr_output"  # OCR로 변환된 데이터 디렉토리
    output_path: str = "data/results_qa.json"
    max_retries: int = 2
    retry_delay: int = 5
    api_call_delay: float = 0.1  # Qwen3-VL은 로컬 모델이므로 짧은 딜레이
    max_workers: int = 1  # 멀티모달 모델은 GPU 메모리 제약으로 1로 설정 권장
    num_questions_per_document: int = 1
    candidate_questions_per_call: int = 2
    max_new_tokens: int = 256  # 최대 생성 토큰 수
    config_file: Optional[str] = "datamorgana_config_template.json"  # 설정 파일 경로
    log_file: Optional[str] = None  # 로그 파일 경로
    # Qwen3-VL 모델 생성 파라미터
    greedy: bool = False
    top_p: float = 0.8
    top_k: int = 20
    temperature: float = 0.7
    repetition_penalty: float = 1.0
    presence_penalty: float = 1.5
    out_seq_length: int = 16384


@dataclass
class Category:
    name: str
    probability: float
    description: str


@dataclass
class Categorization:
    name: str
    categories: List[Category]


class QAGenerator:
    def __init__(self, config: Qwen3vlQaConfig | None = None):
        if config is None:
            config = Qwen3vlQaConfig()
        self.config = config
        self.user_categorizations: List[Categorization] = []
        self.question_categorizations: List[Categorization] = []

        # Qwen3-VL 모델 생성 파라미터를 환경 변수로 설정
        os.environ["greedy"] = str(config.greedy).lower()
        os.environ["top_p"] = str(config.top_p)
        os.environ["top_k"] = str(config.top_k)
        os.environ["temperature"] = str(config.temperature)
        os.environ["repetition_penalty"] = str(config.repetition_penalty)
        os.environ["presence_penalty"] = str(config.presence_penalty)
        os.environ["out_seq_length"] = str(config.out_seq_length)

        # Load model and processor (기존 코드 그대로)
        logger.info(f"모델 로딩 중: {config.model_name}")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.model_name, dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(config.model_name)
        logger.info("모델 로딩 완료")

    def load_configuration(self, config_path: str):
        """Load DataMorgana configuration file"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)

            # Load user categorizations
            if "user_categorizations" in config_data:
                for cat_data in config_data["user_categorizations"]:
                    categories = [
                        Category(
                            name=cat["name"],
                            probability=cat["probability"],
                            description=cat["description"],
                        )
                        for cat in cat_data["categories"]
                    ]
                    self.user_categorizations.append(
                        Categorization(name=cat_data["name"], categories=categories)
                    )

            # Load question categorizations
            if "question_categorizations" in config_data:
                for cat_data in config_data["question_categorizations"]:
                    categories = [
                        Category(
                            name=cat["name"],
                            probability=cat["probability"],
                            description=cat["description"],
                        )
                        for cat in cat_data["categories"]
                    ]
                    self.question_categorizations.append(
                        Categorization(name=cat_data["name"], categories=categories)
                    )

            logger.info(
                f"설정 불러오기 성공: {len(self.user_categorizations)}개의 사용자 분류, {len(self.question_categorizations)}개의 질문 분류"
            )

        except Exception as e:
            logger.error(f"설정 파일 불러오기 실패: {e}")

    def select_categories(self) -> Tuple[List[Category], List[Category]]:
        """Select user and question categories based on probability"""
        selected_user_categories = []
        selected_question_categories = []

        # Select one category for each user categorization
        for categorization in self.user_categorizations:
            categories = categorization.categories
            weights = [cat.probability for cat in categories]
            selected = random.choices(categories, weights=weights, k=1)[0]
            selected_user_categories.append(selected)

        # Select one category for each question categorization
        for categorization in self.question_categorizations:
            categories = categorization.categories
            weights = [cat.probability for cat in categories]
            selected = random.choices(categories, weights=weights, k=1)[0]
            selected_question_categories.append(selected)

        return selected_user_categories, selected_question_categories

    def build_prompt(
        self,
        document: str,
        user_categories: List[Category],
        question_categories: List[Category],
        num_questions: int = 2,
    ) -> str:
        """Build DataMorgana prompt template (based on paper)"""

        # Build user characteristic descriptions
        user_descriptions = []
        for cat in user_categories:
            user_descriptions.append(f"다음 특성을 가져야 합니다: {cat.description}")

        # Build question characteristic descriptions
        question_descriptions = []
        for cat in question_categories:
            question_descriptions.append(
                f"다음 특성을 가져야 합니다: {cat.description}"
            )

        prompt = f"""당신은 대화를 시작하기 위한 {num_questions}개의 후보 질문을 생성하는 사용자 시뮬레이터입니다.

        {num_questions}개의 질문은 지금 제공될 문서에 포함된 **사실 정보**를 기반으로 해야 합니다. 질문을 생성할 때, 시뮬레이션되는 실제 사용자와 질문을 읽는 독자는 **이 문서에 직접 접근할 수 없다고 가정**하세요. 따라서 문서의 저자, 출처, 또는 ‘이 문서에서는’과 같은 표현을 사용하지 마세요.
        각 질문은 **독립적으로 읽혀도 이해 가능해야 하며**, 서로 **내용과 관점이 다르게** 구성되어야 합니다. 서문이나 설명 없이 **질문과 답변만** 반환하세요.
        출력은 각 줄마다 다음 JSON 형식을 따르세요:
        - {{"question": "<question>", "answer": "<answer>"}}

        중요 지침:
        - 질문에는 반드시 문서에 등장하는 **구체적인 개체, 용어, 개념, 사건 등**을 명시적으로 포함해야 합니다.
        - `"이 것"`, `"그 방법"`, `"해당 내용"`, `"그 사례"`와 같은 **모호한 대명사나 추상적 지칭은 사용하지 마세요.**
        - 대신 `"○○ 정책"`, `"△△ 기술"`, `"□□ 시스템"`과 같이 **실제 문서에 등장하는 명확한 명칭이나 표현**을 사용하세요.
        - 이렇게 하면 질문이 단독으로 읽혀도 의미가 분명해지고, 문서 외부 독자에게도 이해 가능합니다.

        생성된 질문과 답변은 **다음 문서에 포함된 사실만**을 기반으로 해야 하며, 추측이나 외부 지식은 사용하지 마세요:
        - {document}

        생성된 각 질문은 다음 특성을 가진 사용자를 반영해야 합니다:
        - {chr(10).join(user_descriptions)}

        생성된 각 질문은 다음 특성을 가져야 합니다:
        - {chr(10).join(question_descriptions)}"""

        return prompt

    def get_model_response(
        self, prompt: str, image_path: str, markdown_content: str = None
    ) -> Optional[str]:
        """Call model to get response (기존 코드 그대로)"""
        for attempt in range(self.config.max_retries):
            try:
                # Build messages
                if image_path and os.path.exists(image_path):
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": image_path,
                                },
                                {
                                    "type": "text",
                                    "text": prompt,
                                },
                            ],
                        }
                    ]
                else:
                    # 이미지가 없으면 텍스트만 사용
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt,
                                },
                            ],
                        }
                    ]

                # Preparation for inference (기존 코드 그대로)
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(self.model.device)

                # Inference: Generation of the output (기존 코드 그대로)
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=self.config.max_new_tokens
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                time.sleep(self.config.api_call_delay)
                return output_text[0] if output_text else None
            except Exception as e:
                logger.error(
                    f"모델 추론 실패 (시도 {attempt+1}/{self.config.max_retries}): {e}"
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2**attempt))
                else:
                    return None
        return None

    def parse_qa_pairs(self, response: str) -> List[Dict[str, str]]:
        """Parse Q&A pairs from model response"""
        qa_pairs = []

        try:
            # Try parsing JSON line by line
            lines = response.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line and line.startswith("{") and line.endswith("}"):
                    try:
                        qa_pair = json.loads(line)
                        if "question" in qa_pair and "answer" in qa_pair:
                            qa_pairs.append(qa_pair)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"Q&A 쌍 파싱 실패: {e}")

        return qa_pairs

    def filter_qa_pairs(
        self,
        qa_pairs: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Filter Q&A pairs to ensure quality (based on paper's filtering strategy)"""
        filtered_pairs = []

        for qa_pair in qa_pairs:
            question = qa_pair.get("question", "").strip()
            answer = qa_pair.get("answer", "").strip()

            # Basic quality check
            if len(question) < 3 or len(answer) < 3:  # Relaxed length limit
                continue

            # Check if document is referenced (violates context-free requirement)
            # Check English references
            english_refs = ["document", "text", "passage", "author"]
            # Check Korean references
            korean_refs = ["문서", "자료", "텍스트", "단락", "저자"]

            question_lower = question.lower()
            if any(ref in question_lower for ref in english_refs) or any(
                ref in question for ref in korean_refs
            ):
                continue

            filtered_pairs.append(qa_pair)

        return filtered_pairs

    def process_document(self, document_item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single document and generate Q&A pairs"""
        try:
            document_id = document_item.get("id", "unknown")
            markdown_path = document_item.get("markdown_path", "")
            images_dir = document_item.get("images_dir", "")

            # Read markdown content
            try:
                with open(markdown_path, "r", encoding="utf-8") as f:
                    markdown_content = f.read()
            except FileNotFoundError:
                logger.warning(f"마크다운 파일을 찾을 수 없습니다: {markdown_path}")
                return document_item

            if not markdown_content:
                logger.warning(f"문서 {document_id} 내용이 비어 있습니다")
                return document_item

            # Get first image from images directory if exists
            image_path = None
            if images_dir and os.path.exists(images_dir):
                image_files = sorted(
                    [
                        f
                        for f in os.listdir(images_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    ]
                )
                if image_files:
                    image_path = os.path.join(images_dir, image_files[0])
                else:
                    logger.warning(f"이미지 디렉토리에 이미지가 없습니다: {images_dir}")
            else:
                logger.warning(f"이미지 디렉토리가 없습니다: {images_dir}")

            generated_qa_pairs = []

            # Generate specified number of Q&A pairs
            for i in range(self.config.num_questions_per_document):
                # Select categories (Step 1)
                user_categories, question_categories = self.select_categories()

                # Build prompt (Step 2-3)
                prompt = self.build_prompt(
                    markdown_content,
                    user_categories,
                    question_categories,
                    self.config.candidate_questions_per_call,
                )

                # Get model response (image_path가 없으면 None 전달)
                if image_path and os.path.exists(image_path):
                    response = self.get_model_response(
                        prompt, image_path, markdown_content
                    )
                else:
                    # 이미지가 없으면 텍스트만으로 처리 (Qwen3-VL은 이미지 없이도 동작 가능)
                    response = self.get_model_response(prompt, None, markdown_content)

                if not response:
                    logger.warning(f"문서 {document_id} {i+1}번째 생성 실패")
                    continue

                # Parse Q&A pairs
                qa_pairs = self.parse_qa_pairs(response)

                # Filter Q&A pairs (Step 4)
                filtered_pairs = self.filter_qa_pairs(qa_pairs)

                # Select best Q&A pair
                if filtered_pairs:
                    selected_pair = random.choice(filtered_pairs)
                    selected_pair["user_categories"] = ", ".join(
                        [cat.name for cat in user_categories]
                    )
                    selected_pair["question_categories"] = ", ".join(
                        [cat.name for cat in question_categories]
                    )
                    selected_pair["document_id"] = document_id
                    selected_pair["images_dir"] = images_dir
                    generated_qa_pairs.append(selected_pair)

            document_item["generated_qa_pairs"] = generated_qa_pairs
            logger.debug(
                f"문서 {document_id}에서 {len(generated_qa_pairs)}개의 Q&A 쌍을 생성했습니다"
            )

        except Exception as e:
            logger.error(f"문서를 처리하는 중 오류가 발생했습니다.: {e}")

        return document_item

    def scan_ocr_output(self, ocr_output_dir: str) -> List[Dict[str, Any]]:
        """Scan ocr_output directory and create document list"""
        documents = []

        if not os.path.exists(ocr_output_dir):
            logger.error(f"OCR directory가 존재하지 않습니다: {ocr_output_dir}")
            return documents

        # ocr_output 디렉토리 구조: ocr_output/pdf_1/[page_id]/result.mmd
        for pdf_dir in sorted(os.listdir(ocr_output_dir)):
            pdf_path = os.path.join(ocr_output_dir, pdf_dir)
            if not os.path.isdir(pdf_path):
                continue

            # 각 PDF 디렉토리 내의 페이지 디렉토리 순회
            for page_dir in sorted(os.listdir(pdf_path)):
                page_path = os.path.join(pdf_path, page_dir)
                if not os.path.isdir(page_path):
                    continue

                markdown_path = os.path.join(page_path, "result.mmd")
                images_dir = os.path.join(page_path, "images")

                # result.mmd 파일이 있는지 확인
                if os.path.exists(markdown_path):
                    # UUID로 고유 ID 생성
                    document_id = str(uuid.uuid4())
                    documents.append(
                        {
                            "id": document_id,
                            "pdf_dir": pdf_dir,
                            "page_dir": page_dir,
                            "markdown_path": markdown_path,
                            "images_dir": (
                                images_dir if os.path.exists(images_dir) else ""
                            ),
                        }
                    )
                    logger.debug(
                        f"문서 발견: {document_id} ({pdf_dir}/{page_dir}) - {markdown_path}"
                    )

        logger.info(f"총 {len(documents)}개의 문서를 발견했습니다")
        return documents

    def generate_benchmark(
        self, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate entire benchmark dataset (using thread pool for acceleration)"""
        logger.info(
            f"벤치마크 데이터셋 생성을 시작합니다, 총 {len(documents)}개의 문서"
        )

        results = []
        completed = 0

        # Shared counter for QA pairs (thread-safe)
        questions_count = 0
        save_counter = 0  # Counter for file naming (100, 200, 300, ...)
        counter_lock = threading.Lock()

        def save_results():
            """Save current results to temporary file (thread-safe)"""
            nonlocal save_counter
            try:
                save_counter += 1
                file_number = save_counter * 100
                temp_path = f"{self.config.output_path}_temp_{file_number}.json"

                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

            except Exception as e:
                logger.error(f"중간 저장 실패: {e}")

        with tqdm(total=len(documents), desc="생성 진행 상황") as pbar:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.max_workers
            ) as executor:
                future_to_doc = {
                    executor.submit(self.process_document, doc): doc
                    for doc in documents
                }
                for future in concurrent.futures.as_completed(future_to_doc):
                    result = future.result()
                    results.append(result)

                    # Count QA pairs in this result (thread-safe)
                    qa_pairs_count = len(result.get("generated_qa_pairs", []))

                    with counter_lock:
                        questions_count += qa_pairs_count
                        if questions_count >= 100:
                            save_results()
                            questions_count = 0

                    completed += 1
                    progress = int((completed / len(documents)) * 100)
                    print(
                        f"PROGRESS:{progress}:문서 처리 중... ({completed}/{len(documents)})",
                        flush=True,
                    )
                    pbar.update(1)

        return results


def run_qa_generation(config: Qwen3vlQaConfig | None = None):
    """
    QA 생성을 실행하는 함수 (외부에서 호출 가능)

    Args:
        config: Qwen3vlQaConfig 객체 (None이면 기본값 사용)
    """
    if config is None:
        config = Qwen3vlQaConfig()

    # 로그 파일 설정 (파일과 콘솔 둘 다 출력)
    if config.log_file:
        logger.add(
            config.log_file,
            rotation="10 MB",
            retention="7 days",
            encoding="utf-8",
            level="INFO",
            enqueue=False,
        )

    # Create generator
    generator = QAGenerator(config)

    # Use config file if specified
    if config.config_file:
        try:
            generator.load_configuration(config.config_file)
            logger.info(f"사용자 지정 설정 파일을 사용합니다: {config.config_file}")
        except FileNotFoundError:
            logger.error(f"지정된 설정 파일을 찾을 수 없습니다: {config.config_file}")

    # Load document data from OCR output directory
    if os.path.isdir(config.input_path):
        # OCR directory 스캔
        logger.info(f"OCR directory를 스캔합니다: {config.input_path}")
        documents = generator.scan_ocr_output(config.input_path)
        if not documents:
            logger.error(
                f"OCR directory에서 문서를 찾을 수 없습니다: {config.input_path}"
            )
            return
    else:
        # 기존 방식: JSON 파일에서 로드
        try:
            with open(config.input_path, "r", encoding="utf-8") as f:
                documents = json.load(f)
            logger.info(f"{len(documents)}개의 문서를 성공적으로 불러왔습니다")
        except FileNotFoundError:
            logger.error(f"입력 파일 {config.input_path} 이(가) 존재하지 않습니다")
            # Create example documents
            documents = [
                {
                    "id": str(uuid.uuid4()),
                    "image_id": "0",
                    "content": "인공지능(AI)**은 컴퓨터가 인간의 지능을 모방하도록 만드는 방법을 연구하는 과학 기술이다. 여기에는 기계학습, 딥러닝, 자연어 처리 등 여러 하위 분야가 포함된다. 최근에는 계산 능력의 향상과 빅데이터의 발전에 힘입어 AI 기술이 획기적인 발전을 이루었다.",
                },
                {
                    "id": str(uuid.uuid4()),
                    "image_id": "0",
                    "content": "기계학습(머신러닝)**은 인공지능의 중요한 한 분야로, 알고리즘을 통해 컴퓨터가 명시적인 프로그래밍 지시 없이도 데이터로부터 자동으로 학습하고 개선할 수 있도록 한다. 대표적인 기계학습 방법에는 지도학습, 비지도학습, 강화학습이 있다.",
                },
            ]
            logger.info("예시 문서를 사용하여 시연합니다.")

    # Generate benchmark dataset
    print(f"PROGRESS:0:총 {len(documents)}개 문서 처리 시작", flush=True)
    results = generator.generate_benchmark(documents)
    print("PROGRESS:100:벤치마크 데이터셋 생성 완료", flush=True)

    # Save results
    with open(config.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Statistics of generated results
    total_qa_pairs = sum(len(doc.get("generated_qa_pairs", [])) for doc in results)
    logger.success("벤치마크 데이터셋 생성 완료!")
    logger.info(f"총 처리한 문서 수: {len(results)}")
    logger.info(f"총 생성된 Q&A 쌍 수: {total_qa_pairs}")
    logger.info(f"결과가 저장된 위치: {config.output_path}")
