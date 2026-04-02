"""
OCR 처리 모듈 - DeepSeek-OCR을 사용하여 이미지를 markdown으로 변환
"""

import os
import pathlib
import torch
from transformers import AutoModel, AutoTokenizer
from dataclasses import dataclass
from typing import List
import fitz  # PyMuPDF


@dataclass
class DeepSeekOCRConfig:
    # infer(self, tokenizer, prompt='', image_file='', output_path = ' ', base_size = 1024, image_size = 640, crop_mode = True, test_compress = False, save_results = False):

    # Tiny: base_size = 512, image_size = 512, crop_mode = False
    # Small: base_size = 640, image_size = 640, crop_mode = False
    # Base: base_size = 1024, image_size = 1024, crop_mode = False
    # Large: base_size = 1280, image_size = 1280, crop_mode = False
    # Gundam: base_size = 1024, image_size = 640, crop_mode = True
    model_name: str = "deepseek-ai/DeepSeek-OCR"
    base_size: int = 1024
    image_size: int = 1024
    crop_mode: bool = False
    cuda_visible_devices: str = "0"  # default to first GPU

class DetectionProcessor:
    def __init__(self, config: DeepSeekOCRConfig | None = None):
        if config is None:
            config = DeepSeekOCRConfig()
        self.config = config
        """OCR 모델 초기화"""
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.cuda_visible_devices

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.config.model_name,
            _attn_implementation="flash_attention_2",
            trust_remote_code=True,
            use_safetensors=True,
        )
        self.model = self.model.eval().cuda().to(torch.bfloat16)


    def detect_from_image(
        self, image_path: str, output_dir: str, ref: str = None
    ) -> dict:
        """
        단일 이미지를 markdown으로 변환

        Args:
            image_file: 입력 이미지 파일 경로
            output_path: 출력 디렉토리 경로
            ref: detection 대상

        Returns:
            {
                "markdown_path": "ocr_output/result.mmd",
                "images_dir": "ocr_output/images/",
                "original_image": "원본 이미지 경로"
            }
        """
        print(ref)
        if ref == None:
            ref = "tank, war plane, war ship, helicopter and soldier"
            prompt = f"<image>\nIdentify {ref} in the image and output them in bounding boxes."
        else:
            prompt = f"<image>\nLocate <|ref|>the {ref}<|/ref|> in the image."
        print(prompt)
        # 출력 디렉토리 생성
        # 파일 이름만 가져오기 (확장자 제외)
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, img_name)
        os.makedirs(output_path, exist_ok=True)

        # OCR 실행
        self.model.infer(
            self.tokenizer,
            prompt=prompt,
            image_file=image_path,
            output_path=output_path,
            base_size=self.config.base_size,
            image_size=self.config.image_size,
            crop_mode=self.config.crop_mode,
            save_results=True,
            test_compress=True,
        )
        
        # 반환값 구성
        output_path_resolved = os.path.abspath(output_path)
        return {
            "output_path": output_path_resolved,
            "original_image_path": image_path,
            "ref": ref,
        }

    


class OCRProcessor:
    def __init__(self, config: DeepSeekOCRConfig | None = None):
        if config is None:
            config = DeepSeekOCRConfig()
        self.config = config
        """OCR 모델 초기화"""
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.cuda_visible_devices

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.config.model_name,
            _attn_implementation="flash_attention_2",
            trust_remote_code=True,
            use_safetensors=True,
        )
        self.model = self.model.eval().cuda().to(torch.bfloat16)

    def process_pdf_page(
        self, pdf_page_path: str, output_dir: str, prompt: str = None
    ) -> dict:
        """
        단일 이미지를 markdown으로 변환

        Args:
            image_file: 입력 이미지 파일 경로
            output_path: 출력 디렉토리 경로
            prompt: OCR 프롬프트 (기본값: markdown 변환 프롬프트)

        Returns:
            {
                "markdown_path": "ocr_output/result.mmd",
                "images_dir": "ocr_output/images/",
                "original_image": "원본 이미지 경로"
            }
        """
        prompt = "<image>\n<|grounding|>Convert the document to markdown."

        # 출력 디렉토리 생성
        # 파일 이름만 가져오기 (확장자 제외)
        pdf_page_name = os.path.splitext(os.path.basename(pdf_page_path))[0]
        output_path = os.path.join(output_dir, pdf_page_name)
        os.makedirs(output_path, exist_ok=True)

        # OCR 실행
        self.model.infer(
            self.tokenizer,
            prompt=prompt,
            image_file=pdf_page_path,
            output_path=output_path,
            base_size=self.config.base_size,
            image_size=self.config.image_size,
            crop_mode=self.config.crop_mode,
            save_results=True,
            test_compress=True,
        )
        
        # 반환값 구성
        markdown_path = os.path.join(output_path, "result.mmd")
        images_dir = os.path.join(output_path, "images")
        
        return {
            "markdown_path": markdown_path if os.path.exists(markdown_path) else "",
            "images_dir": images_dir if os.path.exists(images_dir) else "",
            "original_image": pdf_page_path
        }

    def process_pdf_pages(self, pdf_dir: str, output_dir: str) -> list:
        """
        여러 이미지를 배치 처리

        Args:
            image_files: 입력 이미지 파일 경로 리스트
            output_base_dir: 출력 기본 디렉토리

        Returns:
            각 이미지의 처리 결과 리스트
        """
        for page in os.listdir(pdf_dir):
            pdf_page_path = os.path.join(pdf_dir, page)
            self.process_pdf_page(pdf_page_path, output_dir)

    def process_pdfs(self, pdfs_dir: str, output_dir: str) -> None:
        for pdf_dir in os.listdir(pdfs_dir):
            pdf_dir_path = os.path.join(pdfs_dir, pdf_dir)
            if os.path.isdir(pdf_dir_path):
                output_pdf_dir = os.path.join(output_dir, pdf_dir)
                self.process_pdf_pages(pdf_dir_path, output_pdf_dir)

    def convert_pdf_to_images(
        self, pdf_path: str, output_dir: str, dpi: int = 300
    ) -> List[str]:
        """
        PyMuPDF를 사용하여 PDF 파일을 PNG 이미지로 변환

        Args:
            pdf_path: 입력 PDF 파일 경로
            output_dir: 출력 디렉토리 경로
            dpi: 이미지 해상도 (기본값: 300)

        Raises:
            FileNotFoundError: PDF 파일이 존재하지 않는 경우
        """

        # PDF 파일 이름 (확장자 제외)
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

        image_paths = []

        # PDF 열기
        doc = fitz.open(pdf_path)

        # 출력 디렉토리 생성 (PDF 이름별 디렉토리)
        pdf_output_dir = os.path.join(output_dir, pdf_name)
        os.makedirs(pdf_output_dir, exist_ok=True)

        # 각 페이지를 PNG로 변환
        for page_num in range(len(doc)):
            page = doc[page_num]
            # 페이지를 pixmap으로 렌더링 (DPI 직접 지정)
            pix = page.get_pixmap(dpi=dpi)
            # PNG 파일로 저장
            image_filename = f"{page_num + 1:04d}.png"
            image_path = os.path.join(pdf_output_dir, image_filename)
            pix.save(image_path)
            image_paths.append(image_path)

        doc.close()

        return image_paths

    def convert_pdfs(self, pdf_dir: str, output_dir: str, dpi: int = 300) -> dict:
        """
        디렉토리 내의 모든 PDF 파일을 PNG 이미지로 변환

        Args:
            pdf_dir: PDF 파일들이 있는 디렉토리 경로
            output_dir: 출력 디렉토리 경로
            dpi: 이미지 해상도 (기본값: 300)

        Returns:
            {
                "total_pdfs": 총 PDF 파일 수,
                "total_images": 총 생성된 이미지 수,
                "results": {
                    "pdf_file_name": [생성된 이미지 경로 리스트],
                    ...
                }
            }

        Raises:
            FileNotFoundError: 디렉토리가 존재하지 않는 경우
        """
        if not os.path.exists(pdf_dir):
            raise FileNotFoundError(f"디렉토리를 찾을 수 없습니다: {pdf_dir}")

        if not os.path.isdir(pdf_dir):
            raise ValueError(f"경로가 디렉토리가 아닙니다: {pdf_dir}")

        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)

        # PDF 파일 찾기
        pdf_files = [
            f
            for f in os.listdir(pdf_dir)
            if os.path.isfile(os.path.join(pdf_dir, f)) and f.lower().endswith(".pdf")
        ]

        if not pdf_files:
            return {
                "total_pdfs": 0,
                "total_images": 0,
                "results": {},
            }

        results = {}
        total_images = 0

        # 각 PDF 파일 변환
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            try:
                image_paths = self.convert_pdf_to_images(
                    pdf_path=pdf_path, output_dir=output_dir, dpi=dpi
                )
                results[pdf_file] = image_paths
                total_images += len(image_paths)
            except Exception as e:
                # 에러 발생 시 빈 리스트로 저장
                results[pdf_file] = []
                print(f"PDF 변환 실패 ({pdf_file}): {e}")

        return {
            "total_pdfs": len(pdf_files),
            "total_images": total_images,
            "results": results,
        }
