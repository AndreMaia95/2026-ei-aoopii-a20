from __future__ import annotations

from typing import Any

import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract

from config.settings import OCR_LANG, OCR_MIN_CONFIDENCE, PDF_RENDER_ZOOM, TESSERACT_CMD


if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def validate_tesseract() -> str:
    """Validate that Tesseract is available and return its version."""
    try:
        return str(pytesseract.get_tesseract_version())
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract OCR não foi encontrado. Instala o Tesseract ou define "
            "a variável TESSERACT_CMD com o caminho do executável."
        ) from exc


def preprocess_for_ocr(uploaded_file: Any) -> list[dict[str, Any]]:
    """
    Read an uploaded PDF/image and prepare pages for OCR.

    Returns one dictionary per page with:
    - page_number: 1-based page number
    - processed_img: thresholded image used by OCR
    - original_img: original BGR image, useful later for visual evidence
    """
    filename = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()
    pages: list[dict[str, Any]] = []

    if filename.endswith(".pdf"):
        pdf = fitz.open(stream=file_bytes, filetype="pdf")
        zoom = fitz.Matrix(PDF_RENDER_ZOOM, PDF_RENDER_ZOOM)

        for page_index, page in enumerate(pdf, start=1):
            pix = page.get_pixmap(matrix=zoom)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height,
                pix.width,
                pix.n,
            )

            if pix.n == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            elif pix.n == 1:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            else:
                raise ValueError(f"Formato de imagem PDF não suportado: {pix.n} canais.")

            processed_img, original_img = _process_image(img_array)
            pages.append(
                {
                    "page_number": page_index,
                    "processed_img": processed_img,
                    "original_img": original_img,
                }
            )

        pdf.close()
        return pages

    img_array = cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_array is None:
        raise ValueError("Não foi possível ler a imagem. Verifica o formato do ficheiro.")

    processed_img, original_img = _process_image(img_array)
    pages.append(
        {
            "page_number": 1,
            "processed_img": processed_img,
            "original_img": original_img,
        }
    )
    return pages


def _process_image(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply basic preprocessing to improve OCR quality."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh, img


def segment_and_ocr(processed_img: np.ndarray, page_number: int) -> list[dict[str, Any]]:
    """
    Extract text blocks with Tesseract and keep coordinates for explainability.
    """
    custom_config = f"--oem 3 --psm 3 -l {OCR_LANG}"
    data = pytesseract.image_to_data(
        processed_img,
        config=custom_config,
        output_type=pytesseract.Output.DICT,
    )

    blocks: list[dict[str, Any]] = []
    current_block_key: tuple[int, int, int] | None = None
    current_text: list[str] = []
    current_confidences: list[float] = []
    left = top = right = bottom = 0

    def flush_current_block() -> None:
        nonlocal current_text, current_confidences, left, top, right, bottom
        if not current_text:
            return

        avg_confidence = (
            round(sum(current_confidences) / len(current_confidences), 2)
            if current_confidences
            else None
        )
        blocks.append(
            {
                "text": " ".join(current_text),
                "page_number": page_number,
                "coords": {
                    "x": int(left),
                    "y": int(top),
                    "w": int(max(0, right - left)),
                    "h": int(max(0, bottom - top)),
                },
                "confidence": avg_confidence,
            }
        )
        current_text = []
        current_confidences = []

    for i, raw_text in enumerate(data["text"]):
        text = raw_text.strip()
        if not text:
            continue

        try:
            confidence = float(data["conf"][i])
        except (TypeError, ValueError):
            confidence = -1

        if confidence < OCR_MIN_CONFIDENCE:
            continue

        block_key = (
            int(data["page_num"][i]),
            int(data["block_num"][i]),
            int(data["par_num"][i]),
        )

        word_left = int(data["left"][i])
        word_top = int(data["top"][i])
        word_right = word_left + int(data["width"][i])
        word_bottom = word_top + int(data["height"][i])

        if block_key != current_block_key:
            flush_current_block()
            current_block_key = block_key
            left, top, right, bottom = word_left, word_top, word_right, word_bottom
        else:
            left = min(left, word_left)
            top = min(top, word_top)
            right = max(right, word_right)
            bottom = max(bottom, word_bottom)

        current_text.append(text)
        current_confidences.append(confidence)

    flush_current_block()
    return blocks
