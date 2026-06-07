from __future__ import annotations

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image


def render_page_with_highlight(
    file_bytes: bytes,
    page_number: int,
    coords: dict,
    zoom: float = 2.0,
    color: tuple[int, int, int] = (99, 102, 241),
    thickness: int = 4,
) -> Image.Image:
    """
    Renderiza uma página do PDF como imagem e desenha um retângulo
    à volta do bloco de texto fonte (XAi visual).

    Args:
        file_bytes: Bytes do ficheiro PDF.
        page_number: Número da página (1-indexed).
        coords: Dicionário com x, y, w, h em pixels (espaço zoom).
        zoom: Fator de zoom usado na extração OCR (default 2.0).
        color: Cor BGR do retângulo (default amarelo).
        thickness: Espessura do retângulo em pixels.

    Returns:
        Imagem PIL com o retângulo desenhado.
    """
    pdf = fitz.open(stream=file_bytes, filetype="pdf")
    # fitz usa índice 0-based
    page = pdf[page_number - 1]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    pdf.close()

    # Converter pixmap → numpy array BGR
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Desenhar retângulo à volta do bloco
    if coords:
        x, y, w, h = (
            int(coords.get("x", 0)),
            int(coords.get("y", 0)),
            int(coords.get("w", 0)),
            int(coords.get("h", 0)),
        )
        # Padding para o retângulo ficar mais visível
        pad = 6
        cv2.rectangle(
            img,
            (max(0, x - pad), max(0, y - pad)),
            (x + w + pad, y + h + pad),
            color,
            thickness,
        )

    # Converter para PIL (RGB) para o Streamlit
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def confidence_color(confidence: float) -> str:
    """Retorna cor CSS baseada na confiança OCR."""
    if confidence >= 80:
        return "#22c55e"   # verde
    elif confidence >= 60:
        return "#f59e0b"   # amarelo
    else:
        return "#ef4444"   # vermelho


def confidence_label(confidence: float) -> str:
    """Rótulo textual da confiança OCR."""
    if confidence >= 80:
        return "Alta"
    elif confidence >= 60:
        return "Média"
    else:
        return "Baixa"
