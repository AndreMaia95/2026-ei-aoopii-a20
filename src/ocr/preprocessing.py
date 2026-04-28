import cv2
import fitz  # PyMuPDF — para suporte a PDFs
import pytesseract
import numpy as np

# Caminho para o executável do Tesseract no Windows
# Ajusta se instalaste noutra pasta
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def preprocess_for_ocr(uploaded_file):
    """
    Lê o ficheiro enviado (PDF ou imagem) e aplica filtros para melhorar o OCR.
    Retorna uma lista de tuplos (processed_img, original_img) — um por página.
    """
    filename = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()
    pages = []

    if filename.endswith(".pdf"):
        # Converter cada página do PDF para imagem com PyMuPDF
        pdf = fitz.open(stream=file_bytes, filetype="pdf")
        for page in pdf:
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom para melhor resolução OCR
            pix = page.get_pixmap(matrix=mat)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:  # RGBA → RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            else:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            pages.append(_process_image(img_array))
        pdf.close()
    else:
        # Imagem normal (PNG, JPG, JPEG)
        img_array = cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_array is None:
            raise ValueError("Não foi possível ler a imagem. Verifica o formato do ficheiro.")
        pages.append(_process_image(img_array))

    return pages  # Lista de (processed_img, original_img)

def _process_image(img):
    """Aplica pré-processamento de imagem para melhorar qualidade do OCR."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh, img


def segment_and_ocr(processed_img):
    """
    Identifica blocos de texto, extrai o conteúdo com Tesseract 
    e guarda as coordenadas para XAi.
    """
    # Usamos o modo 6 (assumir bloco de texto único) ou 3 (automático)
    custom_config = r'--oem 3 --psm 3 -l por'
    
    # Obter dados detalhados do OCR (incluindo coordenadas)
    data = pytesseract.image_to_data(processed_img, config=custom_config, output_type=pytesseract.Output.DICT)
    
    blocks = []
    n_boxes = len(data['text'])
    
    # Agrupar palavras em blocos/linhas (simplificado para este exemplo)
    # O Tesseract já nos dá o 'block_num'
    current_block_id = -1
    current_text = []
    current_coords = {}

    for i in range(n_boxes):
        if int(data['conf'][i]) > 40: # Filtrar apenas texto com confiança razoável
            text = data['text'][i].strip()
            if text:
                block_id = data['block_num'][i]
                
                if block_id != current_block_id:
                    if current_text:
                        blocks.append({
                            "text": " ".join(current_text),
                            "coords": current_coords
                        })
                    current_text = [text]
                    current_coords = {
                        "x": data['left'][i],
                        "y": data['top'][i],
                        "w": data['width'][i],
                        "h": data['height'][i]
                    }
                    current_block_id = block_id
                else:
                    current_text.append(text)
                    # Expandir a bounding box do bloco
                    current_coords["w"] = (data['left'][i] + data['width'][i]) - current_coords["x"]
                    current_coords["h"] = max(current_coords["h"], data['height'][i])

    # Adicionar o último bloco
    if current_text:
        blocks.append({"text": " ".join(current_text), "coords": current_coords})
        
    return blocks