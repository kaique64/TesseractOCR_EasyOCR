"""Etapa 3 (EasyOCR): reconhecimento, votação entre variantes e correção pelo padrão Mercosul.

Reaproveita segmentação e pré-processamento do pipeline do Tesseract; só o OCR muda.
"""
import re
from collections import Counter

import cv2

from src.geometria import desempatar_placa
from src.ocr_tesseract import ALFABETO, PADRAO, corrigir_padrao
from src.preprocessamento import VARIANTES, binarizar, isolar_caracteres

_leitor = None


def obter_leitor():
    """Cria o Reader uma única vez (a primeira execução baixa os pesos do modelo)."""
    global _leitor
    if _leitor is None:
        import easyocr
        _leitor = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _leitor


def _ler_linha(img):
    """Reconhece a linha inteira SEM rodar o detector CRAFT (a imagem já é só a linha)."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    resultados = obter_leitor().recognize(img, horizontal_list=[[0, w, 0, h]], free_list=[],
                                          allowlist=ALFABETO, detail=0)
    return re.sub(r"[^A-Z0-9]", "", "".join(resultados).upper())


def ler_placa(cinza_normalizada):
    """Retorna (leituras_brutas, texto_final) para uma placa já normalizada."""
    brutas, mascaras_ref = [], None
    for variante in VARIANTES:
        linha, mascaras = isolar_caracteres(binarizar(cinza_normalizada, variante))
        if variante == "blur7":
            mascaras_ref = mascaras
        brutas.append(_ler_linha(linha))

    corrigidas = [corrigir_padrao(t) for t in brutas]
    validas = [t for t in corrigidas if len(t) == 7]
    if not validas:
        return brutas, max(corrigidas, key=len, default="")
    final = [Counter(t[i] for t in validas).most_common(1)[0][0] for i in range(7)]

    # Desempate de pares parecidos (I/L, Y/V, Q/O, 7/2, 4/1) pela forma do glifo
    return brutas, desempatar_placa("".join(final), mascaras_ref or [], PADRAO)
