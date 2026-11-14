"""Etapa 3: OCR com Tesseract, votação entre variantes e correção pelo padrão Mercosul."""
import re
from collections import Counter

import pytesseract

from src.preprocessamento import VARIANTES, binarizar, isolar_caracteres

ALFABETO = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
PSMS = (7, 13)  # 7: linha única | 13: linha "crua", sem heurísticas de página
PADRAO = "LLLNLNN"  # formato Mercosul: ABC1D23 (L = letra, N = número)

# Confusões observadas nesta fonte/dataset. Note que o "0" costuma sair como "G"
# por causa da marca d'água, enquanto o "6" verdadeiro sai como "6".
PARA_LETRA = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B", "4": "A", "7": "T", "9": "Q"}
PARA_NUMERO = {"O": "0", "Q": "0", "D": "0", "G": "0", "U": "0",
               "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "A": "4", "T": "7"}


def corrigir_padrao(texto):
    """Se tem 7 caracteres, força letra/número conforme a posição do padrão."""
    texto = re.sub(r"[^A-Z0-9]", "", texto.upper())
    if len(texto) != 7:
        return texto
    return "".join(
        (PARA_LETRA.get(c, c) if c.isdigit() else c) if t == "L"
        else (PARA_NUMERO.get(c, c) if c.isalpha() else c)
        for c, t in zip(texto, PADRAO))


def _ler_linha(img, psm):
    cfg = f"--oem 3 --psm {psm} -c tessedit_char_whitelist={ALFABETO}"
    return re.sub(r"[^A-Z0-9]", "", pytesseract.image_to_string(img, config=cfg).upper())


def _largura_topo(mascara):
    """Fração da largura do glifo coberta por tinta nos 15% superiores.
    'I' tem serifa no topo (~0.6); 'L' só tem a haste (~0.3)."""
    h, w = mascara.shape
    return mascara[: max(1, int(0.15 * h))].any(axis=0).sum() / w


def ler_placa(cinza_normalizada):
    """Retorna (leituras_brutas, texto_final) para uma placa já normalizada."""
    brutas, mascaras_ref = [], None
    for variante in VARIANTES:
        linha, mascaras = isolar_caracteres(binarizar(cinza_normalizada, variante))
        if variante == "blur7":
            mascaras_ref = mascaras
        for psm in PSMS:
            brutas.append(_ler_linha(linha, psm))

    # Votação por posição entre as leituras de 7 caracteres (já corrigidas pelo padrão)
    corrigidas = [corrigir_padrao(t) for t in brutas]
    validas = [t for t in corrigidas if len(t) == 7]
    if not validas:
        return brutas, max(corrigidas, key=len, default="")
    final = [Counter(t[i] for t in validas).most_common(1)[0][0] for i in range(7)]

    # Desempate I x L pela geometria do glifo (só se achamos exatamente 7 caracteres)
    if mascaras_ref and len(mascaras_ref) == 7:
        for i, m in enumerate(mascaras_ref):
            if PADRAO[i] == "L" and final[i] in "IL":
                final[i] = "I" if _largura_topo(m) > 0.45 else "L"
    return brutas, "".join(final)
