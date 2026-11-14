"""Etapa 2: pré-processamento da placa recortada para melhorar o OCR.

Fluxo: ROI dos caracteres -> normalização de iluminação -> binarização (várias
variantes) -> isolamento dos caracteres por componentes conectados.
"""
import cv2
import numpy as np

# Variantes de binarização. Cada uma "erra" em placas diferentes; o OCR vota entre elas.
VARIANTES = ("otsu", "blur7", "blur11", "blur15")


def normalizar(placa_bgr):
    """Recorta a região dos caracteres e remove gradientes de iluminação."""
    h, w = placa_bgr.shape[:2]
    # Tira a faixa azul superior, o "BR" à esquerda e as bordas pretas
    roi = placa_bgr[int(h * 0.23):int(h * 0.80), int(w * 0.08):int(w * 0.99)]
    cinza = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    cinza = cv2.resize(cinza, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # Divide pelo fundo estimado (fechamento morfológico grande): sombras/reflexos somem
    fundo = cv2.morphologyEx(cinza, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61)))
    fundo = cv2.GaussianBlur(fundo, (31, 31), 0)
    return cv2.divide(cinza, fundo, scale=255)


def binarizar(cinza, variante):
    """Otsu sobre imagem suavizada. O blur apaga a marca d'água dentro dos glifos."""
    if variante == "otsu":
        x = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(cinza)
        x = cv2.bilateralFilter(x, 7, 50, 50)
    else:  # "blurK": desfoque gaussiano de lado K
        k = int(variante[4:])
        x = cv2.GaussianBlur(cinza, (k, k), 0)
    _, b = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Garante texto escuro sobre fundo claro
    return cv2.bitwise_not(b) if b.mean() < 127 else b


def _fundir_empilhados(caixas):
    """Une pedaços do mesmo caractere partido na vertical (ex.: '5' cortado por sombra).

    caixas: lista de dicts {x0,y0,x1,y1,rotulos:set}. Funde pares com forte sobreposição
    horizontal e pouca distância vertical.
    """
    mudou = True
    while mudou:
        mudou = False
        for i in range(len(caixas)):
            for j in range(i + 1, len(caixas)):
                a, b = caixas[i], caixas[j]
                inter = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])
                menor = min(a["x1"] - a["x0"], b["x1"] - b["x0"])
                gap = max(a["y0"], b["y0"]) - min(a["y1"], b["y1"])
                if menor > 0 and inter >= 0.6 * menor and gap <= 15:
                    a.update(x0=min(a["x0"], b["x0"]), y0=min(a["y0"], b["y0"]),
                             x1=max(a["x1"], b["x1"]), y1=max(a["y1"], b["y1"]),
                             rotulos=a["rotulos"] | b["rotulos"])
                    caixas.pop(j)
                    mudou = True
                    break
            if mudou:
                break
    return caixas


def isolar_caracteres(binaria):
    """Mantém só componentes com tamanho de caractere.

    Retorna (imagem_da_linha_de_texto, lista_de_mascaras_ordenadas_por_x).
    A imagem fica com texto preto sobre fundo branco, pronta para o Tesseract.
    """
    H, W = binaria.shape
    n, rotulos, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(binaria), 8)
    caixas = [dict(x0=x, y0=y, x1=x + w, y1=y + h, rotulos={i})
              for i, (x, y, w, h, a) in enumerate(stats)
              if i > 0 and a > 40 and w <= 0.30 * W and h <= 0.98 * H]
    caixas = _fundir_empilhados(caixas)

    # Filtra por tamanho/proporção (descarta barras, bordas e fragmentos finos)
    caixas = [c for c in caixas
              if 0.30 * H <= c["y1"] - c["y0"] and (c["x1"] - c["x0"]) >= 0.28 * (c["y1"] - c["y0"])]
    if caixas:  # caracteres têm altura parecida: descarta quem destoa da mediana
        med = np.median([c["y1"] - c["y0"] for c in caixas])
        caixas = [c for c in caixas if 0.70 * med <= c["y1"] - c["y0"] <= 1.30 * med]
    caixas.sort(key=lambda c: c["x0"])

    limpa = np.full_like(binaria, 255)
    mascaras = []
    for c in caixas:
        m = np.isin(rotulos, list(c["rotulos"]))
        limpa[m] = 0
        mascaras.append(m[c["y0"]:c["y1"], c["x0"]:c["x1"]])
    if caixas:
        x0 = max(min(c["x0"] for c in caixas) - 15, 0)
        x1 = min(max(c["x1"] for c in caixas) + 15, W)
        y0 = max(min(c["y0"] for c in caixas) - 10, 0)
        y1 = min(max(c["y1"] for c in caixas) + 10, H)
        limpa = limpa[y0:y1, x0:x1]
    limpa = cv2.copyMakeBorder(limpa, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    return limpa, mascaras
