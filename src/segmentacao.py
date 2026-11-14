"""Etapa 1: detecção e segmentação da placa.

Estratégia: a faixa azul superior ("BRASIL") da placa Mercosul é a característica
mais distintiva. Segmentamos o azul em HSV, listamos todas as faixas alongadas
(descartando placas minúsculas de fundo) e, para cada uma, estimamos o retângulo da
placa inteira, já corrigindo rotação/perspectiva com warpPerspective.
"""
import cv2
import numpy as np

ALTURA_SOBRE_FAIXA = 5.0  # altura da placa ≈ 5x a espessura da faixa azul (a proporção
                          # largura/altura varia entre as imagens do dataset)


def _ordenar_pontos(pts):
    """Ordena 4 pontos em: topo-esq, topo-dir, base-dir, base-esq."""
    pts = np.array(pts, dtype="float32")
    soma, dif = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(soma)], pts[np.argmin(dif)],
                     pts[np.argmax(soma)], pts[np.argmax(dif)]], dtype="float32")


# Placas menores que isso (relativo à imagem e à maior placa) são tratadas como ruído de
# fundo: carros distantes cujas placas não dá para ler. Calibrado no dataset.
LARGURA_MIN_REL_IMAGEM = 0.15
LARGURA_MIN_REL_MAIOR = 0.40


def _corpo_claro(placa):
    """O corpo da placa (abaixo da faixa azul) deve ter região clara."""
    cinza = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)
    corpo = cinza[int(placa.shape[0] * .4):int(placa.shape[0] * .8)]
    # Placas cinzas/sombreadas têm pouco pixel muito claro: limiar baixo
    return np.mean(corpo > np.percentile(cinza, 40)) > 0.25 and np.mean(corpo) > 70


def segmentar_placas(img_bgr):
    """Detecta TODAS as placas legíveis da imagem.

    Retorna lista de (placa_recortada, caixa_da_placa), da maior para a menor.
    Lista vazia se nenhuma placa for encontrada.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Faixa azul da placa (H do OpenCV: 0-179; azul ~ 100-130)
    azul = cv2.inRange(hsv, (100, 120, 50), (130, 255, 255))
    # A bandeira (verde/amarelo) interrompe a faixa; incluí-la deixa a faixa inteira
    bandeira = cv2.inRange(hsv, (20, 100, 80), (85, 255, 255))
    mascara = cv2.bitwise_or(azul, bandeira)

    # Fecha buracos (texto "BRASIL", estrelas) e remove ruído
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidatos = []
    for c in contornos:
        (cx, cy), (w, h), ang = cv2.minAreaRect(c)
        if min(w, h) < 5:
            continue
        razao = max(w, h) / min(w, h)
        # Faixa azul: muito alongada e bem preenchida (carroceria azul não é retângulo)
        preenchimento = cv2.contourArea(c) / (w * h)
        if 12 <= razao <= 40 and preenchimento > 0.8:
            candidatos.append((max(w, h), c))
    if not candidatos:
        return []

    candidatos.sort(key=lambda t: -t[0])
    maior = candidatos[0][0]
    largura_img = img_bgr.shape[1]

    placas = []
    for largura, c in candidatos:
        if largura < LARGURA_MIN_REL_IMAGEM * largura_img or largura < LARGURA_MIN_REL_MAIOR * maior:
            continue  # placa pequena demais: ruído de fundo
        placa, caixa = _recortar(img_bgr, c)
        # A maior placa é sempre aceita (fallback); as demais precisam ter corpo claro
        if not placas or _corpo_claro(placa):
            placas.append((placa, caixa))
    return placas


def segmentar_placa(img_bgr):
    """Compatibilidade: só a maior placa. Retorna (placa, caixa) ou (None, None)."""
    placas = segmentar_placas(img_bgr)
    return placas[0] if placas else (None, None)


def _recortar(img_bgr, contorno):
    """Estima a placa inteira a partir da faixa azul e retifica com perspectiva."""
    faixa = _ordenar_pontos(cv2.boxPoints(cv2.minAreaRect(contorno)))
    tl, tr, br, bl = faixa
    largura_vec = tr - tl
    largura = np.linalg.norm(largura_vec)
    # vetor "para baixo" perpendicular à faixa, com altura de placa proporcional
    baixo = np.array([-largura_vec[1], largura_vec[0]]) / largura
    if baixo[1] < 0:  # garante que "baixo" aponta para baixo na imagem
        baixo = -baixo
    espessura = np.linalg.norm(bl - tl)
    altura = espessura * ALTURA_SOBRE_FAIXA
    src = np.array([tl, tr, tr + baixo * altura, tl + baixo * altura], dtype="float32")

    W = 620
    H = int(W * altura / largura)
    dst = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    placa = cv2.warpPerspective(img_bgr, M, (W, H))
    return placa, src.astype(int)
