"""Desempate de caracteres parecidos pela FORMA do glifo (máscara binária).

Os OCRs erram de forma consistente em alguns pares (I/L, Y/V, 7/2, 4/1, Q/O). Como a
segmentação já isola cada caractere, medimos a largura da tinta por faixa vertical.
Limiares calibrados nas 20 imagens do dataset (fonte única da placa Mercosul).
"""


def _largura(m, a, b):
    """Fração da largura do glifo coberta por tinta nas linhas [a*h, b*h)."""
    h, w = m.shape
    i0 = int(a * h)
    return m[i0:max(int(b * h), i0 + 1)].any(axis=0).sum() / w


def _razao_inferior_direita(m):
    """Tinta no canto inferior-direito / inferior-esquerdo (a cauda do Q a eleva)."""
    h, w = m.shape
    baixo = m[int(0.75 * h):]
    return baixo[:, w // 2:].sum() / max(baixo[:, : w // 2].sum(), 1)


def desempatar(caractere, mascara, tipo):
    """tipo: 'L' (posição de letra) ou 'N' (posição de número)."""
    h, w = mascara.shape
    if w / h > 1.6:  # máscara larga demais para um caractere (fragmento de borda colado)
        return caractere
    base = _largura(mascara, 0.85, 1.0)
    if tipo == "L":
        if caractere in "IL":   # 'I' tem serifa no topo; 'L' só a haste
            return "I" if _largura(mascara, 0, 0.15) > 0.45 else "L"
        if caractere in "YV":   # 'Y' tem haste estreita embaixo; 'V' termina em ponta mais larga
            return "Y" if _largura(mascara, 0.85, 0.95) < 0.42 else "V"
        if caractere in "OQ":   # 'Q' tem cauda no canto inferior-direito
            return "Q" if _razao_inferior_direita(mascara) > 1.08 else "O"
    else:
        if caractere in "27":   # '7' tem base estreita; '2' tem base cheia
            return "7" if base < 0.6 else "2"
        if caractere in "14":   # '1' tem barra de base; '4' só a haste
            return "4" if base < 0.6 else "1"
    return caractere


def desempatar_placa(texto, mascaras, padrao):
    """Aplica o desempate em cada posição quando há exatamente 7 glifos segmentados."""
    if len(texto) != 7 or len(mascaras) != 7:
        return texto
    return "".join(desempatar(c, m, "L" if t == "L" else "N")
                   for c, m, t in zip(texto, mascaras, padrao))
