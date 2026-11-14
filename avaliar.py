"""Compara results/resultados_*.txt com o gabarito.

Aceita várias placas por imagem (chaves repetidas no gabarito). Para cada imagem,
casa as leituras com o gabarito na melhor ordem possível (a ordem das placas não importa).
"""
import itertools
import sys
from collections import defaultdict


def carregar_gabarito(caminho="gabarito.txt"):
    g = defaultdict(list)
    for l in open(caminho, encoding="utf-8"):
        if "->" in l:
            a, b = l.split("->")
            g[a.strip()].append(b.strip())
    return g


def distancia(a, b):
    """Distância de Levenshtein."""
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (ca != cb))
    return d[-1]


def melhor_casamento(esperadas, lidas):
    """Permutação das leituras que minimiza a distância total ('' completa o que faltar)."""
    n = max(len(esperadas), len(lidas))
    esp = esperadas + [""] * (n - len(esperadas))
    melhor = None
    for perm in itertools.permutations(lidas + [""] * (n - len(lidas)), n):
        custo = sum(distancia(p, e) for p, e in zip(perm, esp))
        if melhor is None or custo < melhor[0]:
            melhor = (custo, list(zip(esp, perm)))
    return melhor[1]


def avaliar(arquivo):
    g = carregar_gabarito()
    lidas = defaultdict(list)
    for l in open(arquivo, encoding="utf-8").read().splitlines()[1:]:
        p = l.split("\t")
        lidas[p[0].split(".")[0]].append(p[2] if len(p) > 2 else "")
    certas = total = chars_ok = total_chars = 0
    saida = []
    for idx in sorted(g, key=int):
        for esp, pred in melhor_casamento(g[idx], lidas.get(idx, [])):
            if not esp:  # leitura sobrando (placa detectada que não está no gabarito)
                saida.append(f"{idx:>2} EXTRA lido={pred}")
                total += 1
                continue
            ok = pred == esp
            certas += ok
            total += 1
            total_chars += len(esp)
            chars_ok += max(len(esp) - distancia(pred, esp), 0)
            saida.append(f"{idx:>2} {'OK ' if ok else 'ERR'} esperado={esp} lido={pred}")
    return certas, total, chars_ok / total_chars, saida


if __name__ == "__main__":
    arq = sys.argv[1] if len(sys.argv) > 1 else "results/resultados_tesseract.txt"
    c, n, cc, linhas = avaliar(arq)
    cab = f"Placas exatas: {c}/{n} ({100*c/n:.0f}%) | Caracteres corretos: {100*cc:.1f}%"
    relatorio = cab + "\n" + "\n".join(linhas)
    print(relatorio)
    open(arq.replace("resultados_", "avaliacao_"), "w", encoding="utf-8").write(relatorio + "\n")
