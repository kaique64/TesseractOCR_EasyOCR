"""Laço comum aos dois pipelines: segmentação -> pré-processamento -> OCR (função recebida)."""
import os
import sys

import cv2

from src.preprocessamento import binarizar, isolar_caracteres, normalizar
from src.segmentacao import segmentar_placas

DATASET = "dataset"
SAIDA = "results"
DEBUG = os.path.join(SAIDA, "debug")


def chave_numerica(nome):
    base = os.path.splitext(nome)[0]
    return (0, int(base)) if base.isdigit() else (1, base)


def processar_dataset(ler_placa, nome_saida):
    """Roda o pipeline em todas as imagens (ou num intervalo: `script.py ini fim`).

    Cada placa detectada gera uma linha `arquivo<TAB>leituras_brutas<TAB>texto_final`.
    Imagens com mais de uma placa geram uma linha por placa (da maior para a menor).
    """
    os.makedirs(DEBUG, exist_ok=True)
    arquivos = sorted((f for f in os.listdir(DATASET)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))), key=chave_numerica)
    intervalo = len(sys.argv) == 3
    if intervalo:
        ini, fim = int(sys.argv[1]), int(sys.argv[2])
        arquivos = [f for f in arquivos if ini <= int(os.path.splitext(f)[0]) <= fim]
    linhas = []

    for nome in arquivos:
        img = cv2.imread(os.path.join(DATASET, nome))
        placas = segmentar_placas(img)
        base = os.path.splitext(nome)[0]

        if not placas:
            print(f"{nome}: placa não encontrada")
            linhas.append(f"{nome}\tNAO_ENCONTRADA\tNAO_ENCONTRADA")
            continue

        marcada = img.copy()
        for k, (placa, caixa) in enumerate(placas, start=1):
            cinza = normalizar(placa)
            brutas, final = ler_placa(cinza)
            sufixo = f"_p{k}" if len(placas) > 1 else ""
            print(f"{nome}{' (placa ' + str(k) + ')' if sufixo else ''}: {final}")
            linhas.append(f"{nome}\t{'|'.join(brutas)}\t{final}")

            # Imagens intermediárias para análise/relatório
            cv2.imwrite(os.path.join(DEBUG, f"{base}{sufixo}_placa.png"), placa)
            linha, _ = isolar_caracteres(binarizar(cinza, "blur7"))
            cv2.imwrite(os.path.join(DEBUG, f"{base}{sufixo}_binaria.png"), linha)
            cv2.polylines(marcada, [caixa.reshape(-1, 1, 2)], True, (0, 255, 0), 3)
        cv2.imwrite(os.path.join(DEBUG, f"{base}_deteccao.jpg"), marcada)

    destino = os.path.join(SAIDA, nome_saida)
    cabecalho = "arquivo\tleituras_brutas(separadas por |)\ttexto_final\n"
    if intervalo and os.path.exists(destino):  # intervalo: mescla com o que já existe
        novos = {l.split("\t")[0] for l in linhas}
        antigas = [l for l in open(destino, encoding="utf-8").read().splitlines()[1:]
                   if l.split("\t")[0] not in novos]
        linhas = sorted(antigas + linhas, key=lambda l: chave_numerica(l.split("\t")[0]))
    with open(destino, "w", encoding="utf-8") as f:
        f.write(cabecalho + "\n".join(linhas) + "\n")
