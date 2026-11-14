"""Roda os pipelines Tesseract e EasyOCR, compara os resultados com o gabarito e
mede o desempenho de execução (CPU, memória RAM e GPU) de cada engine.

Uso: python comparar.py [numero_inicial numero_final]

Gera:
  results/comparativo.txt     leitura por placa + resumo de acurácia + resumo de desempenho
  results/desempenho.png      gráficos (CPU, RAM, GPU e barras-resumo) por engine
"""
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict

import psutil
import matplotlib
matplotlib.use("Agg")  # sem janela: só salva o PNG
import matplotlib.pyplot as plt

from avaliar import carregar_gabarito, distancia, melhor_casamento

INTERVALO = 0.25  # segundos entre amostras

# nome -> (script executado, arquivo de resultados)
ENGINES = {
    "Tesseract": ("main_tesseract.py", "results/resultados_tesseract.txt"),
    "EasyOCR": ("main_easyocr.py", "results/resultados_easyocr.txt"),
}
RESULTADOS = {nome: arq for nome, (_, arq) in ENGINES.items()}
CORES = {"Tesseract": "#1f77b4", "EasyOCR": "#d62728"}


# --------------------------------------------------------------------------- GPU
def gpu_disponivel():
    """True se houver nvidia-smi respondendo (GPU/CUDA presente na máquina)."""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        return r.returncode == 0 and r.stdout.strip() != ""
    except Exception:
        return False


def ler_gpu():
    """(utilização %, memória usada MB) da GPU, ou None se indisponível.

    Obs.: o nvidia-smi reporta uso global da placa, não por processo.
    """
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            return None
        util, mem = r.stdout.strip().splitlines()[0].split(",")
        return float(util), float(mem)
    except Exception:
        return None


# --------------------------------------------------------------- monitoramento
def rodar_monitorado(script, args, gpu_on):
    """Executa `script` e amostra CPU/RAM/GPU da árvore de processos durante a corrida.

    Retorna (amostras, tempo_total) onde amostras tem listas paralelas
    t/cpu/rss (e gpu/gpumem se gpu_on).
    """
    print(f"== Executando {script} (monitorando) ==")
    ini = time.time()
    proc = subprocess.Popen([sys.executable, script, *args])
    raiz = psutil.Process(proc.pid)

    amostras = {"t": [], "cpu": [], "rss": [], "gpu": [], "gpumem": []}
    stop = threading.Event()

    def arvore():
        procs = [raiz]
        try:
            procs += raiz.children(recursive=True)
        except psutil.NoSuchProcess:
            pass
        return procs

    def amostrar():
        for p in arvore():  # "aquece" cpu_percent (1ª chamada sempre retorna 0)
            try:
                p.cpu_percent(None)
            except psutil.NoSuchProcess:
                pass
        while not stop.is_set():
            time.sleep(INTERVALO)
            cpu = rss = 0.0
            for p in arvore():
                try:
                    cpu += p.cpu_percent(None)
                    rss += p.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            amostras["t"].append(time.time() - ini)
            amostras["cpu"].append(cpu)
            amostras["rss"].append(rss / 1e6)
            if gpu_on:
                g = ler_gpu()
                amostras["gpu"].append(g[0] if g else None)
                amostras["gpumem"].append(g[1] if g else None)

    th = threading.Thread(target=amostrar, daemon=True)
    th.start()
    proc.wait()
    total = time.time() - ini
    stop.set()
    th.join(timeout=1)
    return amostras, total


def resumir(amostras, total):
    cpu, rss = amostras["cpu"], amostras["rss"]
    gpu = [g for g in amostras["gpu"] if g is not None]
    media = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return {
        "tempo": total,
        "cpu_medio": media(cpu),
        "cpu_pico": max(cpu) if cpu else 0.0,
        "ram_media": media(rss),
        "ram_pico": max(rss) if rss else 0.0,
        "gpu_medio": media(gpu) if gpu else None,
        "gpu_pico": max(gpu) if gpu else None,
    }


# ---------------------------------------------------------------------- gráfico
def plotar_desempenho(medidas, gpu_on, caminho="results/desempenho.png"):
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Desempenho de execução: Tesseract OCR vs EasyOCR", fontsize=14)

    # CPU vs tempo
    ax = axs[0][0]
    for nome, (s, _, _) in medidas.items():
        ax.plot(s["t"], s["cpu"], label=nome, color=CORES[nome], lw=1.2)
    ax.set_title("CPU (%) — soma da árvore de processos")
    ax.set_xlabel("tempo (s)")
    ax.set_ylabel("% (1 núcleo = 100)")
    ax.legend()
    ax.grid(alpha=.3)

    # RAM vs tempo
    ax = axs[0][1]
    for nome, (s, _, _) in medidas.items():
        ax.plot(s["t"], s["rss"], label=nome, color=CORES[nome], lw=1.2)
    ax.set_title("Memória RSS (MB)")
    ax.set_xlabel("tempo (s)")
    ax.set_ylabel("MB")
    ax.legend()
    ax.grid(alpha=.3)

    # GPU vs tempo (ou nota CPU-only)
    ax = axs[1][0]
    if gpu_on:
        plotou = False
        for nome, (s, _, _) in medidas.items():
            pts = [(t, g) for t, g in zip(s["t"], s["gpu"]) if g is not None]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        label=nome, color=CORES[nome], lw=1.2)
                plotou = True
        ax.set_title("GPU — utilização global (%)")
        ax.set_xlabel("tempo (s)")
        ax.set_ylabel("%")
        if plotou:
            ax.legend()
        ax.grid(alpha=.3)
    else:
        ax.set_title("GPU — utilização (%)")
        ax.text(.5, .5,
                "CPU-only\n(sem GPU/CUDA disponível\nou EasyOCR com gpu=False)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_axis_off()

    # Barras-resumo: tempo total e pico de RAM
    ax = axs[1][1]
    nomes = list(medidas)
    x = list(range(len(nomes)))
    larg = .35
    tempos = [medidas[n][2]["tempo"] for n in nomes]
    rams = [medidas[n][2]["ram_pico"] for n in nomes]
    b1 = ax.bar([i - larg / 2 for i in x], tempos, larg, color="#7f7f7f", label="tempo (s)")
    ax2 = ax.twinx()
    b2 = ax2.bar([i + larg / 2 for i in x], rams, larg, color="#bcbd22", label="pico RAM (MB)")
    ax.set_xticks(x)
    ax.set_xticklabels(nomes)
    ax.set_title("Resumo por engine: tempo total (cinza) e pico de RAM (oliva)")
    ax.set_ylabel("tempo (s)")
    ax2.set_ylabel("RAM (MB)")
    ax.grid(alpha=.3, axis="y")
    for b in b1:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.1f}s",
                ha="center", va="bottom", fontsize=9)
    for b in b2:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.0f}MB",
                 ha="center", va="bottom", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, .96])
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    return caminho


def resumo_desempenho_txt(medidas, gpu_on):
    linhas = ["", "Desempenho de execução (nesta corrida):"]
    for nome, (_, _, r) in medidas.items():
        g = (f" | GPU médio={r['gpu_medio']:.0f}% pico={r['gpu_pico']:.0f}%"
             if gpu_on and r["gpu_medio"] is not None else "")
        linhas.append(
            f"  {nome:<10} tempo={r['tempo']:.1f}s | "
            f"CPU médio={r['cpu_medio']:.0f}% pico={r['cpu_pico']:.0f}% | "
            f"RAM pico={r['ram_pico']:.0f}MB médio={r['ram_media']:.0f}MB{g}")
    linhas.append("  GPU: " + ("uso global da placa (nvidia-smi)" if gpu_on
                               else "N/A — execução em CPU"))
    linhas.append("  Gráfico: results/desempenho.png")
    return "\n".join(linhas)


# ------------------------------------------------------------------- comparação
def carregar_leituras(arquivo):
    lidas = defaultdict(list)
    for l in open(arquivo, encoding="utf-8").read().splitlines()[1:]:
        p = l.split("\t")
        lidas[p[0].split(".")[0]].append(p[2] if len(p) > 2 else "")
    return lidas


def comparar():
    g = carregar_gabarito()
    leituras = {nome: carregar_leituras(arq) for nome, arq in RESULTADOS.items()}

    linhas = [f"{'img':>3}  {'Tesseract':<12} {'EasyOCR':<12} esperado"]
    stats = {n: {"certas": 0, "total": 0, "chars_ok": 0, "chars": 0} for n in RESULTADOS}

    for idx in sorted(g, key=int):
        # casa cada OCR com o gabarito na mesma ordem (melhor casamento é determinístico
        # para o mesmo gabarito), para alinhar as placas nas duas leituras
        casas = {nome: melhor_casamento(g[idx], leituras[nome].get(idx, []))
                 for nome in RESULTADOS}
        n = max(len(c) for c in casas.values())
        for k in range(n):
            esp = casas["Tesseract"][k][0] if k < len(casas["Tesseract"]) else ""
            cols = [f"{idx:>3}  "]
            for nome in RESULTADOS:
                if k < len(casas[nome]):
                    e, pred = casas[nome][k]
                    esp = esp or e
                else:
                    pred = ""
                s = stats[nome]
                if esp:
                    s["certas"] += pred == esp
                    s["total"] += 1
                    s["chars_ok"] += max(len(esp) - distancia(pred, esp), 0)
                    s["chars"] += len(esp)
                    status = "OK" if pred == esp else "ERR"
                elif pred:  # placa extra detectada
                    s["total"] += 1
                    status = "EXT"
                else:
                    status = "-"
                cols.append(f" {pred or '-':<8} {status:<3}")
            if esp:
                cols.append(f" esperado={esp}")
            linhas.append(" ".join(cols))

    resumo = []
    for nome, s in stats.items():
        t = s["total"] or 1
        resumo.append(f"{nome:<10} Placas exatas: {s['certas']}/{s['total']} "
                      f"({100*s['certas']/t:.0f}%) | Caracteres: {100*s['chars_ok']/max(s['chars'],1):.1f}%")
    return "\n".join(linhas + ["", *resumo])


if __name__ == "__main__":
    gpu_on = gpu_disponivel()
    print(f"GPU: {'disponível (nvidia-smi)' if gpu_on else 'indisponível — execução em CPU'}")

    medidas = {}
    for nome, (script, _) in ENGINES.items():
        amostras, total = rodar_monitorado(script, sys.argv[1:], gpu_on)
        medidas[nome] = (amostras, total, resumir(amostras, total))

    grafico = plotar_desempenho(medidas, gpu_on)

    relatorio = comparar() + "\n" + resumo_desempenho_txt(medidas, gpu_on)
    print("\n" + relatorio)
    with open("results/comparativo.txt", "w", encoding="utf-8") as f:
        f.write(relatorio + "\n")
    print(f"\nRelatório gravado em results/comparativo.txt")
    print(f"Gráfico gravado em {grafico}")
