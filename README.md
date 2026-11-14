# Segmentação de placas + OCR (Tesseract e EasyOCR)

Pipeline simples: **segmentação → pré-processamento → OCR (Tesseract ou EasyOCR)**.
Os dois OCRs reaproveitam as mesmas etapas de segmentação e pré-processamento.

**Documentação detalhada da implementação:** [`docs/implementacao.md`](docs/implementacao.md)

## Estrutura
```
dataset/                            imagens de entrada
src/segmentacao.py                  detecta TODAS as faixas azuis (HSV) e retifica cada placa (perspectiva)
src/pipeline.py                     laço comum (uma linha de resultado por placa detectada)
src/preprocessamento.py             recorte, normalização de iluminação, CLAHE, Otsu, filtro de componentes
src/ocr_tesseract.py                Tesseract (psm 7/13, whitelist A-Z/0-9), voto e correção de padrão
src/ocr_easyocr.py                  EasyOCR (recognize na linha, allowlist A-Z/0-9), voto e correção de padrão
src/geometria.py                    desempate de pares parecidos (I/L, Y/V, Q/O, 7/2, 4/1) pela forma do glifo
main_easyocr.py                     pipeline completo com EasyOCR -> results/resultados_easyocr.txt
main_tesseract.py                   executa tudo e grava os resultados
comparar.py                         roda os dois pipelines e gera o comparativo -> results/comparativo.txt
results/comparativo.txt             leitura por placa + resumo de acurácia + desempenho
results/desempenho.png              gráficos de CPU/RAM/GPU e barras-resumo por engine
results/resultados_tesseract.txt    arquivo, texto_bruto, texto_corrigido
results/debug/                      placa recortada, binarizada e detecção por imagem
results/avaliacao_tesseract.txt     comparação com o gabarito
gabarito.txt / avaliar.py           resultado esperado e script de avaliação
```

## Como rodar localmente

Pré-requisitos: Python 3.10+ e o binário do Tesseract instalado no sistema.

1. Instale o Tesseract OCR (necessário apenas para o pipeline `main_tesseract.py`):
   ```bash
   # Linux (Debian/Ubuntu)
   sudo apt install tesseract-ocr
   # macOS
   brew install tesseract
   # Windows: instalador em https://github.com/UB-Mannheim/tesseract/wiki
   ```
2. Crie e ative um ambiente virtual (opcional, recomendado):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
   O `easyocr` puxa o PyTorch (CPU), então o primeiro download é demorado.
4. Rode os pipelines:
   ```bash
   python main_tesseract.py     # -> results/resultados_tesseract.txt + results/debug/
   python main_easyocr.py       # -> results/resultados_easyocr.txt
   python avaliar.py            # compara com gabarito.txt -> results/avaliacao_tesseract.txt
   python comparar.py           # roda os dois OCRs e compara -> results/comparativo.txt
   ```
   Os pipelines e o `comparar.py` aceitam um intervalo de imagens: `python main_tesseract.py 1 5`.

## Como a detecção de placas funciona

A detecção é a mesma para os dois OCRs; só a etapa final de leitura muda.

### 1. Segmentação (`src/segmentacao.py`)
- A faixa azul superior ("BRASIL") da placa Mercosul é a característica mais distintiva.
  Ela é segmentada por cor em **HSV** (azul ~ H 100-130), incluindo a bandeira
  (verde/amarelo) para não interromper a faixa.
- Morfologia (CLOSE 25x3 + OPEN 3x3) fecha os buracos do texto "BRASIL" e das estrelas.
- Cada contorno é avaliado pelo `minAreaRect`: só passam faixas muito alongadas
  (razão 12-40) e bem preenchidas (> 80% do retângulo), o que descarta carrocerias azuis.
- Placas menores que 15% da largura da imagem ou que 40% da maior faixa são descartadas
  (carros distantes); as demais ainda precisam ter o corpo claro abaixo da faixa.
- Para cada faixa aceita, estima-se o retângulo da placa inteira (altura ≈ 5x a
  espessura da faixa) e retifica-se rotação/perspectiva com `warpPerspective`,
  gerando um recorte de 620 px de largura.

### 2. Pré-processamento (`src/preprocessamento.py`)
- Recorte da ROI dos caracteres, normalização de iluminação e CLAHE.
- 4 binarizações: Otsu com CLAHE e Otsu após blur (kernel 7, 11 e 15).
- Componentes conectados com fusão de pedaços empilhados (ex.: dígitos quebrados).

### 3a. OCR com Tesseract (`src/ocr_tesseract.py`)
- Cada binarização é lida com `psm 7` (linha única) e `psm 13` (linha bruta),
  restringindo a whitelist a `A-Z0-9`.
- As leituras são combinadas por **voto por posição** e corrigidas pelo padrão
  Mercosul `LLLNLNN`; o desempate `I`/`L` usa a largura da serifa no topo do glifo.

### 3b. OCR com EasyOCR (`src/ocr_easyocr.py`)
- O EasyOCR lê a linha já recortada com `Reader.recognize` (sem rodar o detector
  CRAFT), com `allowlist` `A-Z0-9`, nas mesmas 4 binarizações.
- Depois aplicam-se o mesmo voto por posição, a correção `LLLNLNN` e o desempate
  geométrico (`src/geometria.py`) para pares parecidos (I/L, Y/V, Q/O, 7/2, 4/1).

## Resultado (20 imagens)
Veja `results/avaliacao_tesseract.txt`: 16/21 placas exatas (76%), 96,6% dos caracteres (21 placas em 20 imagens).
Erros restantes: `9`->`0`, `J`->`I`, `O`->`Q`, `G`->`E`, `1`->`4` (glifos parecidos com a marca d'água).

Para reavaliar: `python main_tesseract.py && python avaliar.py`

## Comparação Tesseract x EasyOCR
Mesma segmentação e mesmo pré-processamento; muda só o OCR (veja a seção
"Como a detecção de placas funciona" acima). Para gerar o comparativo placa a placa:
`python comparar.py` (grava em `results/comparativo.txt`).

O `comparar.py` também mede o desempenho de execução de cada engine (CPU % da árvore de
processos, memória RSS e utilização da GPU via `nvidia-smi`, amostrados a cada 0,25 s) e
grava os gráficos em `results/desempenho.png`; o resumo numérico vai para o
`results/comparativo.txt`. Obs.: a GPU mostrada é o uso global da placa; neste ambiente o
EasyOCR roda com `gpu=False` (CPU-only), então o painel de GPU é apenas informativo.
Dependências extras: `psutil` e `matplotlib` (já no `requirements.txt`).

| Etapa (EasyOCR)                    | Placas exatas | Caracteres |
|------------------------------------|---------------|------------|
| Voto entre 4 binarizações (medido com 20 placas, antes da 2ª placa da imagem 2) | 13/20 (65%) | 95,0% |
| + desempate geométrico (final)     | 21/21 (100%)  | 100%       |
| Tesseract (final)                  | 16/21 (76%)   | 96,6%      |

Atenção: os limiares do desempate geométrico foram calibrados nestas mesmas 20 imagens, então 100% é
um resultado de calibração, não uma estimativa de desempenho em imagens novas. O desempate (`src/geometria.py`)
ainda não está ligado ao pipeline do Tesseract, por isso a comparação acima não é totalmente justa.

## Várias placas por imagem
A imagem 2 tem duas placas legíveis. A segmentação lista todas as faixas azuis e descarta as muito pequenas
(menos de 15% da largura da imagem ou menos de 40% da maior faixa), que são placas de carros distantes
(as de `16` e `19`). Cada placa gera uma linha no `.txt`, da maior para a menor. O avaliador casa as leituras
com o gabarito em qualquer ordem. Esses limiares também foram calibrados neste dataset.
