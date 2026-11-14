"""Pipeline completo: segmentação -> pré-processamento -> EasyOCR.

Uso: python main_easyocr.py [numero_inicial numero_final]
"""
import warnings

warnings.filterwarnings("ignore")  # ruídos do PyTorch em CPU

from src.ocr_easyocr import ler_placa  # noqa: E402
from src.pipeline import processar_dataset  # noqa: E402

if __name__ == "__main__":
    processar_dataset(ler_placa, "resultados_easyocr.txt")
