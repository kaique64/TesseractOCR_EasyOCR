"""Pipeline completo: segmentação -> pré-processamento -> Tesseract OCR.

Uso: python main_tesseract.py [numero_inicial numero_final]
"""
from src.ocr_tesseract import ler_placa
from src.pipeline import processar_dataset

if __name__ == "__main__":
    processar_dataset(ler_placa, "resultados_tesseract.txt")
