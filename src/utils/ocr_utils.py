import re

PARA_LETRA = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B", "4": "A", "7": "T", "9": "Q"}
PARA_NUMERO = {"O": "0", "Q": "0", "D": "0", "G": "0", "U": "0",
               "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "A": "4", "T": "7"}

ALFABETO = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
PADRAO = "LLLNLNN"  # formato Mercosul: ABC1D23 (L = letra, N = número)

def corrigir_padrao(texto):
    """Se tem 7 caracteres, força letra/número conforme a posição do padrão."""
    texto = re.sub(r"[^A-Z0-9]", "", texto.upper())
    if len(texto) != 7:
        return texto
    return "".join(
        (PARA_LETRA.get(c, c) if c.isdigit() else c) if t == "L"
        else (PARA_NUMERO.get(c, c) if c.isalpha() else c)
        for c, t in zip(texto, PADRAO))
