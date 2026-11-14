"""Sessão sem GPU (14/11): SÓ se o A5 for cortado por tempo (sem early stopping). Previsões na VALIDAÇÃO do melhor checkpoint
(modelos/A5_completo_melhor.keras = menor perda de validação até o corte). carregar_indices corta o grupo teste.
Nome novo A5_completo_parcial14 (não sobrescreve o parcial de 13/11 nem engana a retomada, que procura A5_completo_previsoes_validacao.npy).
Depois: python noite_fila.py avaliar --nome A5_completo_parcial14
"""
import json
import sys
from pathlib import Path

import keras
import numpy as np

import a1_cnn
from ecg_dados import GeradorECG

RAIZ = Path(__file__).resolve().parent
NOME = "A5_completo_parcial14"
saida = RAIZ / "resultados" / f"{NOME}_previsoes_validacao.npy"
assert not saida.exists()
_, va, _ = a1_cnn.carregar_indices(partes=[0, 1, 2, 3])
m = keras.models.load_model(RAIZ / "modelos" / "A5_completo_melhor.keras", compile=False)
S = m.predict(GeradorECG(va, normalizar=True), verbose=0)
assert S.shape == (len(va), 6)
np.save(saida, S.astype(np.float32))
est = json.loads((RAIZ / "logs" / "noite_A5_completo_estado.json").read_text(encoding="utf-8"))
h = est["historia"]
melhor = 1 + int(np.argmin([x["val_loss"] for x in h]))
est["termino"] = f"CORTADO POR TEMPO na sessão de 14/11 ({len(h)} épocas fechadas; melhor checkpoint = época {melhor}; espera {est['es_wait']}/3): PARCIAL, NÃO CONCLUSIVO"
(RAIZ / "logs" / f"noite_{NOME}_estado.json").write_text(json.dumps(est, ensure_ascii=False), encoding="utf-8")
print("previsões salvas:", saida, S.shape, est["termino"], file=sys.stderr)
