"""Avaliação PARCIAL, NÃO CONCLUSIVA (13/11): previsões na VALIDAÇÃO do melhor checkpoint (menor perda de validação) do A5 interrompido pelo usuário. Só validação (carregar_indices corta o teste).
Grava resultados/A5_completo_parcial_previsoes_validacao.npy (nome diferente do de um treino completo, para a retomada não achar que o A5 acabou) e copia o estado do treino
com o rótulo de término; depois rode `python noite_fila.py avaliar --nome A5_completo_parcial`. Não altera nada rastreado.
"""
import json
from pathlib import Path

import keras
import numpy as np

import a1_cnn
from ecg_dados import GeradorECG

RAIZ = Path(__file__).resolve().parent
_, va, _ = a1_cnn.carregar_indices(partes=[0, 1, 2, 3])
g = GeradorECG(va, normalizar=True)
m = keras.models.load_model(RAIZ / "modelos" / "A5_completo_melhor.keras", compile=False)
S = m.predict(g, verbose=0)
assert S.shape == (len(va), 6)
saida = RAIZ / "resultados" / "A5_completo_parcial_previsoes_validacao.npy"
assert not saida.exists()
np.save(saida, S.astype(np.float32))
est = json.loads((RAIZ / "logs" / "noite_A5_completo_estado.json").read_text(encoding="utf-8"))
est["termino"] = "INTERROMPIDO PELO USUÁRIO na época 4 (3 épocas fechadas; melhor checkpoint = época 3): PARCIAL, NÃO CONCLUSIVO"
(RAIZ / "logs" / "noite_A5_completo_parcial_estado.json").write_text(json.dumps(est, ensure_ascii=False), encoding="utf-8")
print("previsões salvas:", saida, S.shape)
