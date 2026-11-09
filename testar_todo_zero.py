"""Teste da regra de exclusão (só arrays sintéticos; não lê nenhum dado real). Uso: python testar_todo_zero.py"""
import numpy as np

from avaliar_externo import mascara_todo_zero
from ecg_dados import n_deriv_zeradas, todo_zero

rng = np.random.default_rng(0)
normal = rng.normal(size=(4096, 12)).astype(np.float32)
z12 = np.zeros((4096, 12), np.float32)
z11 = normal.copy(); z11[:, :11] = 0           # 11 derivações zeradas, 1 com sinal
z1 = normal.copy(); z1[:, 4] = 0               # 1 zerada
const = np.ones((4096, 12), np.float32)        # constante não-zero: NÃO é todo-zero
X = np.stack([z12, z11, normal, z1, const])
assert list(n_deriv_zeradas(X)) == [12, 11, 0, 1, 0]
assert list(todo_zero(X)) == [True, False, False, False, False]


class Falso:  # imita um gerador em lotes de 2: len() e [i]
    def __len__(self): return 3
    def __getitem__(self, i): return X[2 * i:2 * i + 2]


assert list(mascara_todo_zero(Falso())) == [True, False, False, False, False]
print("OK: exclui só o exame com 12 derivações zeradas (11, 1 e constante ficam)")
