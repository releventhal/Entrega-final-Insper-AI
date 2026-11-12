"""Fase 2 de robustez (12/11): funções comuns. SÓ validação do A3; o teste (interno, PTB-XL, CODE-TEST) nunca é lido.
Nada aqui altera modelo, limiares, split ou JSON. Toda saída vai para arquivos NOVOS com prefixo robustez_.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ecg_dados import ROTULOS

RAIZ = Path(__file__).parent
RES, FIG, MODELOS = RAIZ / "resultados", RAIZ / "figuras", RAIZ / "modelos"
SHA_A3 = "bae64fa994934072434853ce6c5f6dfffd485a1f126336be1d6e8d526ca12a89"  # congelado_candidato_A3.json
FAIXAS = [(0, 40, "<40"), (40, 60, "40-59"), (60, 70, "60-69"), (70, 200, "70+")]  # as mesmas de analise_erros_A3.py


def salvar_novo(df_ou_fig, caminho, **kw):
    """Nunca sobrescreve: se o arquivo já existe, aborta."""
    caminho = Path(caminho)
    assert not caminho.exists(), f"já existe, não sobrescrevo: {caminho}"
    if hasattr(df_ou_fig, "to_csv"):
        df_ou_fig.to_csv(caminho, index=False, float_format="%.6g", **kw)
    else:
        df_ou_fig.savefig(caminho, dpi=130, bbox_inches="tight")


def sha256(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def conferir_modelo():
    """Antes de usar A3_melhor.keras: SHA256 tem que ser o do JSON congelado."""
    s = sha256(MODELOS / "A3_melhor.keras")
    assert s == SHA_A3, f"SHA256 do A3 diferente: {s}"
    return s


def limiares_json():
    return json.load(open(RAIZ / "congelado_candidato_A3.json", encoding="utf-8"))["limiares"]["valores"]


def carregar_val():
    """Validação do A3 (8.158 exames, ordem (parte, linha)): S, Y, age, is_male, patient_id.
    Vem de resultados/erros_A3/previsoes_e_rotulos.npy (gerado por analise_erros_A3.py; recalculado do modelo)."""
    a = np.load(RES / "erros_A3" / "previsoes_e_rotulos.npy")
    S, Y, age, male, pid = a[:, :6], a[:, 6:12].astype(int), a[:, 12], a[:, 13].astype(bool), a[:, 14].astype(np.int64)
    salvo = np.load(RES / "A3_previsoes_validacao.npy")
    assert np.abs(S - salvo).max() < 1e-5, "previsões de erros_A3 diferem das salvas do A3"
    return S, Y, age, male, pid


def reamostras(pid, n=1000, seed=42):
    """n reamostras de PACIENTES (com todos os seus exames); devolve lista de arrays de índices de exame."""
    rng = np.random.default_rng(seed)
    up, inv = np.unique(pid, return_inverse=True)
    grupos = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    return [np.concatenate([grupos[j] for j in rng.integers(0, len(up), len(up))]) for _ in range(n)]


def auc_ap(y, s):
    """(AUROC, AUPRC); NaN se a amostra não tem os dois lados."""
    if y.min() == y.max():
        return np.nan, np.nan
    return roc_auc_score(y, s), average_precision_score(y, s)


def macro_auc_ap(Y, S, idx=None):
    """vetores (6,) de AUROC e AUPRC por alteração, nos exames `idx` (todos se None)."""
    if idx is not None:
        Y, S = Y[idx], S[idx]
    r = np.array([auc_ap(Y[:, k], S[:, k]) for k in range(Y.shape[1])])
    return r[:, 0], r[:, 1]


def ic(v, nivel=95):
    """Percentis 2,5 e 97,5 ignorando NaN."""
    return np.nanpercentile(v, [(100 - nivel) / 2, 100 - (100 - nivel) / 2])
