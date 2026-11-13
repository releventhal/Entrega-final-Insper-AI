"""Diagnóstico (13/11): A3 congelado por fração de preenchimento (amostras com as 12 derivações zeradas / 4.096) na VALIDAÇÃO. Descritivo, sem IC, sem causa.
Os quartis colapsam (p25 = 0; p50 = p75 = 0,284), então os grupos são os naturais: ==0, de 0 a 0,2837 e acima. Usa as notas salvas do A3 (nenhum predict) e lê o sinal cru só para contar zeros.
Saída: resultados/diagnostico_preenchimento.csv (aborta se já existe). Uso (na raiz, com o .venv): python diagnostico_preenchimento.py
"""
import sys
from pathlib import Path
RES = Path(__file__).resolve().parent / "resultados"
assert not (RES / "diagnostico_preenchimento.csv").exists(), "já existe, não sobrescrevo"
import numpy as np, pandas as pd
from a1_cnn import carregar_indices
from ecg_dados import GeradorECG, ROTULOS
from robustez_comum import carregar_val, limiares_json, auc_ap
S, Y, age, male, pid = carregar_val()
lim = np.array([limiares_json()[r] for r in ROTULOS])
_, va, _ = carregar_indices(partes=[0, 1, 2, 3])
g = GeradorECG(va, normalizar=False)
assert (g.info.patient_id.to_numpy() == pid).all(), "ordem difere"
fr = []
for i in range(len(g)):
    X, _ = g[i]
    ativo = (np.abs(X).max(axis=2) > 0)               # (n, 4096): amostra com alguma derivação != 0
    fr.append(1 - ativo.mean(1))
fr = np.concatenate(fr)
print("n", len(fr), "| fração de zeros: min %.3f p25 %.3f mediana %.3f p75 %.3f máx %.3f" % tuple(np.percentile(fr, [0, 25, 50, 75, 100])))
print("valores mais comuns:\n", pd.Series(fr.round(3)).value_counts().head(6).to_string())
q = np.unique(np.quantile(fr, [.25, .5, .75]))
print("cortes dos quartis:", q.round(4))
grp = np.searchsorted(q, fr, side="left")  # empates ficam juntos
print("tamanho dos grupos:", np.bincount(grp).tolist())
pd.set_option("display.width", 250)
rows = []
for gi in range(grp.max() + 1):
    m = grp == gi
    for k, r in enumerate(ROTULOS):
        y, s = Y[m, k], S[m, k]
        a, p = auc_ap(y, s)
        rows.append(dict(grupo=gi + 1, zeros_min=fr[m].min().round(3), zeros_max=fr[m].max().round(3), n=int(m.sum()), alteracao=r, n_pos=int(y.sum()),
                         prevalencia=y.mean(), auroc=a, auprc=p, pos_acima_limiar=(s[y == 1] >= lim[k]).mean() if y.sum() else np.nan))
T = pd.DataFrame(rows)
T.to_csv(RES / "diagnostico_preenchimento.csv", index=False)
for r in ROTULOS:
    print("\n", r); print(T[T.alteracao == r].drop(columns="alteracao").round(3).to_string(index=False))
