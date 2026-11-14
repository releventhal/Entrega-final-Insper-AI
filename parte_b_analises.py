"""Parte B: duas leituras práticas dos modelos já treinados. SÓ VALIDAÇÃO, nada é treinado nem reajustado aqui.

Lê as notas de validação já salvas (parte_b_alteracoes_<tag>_previsoes_validacao.csv: B0, B4, B4n, B4p, B3z) e o laudo
(exams.csv). O teste da Parte B não é aberto.

1. "Risco escondido": entre os pacientes de validação SEM nenhuma das 6 alterações no laudo (e, à parte, entre os com
   laudo "ECG normal"), os modelos com ECG ainda separam quem morre em 2 anos melhor que idade + sexo (B0)?
   Se sim, o ECG carrega informação de risco que o laudo não registra. AUROC e AUPRC com IC 95% por bootstrap de
   pacientes (1.000, seed 42; 1 linha = 1 paciente) e diferença pareada contra o B0.
2. Triagem: se uma equipe só pudesse chamar os X% de pacientes com maior risco segundo cada modelo, que fração dos
   óbitos em 2 anos estaria nesse grupo? (X = 5, 10, 20%.) Diferença pareada contra o B0, mesmo bootstrap.
   Empates na nota de corte: entra exatamente ceil(X% * n) pacientes, pela ordem da nota (estável).

Uso: .venv\\Scripts\\python parte_b_analises.py [--partes 0,1,...,7]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import ecg_dados as E
from parte_b_cnn import PARTES_B, nome_partes

RAIZ = Path(__file__).parent
SAIDA = RAIZ / "resultados"
R = E.ROTULOS
MODELOS = ["B0", "B4", "B4p", "B3z"]
FRACOES = [0.05, 0.10, 0.20]
N_BOOT = 1000


def captura(y, s, frac):
    """Fração dos óbitos que estão entre os ceil(frac*n) pacientes de maior nota."""
    k = int(np.ceil(frac * len(y)))
    topo = np.argsort(-s, kind="stable")[:k]
    return y[topo].sum() / y.sum()


def metricas(y, notas):
    r = {}
    for m, s in notas.items():
        r[("auroc", m)], r[("auprc", m)] = roc_auc_score(y, s), average_precision_score(y, s)
        for f in FRACOES:
            r[(f"captura_{int(f * 100)}", m)] = captura(y, s, f)
    return r


def com_ic(y, notas, seed=E.SEED):
    pont = metricas(y, notas)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        boot.append(metricas(y[i], {m: s[i] for m, s in notas.items()}))
    boot = pd.DataFrame(boot)
    linhas = []
    for (met, m), v in pont.items():
        lo, hi = np.percentile(boot[(met, m)], [2.5, 97.5])
        lin = dict(metrica=met, modelo=m, valor=v, ic_inf=lo, ic_sup=hi)
        if m != "B0":                                                 # diferença pareada contra o B0
            d = boot[(met, m)] - boot[(met, "B0")]
            lin.update(dif_b0=v - pont[(met, "B0")], dif_ic_inf=np.percentile(d, 2.5), dif_ic_sup=np.percentile(d, 97.5))
        linhas.append(lin)
    return pd.DataFrame(linhas)


def main(partes):
    tag = nome_partes(partes)
    prev = pd.read_csv(SAIDA / f"parte_b_alteracoes_{tag}_previsoes_validacao.csv")
    laudo = pd.read_csv(RAIZ / "code15" / "exams.csv", usecols=["exam_id", "normal_ecg"] + R).set_index("exam_id")
    d = prev.join(laudo, on="exam_id")
    assert d[R].notna().all().all()
    alguma = d[R].astype(bool).any(axis=1)
    grupos = {"todos": np.ones(len(d), bool), "sem_nenhuma_das_6": ~alguma.to_numpy(),
              "laudo_ecg_normal": d["normal_ecg"].astype(bool).to_numpy()}
    saida = []
    for g, msk in grupos.items():
        y = d.loc[msk, "y"].to_numpy()
        res = com_ic(y, {m: d.loc[msk, m].to_numpy() for m in MODELOS})
        res.insert(0, "grupo", g)
        res["n"], res["obitos"] = int(msk.sum()), int(y.sum())
        saida.append(res)
    saida = pd.concat(saida, ignore_index=True)
    saida["partes"] = tag
    saida.to_csv(SAIDA / f"parte_b_analises_{tag}.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 220)
    print(saida.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--partes", default=",".join(map(str, PARTES_B)))
    main([int(p) for p in ap.parse_args().partes.split(",")])
