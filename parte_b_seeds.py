"""Parte B: estabilidade do B3z entre seeds. SÓ VALIDAÇÃO; nada é treinado aqui; o teste da Parte B não é aberto.

Por que (proposta de 15/11, enviada ao Tom): o B3z das 18 partes rodou com uma seed só (42), e a diferença contra o
B4p (+0,013) é pequena. As seeds 43 e 44 (`parte_b_cnn.py treinar_zero_idade --seed N`, só muda a seed dos pesos
iniciais e do embaralhamento) servem SÓ para relatar a estabilidade. Elas NÃO escolhem nem trocam o modelo final
(o B3z p0-17 seed 42, escolhido pela regra de 14/11).

Lê: parte_b_alteracoes_<tag>_previsoes_validacao.csv (B0, B4, B4p e o B3z seed 42) e
    parte_b_B3z_<tag>_s<seed>_previsoes_validacao.npz (as outras seeds).
Grava: resultados/parte_b_seeds_<tag>.csv: por seed, AUROC/AUPRC e as diferenças pareadas contra B0, B4 e B4p
       (IC 95%, bootstrap de pacientes 1.000, seed 42, as mesmas reamostragens para todos), mais média e desvio.
Uso: .venv\\Scripts\\python parte_b_seeds.py [--partes 0,...,17] [--seeds 43,44]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import ecg_dados as E
from parte_b_cnn import bootstrap, nome_partes

SAIDA = Path(__file__).parent / "resultados"


def main(partes, seeds):
    tag = nome_partes(partes)
    d = pd.read_csv(SAIDA / f"parte_b_alteracoes_{tag}_previsoes_validacao.csv").set_index("exam_id")
    y = d["y"].to_numpy().astype(int)
    notas = {"B0": d["B0"].to_numpy(), "B4": d["B4"].to_numpy(), "B4p": d["B4p"].to_numpy(),
             "B3z_s42": d["B3z"].to_numpy()}
    for s in seeds:
        arq = SAIDA / f"parte_b_B3z_{tag}_s{s}_previsoes_validacao.npz"
        if not arq.exists():
            print(f"seed {s}: {arq.name} não existe, fica de fora")
            continue
        z = np.load(arq)
        sr = pd.Series(z["S"], index=z["exam_id"])
        assert set(sr.index) == set(d.index), f"seed {s}: pacientes de validação diferentes"
        notas[f"B3z_s{s}"] = sr.loc[d.index].to_numpy()
    b3z = [k for k in notas if k.startswith("B3z")]
    pares = [(k, r) for k in b3z for r in ("B0", "B4", "B4p")]
    ic = bootstrap(y, notas, pares)
    met = {"auroc": roc_auc_score, "auprc": average_precision_score}
    linhas = []
    for m in list(notas) + [f"{a} - {b}" for a, b in pares]:
        lin = dict(modelo=m, n_val=len(y), obitos_val=int(y.sum()))
        for k, f in met.items():
            lin[k] = f(y, notas[m]) if m in notas else f(y, notas[m.split(" - ")[0]]) - f(y, notas[m.split(" - ")[1]])
            lin[f"{k}_ic_inf"], lin[f"{k}_ic_sup"] = ic[(k, m)]
        linhas.append(lin)
    res = pd.DataFrame(linhas)
    # média e desvio (ddof=1) entre as seeds, sem IC: só descreve a variação
    for alvo in ["B3z"] + [f"B3z - {r}" for r in ("B0", "B4", "B4p")]:
        sel = res[res["modelo"].isin([alvo.replace("B3z", k) for k in b3z])]
        if len(sel) > 1:
            linhas_m = dict(modelo=f"{alvo} (média de {len(sel)} seeds)", n_val=len(y), obitos_val=int(y.sum()))
            for k in met:
                linhas_m[k], linhas_m[f"{k}_desvio"] = sel[k].mean(), sel[k].std(ddof=1)
            res = pd.concat([res, pd.DataFrame([linhas_m])], ignore_index=True)
    res["partes"] = tag
    res.to_csv(SAIDA / f"parte_b_seeds_{tag}.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 220)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--partes", default=",".join(map(str, range(18))))
    ap.add_argument("--seeds", default="43,44")
    a = ap.parse_args()
    main([int(p) for p in a.partes.split(",")], [int(s) for s in a.seeds.split(",")])
