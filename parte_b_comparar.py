"""Parte B: aplica a regra do modelo final (registrada em 14/11, ANTES de ver os resultados da GPU). SÓ VALIDAÇÃO.

Regra (CLAUDE.md, log de 14/11): entre os modelos com sinal (B3z partes 0-7, B3z partes 0-17 e, se existir, a versão
com pré-treino de idade), vira final o de maior AUROC na validação. Como as validações de p0-7 (9.549 pacientes) e
p0-17 (~18.700) têm tamanhos diferentes, a comparação justa é nos pacientes de validação EM COMUM (os de p0-7 estão
contidos nos de p0-17, porque o split é por paciente sobre a tabela inteira). Também mostramos cada modelo na
validação completa de p0-17, quando ele tem nota para todos (só o B3z p0-17 tem).

Saída: resultados/parte_b_comparar.csv (AUROC/AUPRC nos pacientes em comum, diferença pareada com IC 95%, bootstrap
de pacientes 1.000, seed 42). Nada é treinado; o teste da Parte B não é aberto.
Uso: .venv\\Scripts\\python parte_b_comparar.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

import ecg_dados as E
from parte_b_cnn import bootstrap

SAIDA = Path(__file__).parent / "resultados"
CANDIDATOS = {"B3z_p0-7": ("parte_b_cnn_p0-7_previsoes_validacao.csv", "B3z"),
              "B3z_p0-17": ("parte_b_cnn_p0-17_previsoes_validacao.csv", "B3z")}
REFERENCIAS = {"B0_p0-17": ("parte_b_cnn_p0-17_previsoes_validacao.csv", "B0")}


def main():
    notas, y = {}, None
    for nome, (arq, col) in {**CANDIDATOS, **REFERENCIAS}.items():
        if not (SAIDA / arq).exists():
            print(f"{nome}: {arq} não existe, fica de fora")
            continue
        d = pd.read_csv(SAIDA / arq).set_index("exam_id")
        notas[nome] = d[col]
        y = d["y"] if y is None else y.combine_first(d["y"])
    comum = sorted(set.intersection(*(set(s.index) for s in notas.values())))
    yc = y.loc[comum].to_numpy().astype(int)
    for k in notas:                                        # o rótulo tem de ser o mesmo nos dois arquivos
        d = pd.read_csv(SAIDA / {**CANDIDATOS, **REFERENCIAS}[k][0]).set_index("exam_id")
        assert (d.loc[comum, "y"].to_numpy() == yc).all(), f"rótulos diferentes em {k}"
    nc = {k: s.loc[comum].to_numpy() for k, s in notas.items()}
    cand = [k for k in CANDIDATOS if k in nc]
    pares = [(k, "B0_p0-17") for k in cand if "B0_p0-17" in nc]
    if {"B3z_p0-17", "B3z_p0-7"} <= set(cand):
        pares.append(("B3z_p0-17", "B3z_p0-7"))
    ic = bootstrap(yc, nc, pares)
    from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc
    linhas = []
    for m in list(nc) + [f"{a} - {b}" for a, b in pares]:
        lin = dict(modelo=m, conjunto="validação em comum", n=len(yc), obitos=int(yc.sum()))
        for met, f in (("auroc", auc), ("auprc", ap)):
            lin[met] = f(yc, nc[m]) if m in nc else f(yc, nc[m.split(" - ")[0]]) - f(yc, nc[m.split(" - ")[1]])
            lin[f"{met}_ic_inf"], lin[f"{met}_ic_sup"] = ic[(met, m)]
        linhas.append(lin)
    res = pd.DataFrame(linhas)
    melhor = max(cand, key=lambda k: auc(yc, nc[k]))
    res["final_pela_regra"] = res["modelo"] == melhor
    res.to_csv(SAIDA / "parte_b_comparar.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))
    print(f"\nPela regra registrada (maior AUROC na validação, pacientes em comum): {melhor}")


if __name__ == "__main__":
    main()
