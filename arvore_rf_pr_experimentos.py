"""A0e: acrescenta ao resultados/experimentos.csv as linhas do A0e-1 e do A0e-2 (árvore e floresta; "só ECG" e "ECG + idade + sexo"), no mesmo formato do arvore_rf.py
(a1_cnn.tabela_metricas + gravar_no_csv; só ACRESCENTA: troca apenas linhas de experimento com o mesmo nome). Não treina nada: usa as previsões de validação já salvas por
arvore_rf_pr.py (resultados/arvore_rf_pr_previsoes_validacao.npz), os hiperparâmetros escolhidos em resultados/arvore_rf_pr_grade.csv (maior AUPRC macro, como no treino)
e o limiar por alteração = F1 máximo na validação (a0_baselines.melhor_limiar_f1). Só validação; o teste não é aberto.
Uso (na raiz, com o .venv): python arvore_rf_pr_experimentos.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from a0_baselines import melhor_limiar_f1
from a1_cnn import gravar_no_csv, tabela_metricas
from ecg_dados import ROTULOS

RES = Path(__file__).resolve().parent / "resultados"
FAM = {"arvore": ("árvore", "árvore de decisão"), "rf": ("RF", "random forest")}
EXTRA = {"A0e-1": "p_r_peak", "A0e-2": "p_r_peak + qrs_dwt + qt_dwt"}


def main():
    va = pd.read_csv(RES / "atributos_validacao.csv.gz", usecols=["exam_id", "patient_id"] + ROTULOS)
    tr_n = len(pd.read_csv(RES / "atributos_treino.csv.gz", usecols=["exam_id"]))
    P = np.load(RES / "arvore_rf_pr_previsoes_validacao.npz")
    assert (P["exam_id"] == va.exam_id.to_numpy()).all()
    Y = va[ROTULOS].to_numpy(int)
    g = pd.read_csv(RES / "arvore_rf_pr_grade.csv")
    existentes = set(pd.read_csv(RES / "experimentos.csv").experimento)
    partes = []
    for var in EXTRA:
        for fam, (nome_p, nome_exp) in FAM.items():
            for versao in ("ecg", "ecg+idade+sexo"):
                sub = g[(g.variante == var) & (g.familia == fam) & (g.atributos == versao)]
                cfg = sub.loc[sub.auprc_macro.idxmax()]
                hp = {"max_depth": int(cfg.max_depth)} if fam == "arvore" else {"n_estimators": int(cfg.n_estimators), "min_samples_leaf": int(cfg.min_samples_leaf)}
                chave = f"{var}_{nome_p}" + ("" if versao == "ecg" else "_+idade+sexo")
                S = P[chave]
                nome = f"{var} {nome_exp} (atributos do sinal + {EXTRA[var]}{'' if versao == 'ecg' else ' + idade + sexo'})"
                assert nome not in existentes, f"já existe: {nome}"
                lims = {r: melhor_limiar_f1(Y[:, k], S[:, k]) for k, r in enumerate(ROTULOS)}
                df = tabela_metricas(nome, Y, S, lims)
                df["dados"] = f"partes [0, 1, 2, 3] (provisório); atributos: {versao}; {hp}"
                df["n_treino"] = tr_n
                partes.append(df)
                m = df.iloc[-1]
                print(f"{nome} | {hp} | MACRO AUROC {m.auroc:.4f} AUPRC {m.auprc:.4f} F1 {m.f1:.4f}")
    gravar_no_csv(pd.concat(partes, ignore_index=True))
    print("ok: linhas acrescentadas a resultados/experimentos.csv")


if __name__ == "__main__":
    main()
