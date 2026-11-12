"""Robustez, tarefa 4: subgrupos do A3 (seed 42) na VALIDAÇÃO: sexo e faixa de idade (<40, 40-59, 60-69, 70+, as de analise_erros_A3.py).
AUROC, AUPRC e F1 por alteração, com os limiares do JSON (os mesmos para todos os subgrupos; nenhum é reescolhido), IC 95% por bootstrap
de PACIENTES (1.000 reamostras, seed 42; reamostra os pacientes do subgrupo). Subgrupo com menos de 20 positivos (ou menos de 20 negativos)
= "incerteza grande"; AUROC/AUPRC não definidos sem positivos. Só lê previsões salvas (não carrega modelo).
Saída nova: resultados/robustez_04_subgrupos.csv.
"""
import numpy as np
import pandas as pd

from ecg_dados import ROTULOS
from robustez_comum import FAIXAS, RES, auc_ap, carregar_val, ic, limiares_json, reamostras, salvar_novo

N_BOOT, MIN_POS = 1000, 20


def f1_um(y, s, lim):
    p = s >= lim
    tp, fp, fn = (p & (y == 1)).sum(), (p & (y == 0)).sum(), (~p & (y == 1)).sum()
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan


def main():
    S, Y, age, male, pid = carregar_val()
    lim = limiares_json()
    grupos = [("todos", np.ones(len(S), bool)), ("mulheres", ~male), ("homens", male)]
    grupos += [(f"idade {n}", (age >= lo) & (age < hi)) for lo, hi, n in FAIXAS]
    linhas = []
    for nome, g in grupos:
        ig = np.where(g)[0]
        boots = [ig[b] for b in reamostras(pid[ig], N_BOOT)]  # reamostra pacientes DENTRO do subgrupo; devolve posições em ig
        for k, r in enumerate(ROTULOS):
            y, s = Y[ig, k], S[ig, k]
            au, ap = auc_ap(y, s)
            f1 = f1_um(y, s, lim[r])
            res = np.array([(*auc_ap(Y[i, k], S[i, k]), f1_um(Y[i, k], S[i, k], lim[r])) for i in boots])
            npos = int(y.sum())
            linhas.append(dict(subgrupo=nome, rotulo=r, exames=len(ig), pacientes=len(np.unique(pid[ig])), positivos=npos, prevalencia=y.mean(),
                               auroc=au, auroc_ic_inf=ic(res[:, 0])[0] if npos else np.nan, auroc_ic_sup=ic(res[:, 0])[1] if npos else np.nan,
                               auprc=ap, auprc_ic_inf=ic(res[:, 1])[0] if npos else np.nan, auprc_ic_sup=ic(res[:, 1])[1] if npos else np.nan,
                               f1=f1, f1_ic_inf=ic(res[:, 2])[0] if npos else np.nan, f1_ic_sup=ic(res[:, 2])[1] if npos else np.nan,
                               incerteza_grande=npos < MIN_POS or (len(ig) - npos) < MIN_POS))
    t = pd.DataFrame(linhas)
    salvar_novo(t, RES / "robustez_04_subgrupos.csv")
    pd.set_option("display.width", 250)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
