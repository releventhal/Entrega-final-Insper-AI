r"""Checagem do B4p apontada pelo Claude do Tom (12/11). SÓ TREINO E VALIDAÇÃO; o teste da Parte B não é lido.

O B4p (idade + sexo + as 6 notas do A3 congelado) é ajustado nos pacientes de treino da Parte B (partes 0 a 7).
Os exames das partes 0 a 3 do grupo treino foram usados no treino do próprio A3, então ali as notas do A3 podem ser
mais confiantes do que em pacientes novos. Aqui o B4 e o B4p são reajustados SÓ com os pacientes de treino das
partes 4 a 7 (que o A3 nunca viu) e avaliados nos mesmos 9.549 pacientes de validação, com as mesmas 1.000
reamostragens pareadas por paciente de parte_b_cnn.bootstrap (seed 42).

Uso: .venv\Scripts\python parte_b_b4p_checagem.py   (~2 min; grava resultados/parte_b_b4p_checagem_p0-7.csv)
Resultado de 12/11: B4p 0,8022 -> B4p_47 0,8024 (diferença +0,0003, IC -0,0006 a 0,0011).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import ecg_dados as E
import parte_b_alteracoes as PA
from parte_b_cnn import PARTES_B, bootstrap, carregar, nome_partes


def main():
    tag = nome_partes(PARTES_B)
    el, _ = carregar(PARTES_B, PA.SAIDA)
    for c in E.ROTULOS + ["normal_ecg"]:
        el[c] = el[c].astype(bool)
    el = el.join(PA.notas_a3(el, tag))
    for r in E.ROTULOS:
        p = el[f"p_{r}"].clip(1e-6, 1 - 1e-6)
        el[f"logit_{r}"] = np.log(p / (1 - p))
    tr = el[el.grupo == "treino"]
    va = el[el.grupo == "validação"].sort_values(["parte", "linha"], ignore_index=True)
    print(f"treino da Parte B: {len(tr)} | partes 0-3 (o A3 viu): {int((tr.parte <= 3).sum())} | "
          f"partes 4-7: {int((tr.parte >= 4).sum())} ({int(tr[tr.parte >= 4].y.sum())} óbitos)")
    y = va.y.to_numpy()
    notas, coefs = {}, {}
    for nome, m, t in (("B4", "B4", tr), ("B4p", "B4p", tr), ("B4_47", "B4", tr[tr.parte >= 4]), ("B4p_47", "B4p", tr[tr.parte >= 4])):
        e = StandardScaler().fit(PA.matriz(t, m))
        lr = LogisticRegression(max_iter=2000, random_state=E.SEED).fit(e.transform(PA.matriz(t, m)), t.y)
        notas[nome] = lr.predict_proba(e.transform(PA.matriz(va, m)))[:, 1]
        coefs[nome] = np.round(lr.coef_[0], 3)
    prev = pd.read_csv(PA.SAIDA / f"parte_b_cnn_{tag}_previsoes_validacao.csv").set_index("exam_id")
    for m in ("B0", "B3z"):
        notas[m] = prev.loc[va.exam_id, m].to_numpy()
    pares = [("B4p_47", "B4p"), ("B4_47", "B4"), ("B4p_47", "B4_47"), ("B3z", "B4p_47"), ("B4p_47", "B0")]
    ic = bootstrap(y, notas, pares)
    linhas = []
    for k in list(notas) + [f"{a} - {b}" for a, b in pares]:
        lin = dict(modelo=k, n_val=len(y), obitos_val=int(y.sum()))
        for met, f in (("auroc", roc_auc_score), ("auprc", average_precision_score)):
            lin[met] = f(y, notas[k]) if k in notas else f(y, notas[k.split(" - ")[0]]) - f(y, notas[k.split(" - ")[1]])
            lin[f"{met}_ic_inf"], lin[f"{met}_ic_sup"] = ic[(met, k)]
        linhas.append(lin)
    res = pd.DataFrame(linhas)
    res.to_csv(PA.SAIDA / f"parte_b_b4p_checagem_{tag}.csv", index=False, float_format="%.4f")
    print(res.round(4).to_string(index=False))
    for k, v in coefs.items():
        print(k, "coeficientes (idade, sexo, 6 alterações):", v)


if __name__ == "__main__":
    main()
