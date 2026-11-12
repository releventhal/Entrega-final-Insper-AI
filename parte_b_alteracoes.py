"""Parte B: as 6 alterações do ECG ajudam a prever o óbito em 2 anos além da idade e do sexo? SÓ TREINO E VALIDAÇÃO.

Pergunta dos diretores na intermediária (11/11): "a arritmia dá certeza para prever a morte?". Também serve para
entender o B3z: se um modelo com idade + sexo + as 6 alterações chega perto do B3z, a CNN está basicamente
reconhecendo as alterações conhecidas; se fica perto do B0, a CNN vê no ECG algo além delas.

Modelos (regressão logística, ajustada no treino; avaliados nos MESMOS 9.549 pacientes de validação do parte_b_cnn.py):
  B0   idade + sexo                                              (reproduz o B0 de parte_b_cnn_p0-7.csv)
  B4   B0 + as 6 alterações do LAUDO (colunas do exams.csv, 0/1)
  B4n  B4 + `normal_ecg` do laudo (o laudo disse "ECG normal"?)  -> qualquer anormalidade, não só as 6
  B4p  B0 + as 6 notas do A3 CONGELADO (logit da probabilidade). As notas vêm das 128 características do A3 já salvas
       (parte_b_caracteristicas_A3_p0-7.npz) passadas pela camada final do próprio A3: é o A3 inteiro, sem retreinar.
  B3z  lido de parte_b_cnn_p0-7_previsoes_validacao.csv (CNN do zero + idade + sexo; nada é retreinado aqui)
Uma mudança por vez: B0 -> B4 (entram as alterações), B4 -> B4n (entra "ECG normal"), B4 -> B4p (laudo -> A3).

Também: razão de chances (odds ratio) de cada alteração no B4, ajustada por idade e sexo, ajustada SEM penalização
no treino, com IC 95% por bootstrap de pacientes do treino. Idade em décadas.

Mesmos dados do parte_b_cnn.py (partes 0 a 7, primeiro exame de cada paciente, regra do horizonte de 2 anos,
exames todo-zero fora pelo sinal, grupo teste cortado na carga e nunca lido).
Uso: .venv\\Scripts\\python parte_b_alteracoes.py
"""
import json
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import ecg_dados as E
from parte_b_cnn import MODELO_A3, PARTES_B, SHA_A3, bootstrap, carregar, nome_partes

RAIZ = Path(__file__).parent
SAIDA = RAIZ / "resultados"
R = E.ROTULOS
N_BOOT_OR = 500


def notas_a3(el, tag):
    """Notas (0-1) das 6 alterações pelo A3 congelado: características salvas -> camada final do A3."""
    import hashlib
    assert hashlib.sha256(MODELO_A3.read_bytes()).hexdigest() == SHA_A3, "o A3 não é o congelado"
    W, b = keras.models.load_model(MODELO_A3, compile=False).layers[-1].get_weights()
    c = np.load(SAIDA / f"parte_b_caracteristicas_A3_{tag}.npz")
    F = pd.DataFrame(c["F"], index=c["exam_id"]).loc[el["exam_id"]].to_numpy()
    P = 1 / (1 + np.exp(-(F @ W + b)))
    return pd.DataFrame(P, columns=[f"p_{r}" for r in R], index=el.index)


def conferir_notas_a3(el, P, n=64):
    """Confere em n exames de validação: notas pelas características salvas == A3 inteiro rodando no sinal."""
    va = el[el.grupo == "validação"].sort_values(["parte", "linha"]).head(n)
    g = E.GeradorECG(va, normalizar=True)
    direto = keras.models.load_model(MODELO_A3, compile=False).predict(g[0][0], verbose=0)
    via = P.set_index(el["exam_id"]).loc[g.info["exam_id"].iloc[:len(direto)]].to_numpy()
    return float(np.abs(direto - via).max())


def matriz(d, modelo):
    cols = {"B0": ["age", "is_male"],
            "B4": ["age", "is_male"] + R,
            "B4n": ["age", "is_male"] + R + ["normal_ecg"],
            "B4p": ["age", "is_male"] + [f"logit_{r}" for r in R]}[modelo]
    return d[cols].astype(float).to_numpy()


def razoes_de_chance(tr, seed=E.SEED):
    """OR de cada variável do B4 (idade por década), sem penalização, IC 95% por bootstrap de pacientes do treino."""
    X = tr[["age", "is_male"] + R].astype(float).copy()
    X["age"] = X["age"] / 10
    X, y = X.to_numpy(), tr["y"].to_numpy()
    ajuste = lambda X, y: LogisticRegression(penalty=None, max_iter=5000).fit(X, y).coef_[0]
    coef = ajuste(X, y)
    rng = np.random.default_rng(seed)
    boot = np.array([ajuste(X[i], y[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(N_BOOT_OR))])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    nomes = ["idade (por 10 anos)", "sexo masculino"] + R
    pos = [None, int(tr["is_male"].sum())] + [int(tr[r].sum()) for r in R]
    ob = [None, int(tr.loc[tr["is_male"], "y"].sum())] + [int(tr.loc[tr[r], "y"].sum()) for r in R]
    return pd.DataFrame({"variavel": nomes, "or": np.exp(coef), "or_ic_inf": np.exp(lo), "or_ic_sup": np.exp(hi),
                         "n_treino_com": pos, "obitos_treino_com": ob})


def main():
    tag = nome_partes(PARTES_B)
    el, rel = carregar(PARTES_B, SAIDA)
    for c in R + ["normal_ecg"]:
        el[c] = el[c].astype(bool)
    P = notas_a3(el, tag)
    el = el.join(P)
    for r in R:
        p = el[f"p_{r}"].clip(1e-6, 1 - 1e-6)
        el[f"logit_{r}"] = np.log(p / (1 - p))
    dif = conferir_notas_a3(el, P)
    print(f"conferência das notas do A3 (características salvas vs A3 inteiro no sinal, 64 exames): dif. máx {dif:.2e}")
    assert dif < 1e-4

    tr = el[el.grupo == "treino"]
    va = el[el.grupo == "validação"].sort_values(["parte", "linha"], ignore_index=True)
    y = va["y"].to_numpy()

    notas = {}
    for m in ("B0", "B4", "B4n", "B4p"):
        esc = StandardScaler().fit(matriz(tr, m))                         # escala só do treino
        lr = LogisticRegression(max_iter=2000, random_state=E.SEED).fit(esc.transform(matriz(tr, m)), tr["y"])
        notas[m] = lr.predict_proba(esc.transform(matriz(va, m)))[:, 1]
    prev = pd.read_csv(SAIDA / f"parte_b_cnn_{tag}_previsoes_validacao.csv").set_index("exam_id")
    assert set(prev.index) == set(va["exam_id"]), "pacientes de validação diferentes dos do parte_b_cnn.py"
    notas["B3z"] = prev.loc[va["exam_id"], "B3z"].to_numpy()
    b0_antigo = prev.loc[va["exam_id"], "B0"].to_numpy()
    print(f"B0 refeito vs B0 salvo do parte_b_cnn.py: dif. máx {np.abs(notas['B0'] - b0_antigo).max():.2e}")

    pares = [("B4", "B0"), ("B4n", "B0"), ("B4p", "B0"), ("B4n", "B4"), ("B4p", "B4"),
             ("B3z", "B0"), ("B3z", "B4"), ("B3z", "B4n"), ("B3z", "B4p")]
    ic = bootstrap(y, notas, pares)
    linhas = []
    for m in list(notas) + [f"{a} - {b}" for a, b in pares]:
        lin = dict(modelo=m, n_val=len(y), obitos_val=int(y.sum()))
        for met, f in (("auroc", roc_auc_score), ("auprc", average_precision_score)):
            lin[met] = f(y, notas[m]) if m in notas else f(y, notas[m.split(" - ")[0]]) - f(y, notas[m.split(" - ")[1]])
            lin[f"{met}_ic_inf"], lin[f"{met}_ic_sup"] = ic[(met, m)]
        linhas.append(lin)
    res = pd.DataFrame(linhas)
    res["partes"], res["n_treino"], res["obitos_treino"] = tag, len(tr), int(tr["y"].sum())
    res.to_csv(SAIDA / f"parte_b_alteracoes_{tag}.csv", index=False, float_format="%.4f")

    orr = razoes_de_chance(tr)
    orr.to_csv(SAIDA / f"parte_b_alteracoes_or_{tag}.csv", index=False, float_format="%.4f")

    # descritivo: quantos pacientes de validação têm alguma das 6 (laudo) e a taxa de óbito com e sem
    alguma = va[R].any(axis=1)
    desc = {"validacao_com_alguma_das_6": int(alguma.sum()), "obitos_com": int(va.loc[alguma, "y"].sum()),
            "validacao_sem_nenhuma": int((~alguma).sum()), "obitos_sem": int(va.loc[~alguma, "y"].sum()),
            "normal_ecg_validacao": int(va["normal_ecg"].sum()), "obitos_normal_ecg": int(va.loc[va["normal_ecg"], "y"].sum())}
    pd.DataFrame({"exam_id": va["exam_id"], "y": y, **notas}).to_csv(
        SAIDA / f"parte_b_alteracoes_{tag}_previsoes_validacao.csv", index=False, float_format="%.6f")

    pd.set_option("display.width", 200)
    print("dados:", json.dumps(rel, ensure_ascii=False))
    print("descritivo:", desc)
    print(res.round(4).to_string(index=False))
    print(orr.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
