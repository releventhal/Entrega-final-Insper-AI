"""A0: baselines só com dados tabulares (T3). Treina no TREINO, avalia na VALIDAÇÃO. O teste não é usado.

A0a: sempre negativo (mostra que acurácia engana).
A0b: regressão logística só com idade + sexo, um modelo por alteração (quanto a idade sozinha já explica).
Gera `resultados/experimentos.csv` (tabela de experimentos) e imprime em markdown.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

from ecg_dados import DADOS, ROTULOS, SEED, construir_indice

PASTA_RESULTADOS = DADOS.parent / "resultados"


def melhor_limiar_f1(y, s):
    """Limiar que maximiza o F1 (na validação). Se o modelo não separa nada, devolve 1.0 (nunca prevê positivo)."""
    prec, rec, lim = precision_recall_curve(y, s)  # prec/rec têm 1 elemento a mais que lim
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(lim[np.argmax(f1)]) if f1.max() > 0 else 1.0


def metricas(y, s, limiar):
    """AUROC e AUPRC não dependem de limiar. O resto depende (vem do limiar escolhido na validação)."""
    p = s >= limiar
    tp, fp, fn, tn = (p & (y == 1)).sum(), (p & (y == 0)).sum(), (~p & (y == 1)).sum(), (~p & (y == 0)).sum()
    div = lambda a, b: a / b if b else 0.0  # sem previsão positiva: precisão = 0 por convenção
    sens, prec = div(tp, tp + fn), div(tp, tp + fp)
    return dict(auroc=roc_auc_score(y, s), auprc=average_precision_score(y, s), limiar=limiar,
                sensibilidade=sens, precisao=prec, f1=div(2 * prec * sens, prec + sens),
                especificidade=div(tn, tn + fp), acuracia=(tp + tn) / len(y))


def main():
    info = construir_indice()
    tr, va = info[info.grupo == "treino"], info[info.grupo == "validação"]
    assert not set(tr.patient_id) & set(va.patient_id), "paciente em treino e validação!"
    print(f"treino: {len(tr):,} exames | validação: {len(va):,} exames (partes 0-3; o teste não foi aberto)")

    # idade padronizada (média 0, desvio 1 do TREINO) ajuda o otimizador; sexo já é 0/1
    esc = StandardScaler().fit(tr[["age"]])
    feat = lambda d: np.column_stack([esc.transform(d[["age"]]), d["is_male"].astype(float)])
    Xtr, Xva = feat(tr), feat(va)

    linhas = []
    for r in ROTULOS:
        ytr, yva = tr[r].to_numpy(), va[r].to_numpy()
        # A0a: nota 0 para todo mundo, limiar 0,5 -> ninguém é positivo
        linhas.append(dict(experimento="A0a sempre negativo", rotulo=r, n_val=len(yva), positivos_val=int(yva.sum()),
                           **metricas(yva, np.zeros(len(yva)), 0.5)))
        # A0b: logística; limiar escolhido na validação
        m = LogisticRegression(max_iter=1000, random_state=SEED).fit(Xtr, ytr)
        s = m.predict_proba(Xva)[:, 1]
        linhas.append(dict(experimento="A0b idade+sexo", rotulo=r, n_val=len(yva), positivos_val=int(yva.sum()),
                           **metricas(yva, s, melhor_limiar_f1(yva, s))))

    df = pd.DataFrame(linhas)
    macro = df.groupby("experimento")[["auroc", "auprc", "sensibilidade", "precisao", "f1", "especificidade", "acuracia"]].mean()
    macro = macro.reset_index().assign(rotulo="MACRO", n_val=len(va), positivos_val=np.nan, limiar=np.nan)
    df = pd.concat([df, macro], ignore_index=True)
    PASTA_RESULTADOS.mkdir(exist_ok=True)
    arq = PASTA_RESULTADOS / "experimentos.csv"  # troca só as linhas do A0 e mantém as de outros experimentos (A1, A2...)
    if arq.exists():
        df = pd.concat([pd.read_csv(arq).query("~experimento.isin(@df.experimento.unique())"), df], ignore_index=True)
    df.to_csv(arq, index=False, float_format="%.4f")
    print(df.round(3).to_string(index=False))  # sem to_markdown: exigiria o pacote tabulate (fora do requirements.txt)


if __name__ == "__main__":
    main()
