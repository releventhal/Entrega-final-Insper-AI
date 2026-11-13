"""Robustez, tarefa 3: calibração do A3 (seed 42) na VALIDAÇÃO. DIAGNÓSTICO: nada é aplicado ao modelo nem aos limiares.
TÉCNICA FORA DAS AULAS: curva de confiabilidade, Brier score, ECE, calibração de Platt e isotônica.
Sem otimismo: os pacientes da validação são sorteados em 2 metades (seed 42); o calibrador é ajustado numa metade e avaliado na outra,
e vice-versa; as previsões "fora da dobra" das duas metades são juntadas. Só lê previsões salvas (não carrega modelo).
ECE = soma_bins (n_bin/n) * |fração positiva - nota média|; "largura" = 10 faixas de nota iguais, "massa" = 10 faixas com o mesmo nº de exames.
Saídas novas: resultados/robustez_03_calibracao.csv e figuras/robustez_03_calibracao.png.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from ecg_dados import ROTULOS
from robustez_comum import FIG, RES, auc_ap, carregar_val, reamostras, salvar_novo

BORDAS = np.array([0, .01, .02, .05, .1, .2, .3, .5, .7, .9, 1.0001])  # só para a figura (as classes são raras: muita massa perto de 0)


def ece(y, s, modo):
    b = np.minimum((s * 10).astype(int), 9) if modo == "largura" else np.minimum((np.argsort(np.argsort(s, kind="stable")) * 10 // len(s)), 9)
    return sum((b == i).mean() * abs(y[b == i].mean() - s[b == i].mean()) for i in np.unique(b))


def platt(s_ajuste, y_ajuste, s_novo):
    """regressão logística sobre logit(nota); C grande = quase sem regularização"""
    lg = lambda s: np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6)))[:, None]
    return LogisticRegression(C=1e6, max_iter=1000).fit(lg(s_ajuste), y_ajuste).predict_proba(lg(s_novo))[:, 1]


def isotonica(s_ajuste, y_ajuste, s_novo):
    return IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(s_ajuste, y_ajuste).predict(s_novo)


def main():
    S, Y, age, male, pid = carregar_val()
    rng = np.random.default_rng(42)
    up = np.unique(pid)
    metade = dict(zip(up, rng.permutation(len(up)) % 2))  # sorteio por PACIENTE
    m = np.array([metade[p] for p in pid])
    print(f"metades por paciente: {(m == 0).sum()} e {(m == 1).sum()} exames; pacientes em ambas: {len(set(pid[m == 0]) & set(pid[m == 1]))}")
    OOF = {"original": S.copy(), "Platt": np.zeros_like(S), "isotônica": np.zeros_like(S)}
    for k in range(6):
        for a, b in ((0, 1), (1, 0)):
            OOF["Platt"][m == b, k] = platt(S[m == a, k], Y[m == a, k], S[m == b, k])
            OOF["isotônica"][m == b, k] = isotonica(S[m == a, k], Y[m == a, k], S[m == b, k])
    boots = reamostras(pid, 500)  # 500 reamostras por paciente para o IC da diferença de Brier (declarado)
    linhas = []
    for k, r in enumerate(ROTULOS):
        y = Y[:, k]
        base = brier_score_loss(y, OOF["original"][:, k])
        for n, s in OOF.items():
            s = s[:, k]
            au, ap = auc_ap(y, s)
            br = brier_score_loss(y, s)
            d = np.array([brier_score_loss(y[i], OOF["original"][i, k]) - brier_score_loss(y[i], s[i]) for i in boots]) if n != "original" else None
            linhas.append(dict(rotulo=r, versao=n, positivos=int(y.sum()), prevalencia=y.mean(), brier=br,
                               brier_referencia_so_prevalencia=y.mean() * (1 - y.mean()), ece_largura=ece(y, s, "largura"), ece_massa=ece(y, s, "massa"),
                               nota_media=s.mean(), auroc=au, auprc=ap,
                               ganho_brier_vs_original=base - br if n != "original" else np.nan,
                               ganho_ic_inf=np.percentile(d, 2.5) if d is not None else np.nan, ganho_ic_sup=np.percentile(d, 97.5) if d is not None else np.nan))
    t = pd.DataFrame(linhas)
    salvar_novo(t, RES / "robustez_03_calibracao.csv")
    print(t.round(4).to_string(index=False))
    mac = t.groupby("versao")[["brier", "ece_largura", "ece_massa", "auroc", "auprc"]].mean().round(5)
    print("\nmédia entre as 6 alterações:\n", mac)

    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    cores = {"original": "k", "Platt": "tab:blue", "isotônica": "tab:orange"}
    for k, (r, a) in enumerate(zip(ROTULOS, ax.ravel())):
        a.plot([0, 1], [0, 1], ":", c="gray")
        for n, s in OOF.items():
            b = np.digitize(s[:, k], BORDAS) - 1
            ok = [i for i in np.unique(b) if (b == i).sum() >= 20]  # faixa com menos de 20 exames não entra (ruído)
            a.plot([s[b == i, k].mean() for i in ok], [Y[b == i, k].mean() for i in ok], "o-", ms=3, c=cores[n], label=n)
        a.set_title(f"{r} ({int(Y[:, k].sum())} positivos)")
        a.set_xlabel("nota média da faixa")
        a.set_ylabel("fração positiva")
        if k == 0:
            a.legend()
    fig.suptitle("A3 na validação: confiabilidade (Platt e isotônica ajustadas na outra metade dos pacientes; só diagnóstico)")
    fig.tight_layout()
    salvar_novo(fig, FIG / "robustez_03_calibracao.png")


if __name__ == "__main__":
    main()
