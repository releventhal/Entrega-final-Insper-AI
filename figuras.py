"""Bloco 5 (08/11): figuras para a apresentação intermediária. Só lê resultados já salvos em resultados/
(nenhum modelo é rodado, nenhum sinal é lido). Tudo de VALIDAÇÃO; provisório, a confirmar com a Renata.
Uso: .venv\\Scripts\\python figuras.py   ->  figuras/*.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

R = Path(__file__).parent / "resultados"
F = Path(__file__).parent / "figuras"
ROTULOS = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF"]
CORES = dict(zip(ROTULOS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))  # as do notebook (seção 1)
CINZA = "#555555"
MODELOS = {"A0b": "#999999", "A1": "#56b4e9", "A2": "#0072b2", "A3": "#d55e00", "A4": "#cc79a7", "B0": "#999999", "B1": "#0072b2"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e5e5e5",
                     "axes.axisbelow": True, "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14, "legend.fontsize": 12,
                     "figure.dpi": 100, "savefig.dpi": 110, "savefig.bbox": "tight"})


def salvar(fig, nome):
    fig.savefig(F / nome)
    plt.close(fig)
    print("ok", nome)


def rotular_barras(ax, barras, fmt="{:.3f}"):
    for b in barras:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()), ha="center", va="bottom", fontsize=11)


def a_barras_modelos(exp):
    """A0b/A1/A2/A3 no MESMO conjunto (2.069 exames de validação da parte 0)."""
    nomes = {"A0b": "A0b idade+sexo (mesmo conjunto do A1)", "A1": "A1 CNN (provisório, parte 0)",
             "A2": "A2 provisório (normalização por exame e derivação)",
             "A3": "A3 provisório (partes 0 a 3; normalização por exame e derivação) | só os 2069 exames de validação da parte 0 (limiar da validação completa)"}
    mac = exp[exp.rotulo == "MACRO"].set_index("experimento")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    larg = 0.2
    for i, (m, e) in enumerate(nomes.items()):
        v = mac.loc[e, ["auroc", "auprc", "f1"]].to_numpy(float)
        rotular_barras(ax, ax.bar(np.arange(3) + (i - 1.5) * larg, v, larg, color=MODELOS[m], label=m))
    ax.set_xticks(range(3), ["AUROC", "AUPRC", "F1"])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("média macro (6 alterações)")
    ax.set_title("Validação, mesmos 2.069 exames (parte 0), uma seed")
    ax.legend(ncol=4, loc="upper center", frameon=False)
    salvar(fig, "a_modelos_macro.png")


def b_por_alteracao(exp):
    a0 = exp[exp.experimento == "A0b idade+sexo (mesmo conjunto do A3)"].set_index("rotulo")
    a3 = exp[exp.experimento == "A3 provisório (partes 0 a 3; normalização por exame e derivação)"].set_index("rotulo")
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, met, nome in zip(axs, ["auroc", "auprc"], ["AUROC", "AUPRC"]):
        x = np.arange(6)
        rotular_barras(ax, ax.bar(x - 0.2, a0.loc[ROTULOS, met], 0.4, color=MODELOS["A0b"], label="A0b idade+sexo"), "{:.2f}")
        rotular_barras(ax, ax.bar(x + 0.2, a3.loc[ROTULOS, met], 0.4, color=MODELOS["A3"], label="A3 CNN"), "{:.2f}")
        ax.set_xticks(x, ROTULOS)
        ax.set_ylim(0, 1.12)
        ax.set_title(nome)
    axs[0].legend(frameon=False, loc="lower left")
    fig.suptitle("Por alteração, validação completa (8.158 exames)", fontsize=16)
    salvar(fig, "b_por_alteracao_A0b_A3.png")


def c_roc_pr(arr):
    S, Y = arr[:, :6], arr[:, 6:12]
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    for k, r in enumerate(ROTULOS):
        fpr, tpr, _ = roc_curve(Y[:, k], S[:, k])
        axs[0].plot(fpr, tpr, color=CORES[r], lw=2, label=r)
        p, rec, _ = precision_recall_curve(Y[:, k], S[:, k])
        axs[1].plot(rec, p, color=CORES[r], lw=2, label=f"{r} ({int(Y[:, k].sum())} pos.)")
    axs[0].plot([0, 1], [0, 1], ls="--", color=CINZA, lw=1)
    axs[0].set(xlabel="1 − especificidade", ylabel="sensibilidade", title="Curva ROC")
    axs[1].set(xlabel="sensibilidade (recall)", ylabel="precisão", title="Curva Precisão-Recall")
    axs[0].legend(frameon=False, loc="lower right")
    axs[1].legend(frameon=False, loc="lower left")
    fig.suptitle("A3 na validação (8.158 exames)", fontsize=16)
    salvar(fig, "c_roc_pr_A3.png")


def d_calibracao(cal):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], ls="--", color=CINZA, lw=1, label="calibração perfeita")
    for r in ROTULOS:
        d = cal[(cal.rotulo == r) & (cal.n >= 10)]
        ax.plot(d.nota_media, d.fracao_positiva, "o-", color=CORES[r], lw=2, label=r)
    ax.set(xlabel="nota média do modelo (faixa)", ylabel="fração de positivos de verdade", xlim=(0, 1), ylim=(0, 1),
           title="Calibração do A3 (validação)")
    ax.legend(frameon=False, loc="upper left")
    salvar(fig, "d_calibracao_A3.png")


def e_loss(log, nome="A3", melhor=4):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ep = log.epoch + 1
    ax.plot(ep, log.loss, "o-", color=MODELOS["A2"], lw=2, label="treino")
    ax.plot(ep, log.val_loss, "o-", color=MODELOS["A3"], lw=2, label="validação")
    ax.axvline(melhor, ls="--", color=CINZA, lw=1)
    ax.annotate(f"melhor época ({melhor})", (melhor, log.val_loss.max()), xytext=(5, 0), textcoords="offset points", fontsize=12)
    ax.set(xlabel="época", ylabel="perda (binary cross-entropy)", title=f"Curvas de perda do {nome}")
    ax.legend(frameon=False)
    salvar(fig, "e_loss_A3.png")


def f_simpson(direto, faixas):
    ordem = ["ECG 8+ mais jovem", "semelhante (±8)", "ECG 8+ mais velho"]
    cores = ["#0072b2", "#999999", "#d55e00"]
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1, 2.4]})
    d = direto.set_index("delta").loc[ordem]
    b = axs[0].bar(range(3), d.taxa * 100, color=cores)
    rotular_barras(axs[0], b, "{:.1f}%")
    axs[0].set_xticks(range(3), [f"{n}\n(idade\nmédia {i:.0f})" for n, i in zip(["8+ mais\njovem", "semelhante", "8+ mais\nvelho"], d.idade_media)])
    axs[0].set(ylabel="óbito (%)", title="Sem controlar a idade")
    x = np.arange(len(faixas))
    for i, (g, c) in enumerate(zip(ordem, cores)):
        axs[1].bar(x + (i - 1) * 0.27, faixas[g] * 100, 0.27, color=c, label=g)
    axs[1].set_xticks(x, faixas.index)
    axs[1].set(xlabel="faixa de idade real", ylabel="óbito (%)", title="Dentro de cada faixa de idade")
    axs[1].legend(frameon=False, title="idade do ECG vs idade real")
    fig.suptitle("Paradoxo de Simpson: óbito por 'idade do ECG' (treino + validação, 186.975 pacientes)", fontsize=16)
    salvar(fig, "f_simpson.png")


def g_b0_b1(fx, geral):
    rot = [*fx.faixa, "todas"]
    v0 = [*fx.auroc_B0, geral.auroc_B0]
    v1 = [*fx.auroc_B1, geral.auroc_B1]
    ic0 = np.array([*fx.auroc_B0_ic.map(eval), eval(geral.auroc_B0_ic)])
    ic1 = np.array([*fx.auroc_B1_ic.map(eval), eval(geral.auroc_B1_ic)])
    x = np.arange(len(rot))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for dx, v, ic, m, nome in ((-0.2, v0, ic0, "B0", "B0 idade+sexo"), (0.2, v1, ic1, "B1", "B1 + idade do ECG")):
        v = np.array(v)
        ax.bar(x + dx, v, 0.4, color=MODELOS[m], label=nome, yerr=[v - ic[:, 0], ic[:, 1] - v], capsize=4)
        for xi, vi in zip(x + dx, v):  # valor dentro da barra, para não brigar com a barra de erro
            ax.annotate(f"{vi:.2f}", (xi, 0.415), ha="center", fontsize=11, color="white", fontweight="bold")
    obitos = [*fx.obitos, int(geral.obitos_val)]
    ax.set_xticks(x, [f"{r}\n({o} óbitos)" for r, o in zip(rot, obitos)])
    ax.set_ylim(0.4, 0.95)
    ax.axhline(0.5, color=CINZA, lw=1, ls="--")
    ax.set(ylabel="AUROC (IC 95%)", title="Óbito em 2 anos, validação: B0 vs B1 por faixa de idade")
    ax.legend(frameon=False, loc="upper left")
    salvar(fig, "g_B0_B1_faixas.png")


def h_seeds(tab):
    met = ["auroc", "auprc", "f1"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, m in zip(axs, met):
        d = tab[tab.metrica == m].set_index("rotulo").loc[[*ROTULOS, "MACRO"]]
        cores = [CORES.get(r, CINZA) for r in d.index]
        ax.bar(range(7), d.media, yerr=d.desvio, capsize=5, color=cores)
        for i, (_, lin) in enumerate(d.iterrows()):
            ax.scatter([i] * len(lin.valores), lin.valores, color="black", s=14, zorder=3)
        ax.set_xticks(range(7), d.index, rotation=30)
        ax.set_title(m.upper())
        lo = d.media.min() - 3 * d.desvio.max() - 0.05
        ax.set_ylim(max(0, lo), 1.0)
    fig.suptitle(f"A3 em {int(tab.n_seeds.iloc[0])} seeds (média ± desvio; pontos = cada seed), validação", fontsize=16)
    salvar(fig, "h_seeds_A3.png")


def main():
    F.mkdir(exist_ok=True)
    exp = pd.read_csv(R / "experimentos.csv")
    a_barras_modelos(exp)
    b_por_alteracao(exp)
    if (R / "erros_A3" / "previsoes_e_rotulos.npy").exists():
        c_roc_pr(np.load(R / "erros_A3" / "previsoes_e_rotulos.npy"))
        d_calibracao(pd.read_csv(R / "erros_A3" / "calibracao.csv"))
    e_loss(pd.read_csv(R / "A3_log_por_epoca.csv"))
    f_simpson(pd.read_csv(R / "parte_b_simpson_direto.csv"), pd.read_csv(R / "parte_b_simpson_faixas.csv", header=[0, 1], index_col=0)["mean"])
    sens = pd.read_csv(R / "parte_b_B0_B1.csv")
    g_b0_b1(pd.read_csv(R / "parte_b_faixas_2anos.csv"), sens[sens.horizonte == 2].iloc[0])
    if (R / "seeds_A3.csv").exists():
        t = pd.read_csv(R / "seeds_A3.csv")
        t["valores"] = t.valores.map(eval)
        h_seeds(t)


if __name__ == "__main__":
    main()
