"""Parte B: curvas de sobrevivência (Kaplan-Meier) por quartil de risco, SÓ VALIDAÇÃO. Fora das aulas (análise de
sobrevivência), explicada em linguagem simples: a curva mostra, ano a ano, que fração do grupo ainda está viva,
contando cada paciente só enquanto ele foi acompanhado.

Por que: a regra do horizonte (seção 3.6) tira da avaliação quem ficou vivo e foi acompanhado menos de 2 anos. A curva
de Kaplan-Meier usa TODOS os pacientes de validação (o primeiro exame de cada um, sem os exames todo-zero), inclusive
esses, e vai até o fim do acompanhamento (~7 anos), não só até 2 anos.

Nada é treinado. Notas:
  B0   regressão idade + sexo ajustada nos MESMOS pacientes de treino elegíveis do parte_b_cnn.py (mesma função `ajustar`);
  B3z  o modelo já treinado (modelos/B3z_<tag>_melhor.keras), aplicado aos pacientes de validação; idade padronizada com a
       média e o desvio do treino elegível, igual ao treino. Conferência: nos elegíveis, as notas têm de bater com as salvas.
Grupos: quartis da nota na própria validação (Q1 = 25% de menor risco, Q4 = 25% de maior risco).
Saídas: resultados/parte_b_km_<tag>.csv (sobrevida em 1, 2, 3 e 5 anos por quartil) e figuras/parte_b_km_<tag>.png.
O teste da Parte B não é aberto (o grupo teste é cortado na carga).

Uso: .venv\\Scripts\\python parte_b_km.py [--partes 0,...,7]
"""
import argparse
from pathlib import Path

import keras
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ecg_dados as E
from parte_b_cnn import HORIZONTE, PARTES_B, GeradorObito, nome_partes, todo_zero_por_sinal
from parte_b_tabular import ajustar, features, rotular

RAIZ = Path(__file__).parent
SAIDA, FIG = RAIZ / "resultados", RAIZ / "figuras"
ANOS = [1, 2, 3, 5]


def kaplan_meier(t, e):
    """Devolve (tempos, sobrevida) da curva de Kaplan-Meier. t = anos de acompanhamento, e = morreu (1) ou censurado (0)."""
    o = np.argsort(t, kind="stable")
    t, e = t[o], e[o]
    tempos, mortes = np.unique(t[e == 1], return_counts=True)
    em_risco = len(t) - np.searchsorted(t, tempos, side="left")       # ainda acompanhados logo antes de cada tempo
    return tempos, np.cumprod(1 - mortes / em_risco)


def sobrevida_em(tempos, s, ano):
    i = np.searchsorted(tempos, ano, side="right") - 1
    return 1.0 if i < 0 else float(s[i])


def main(partes):
    tag = nome_partes(partes)
    info = E.construir_indice(partes)
    info = info[info["grupo"].isin(["treino", "validação"])]          # o grupo teste para aqui
    info = info[info["death"].notna()].copy()
    info["death"] = info["death"].astype(bool)
    info["delta"] = info["nn_predicted_age"] - info["age"]
    cache = SAIDA / f"parte_b_todo_zero_{tag}.npy"
    assert cache.exists(), "rode antes: python parte_b_cnn.py preparar"
    info = info[~info["exam_id"].isin(todo_zero_por_sinal(info, cache))]
    el, _ = rotular(info, HORIZONTE)
    tr = el[el.grupo == "treino"]
    va = info[info.grupo == "validação"].sort_values(["parte", "linha"], ignore_index=True)
    va["y"] = 0                                                       # só para o gerador; não é usado

    esc, mods = ajustar(tr)
    notas = {"B0": mods["B0"].predict_proba(features(va, esc, "B0"))[:, 1]}
    g = GeradorObito(va, idade=(tr["age"].mean(), tr["age"].std()))
    m = keras.models.load_model(RAIZ / "modelos" / f"B3z_{tag}_melhor.keras", compile=False)
    s = m.predict(g, verbose=0)[:, 0]
    notas["B3z"] = pd.Series(s, index=g.info["exam_id"]).loc[va["exam_id"]].to_numpy()

    salvas = pd.read_csv(SAIDA / f"parte_b_cnn_{tag}_previsoes_validacao.csv").set_index("exam_id")
    comum = va["exam_id"].isin(salvas.index).to_numpy()
    for k in notas:
        dif = np.abs(notas[k][comum] - salvas.loc[va.loc[comum, "exam_id"], k].to_numpy()).max()
        print(f"{k}: conferência com as notas salvas nos {comum.sum()} elegíveis: dif. máx {dif:.2e}")
        assert dif < 1e-3   # CPU ~5e-7; GPU com TF32 pode dar ~1e-4 (14/11)

    t, e = va["timey"].to_numpy(float), va["death"].to_numpy(int)
    linhas = []
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    cores = ["#2a9d8f", "#8ab17d", "#e9c46a", "#e76f51"]
    for ax, (k, sc) in zip(eixos, notas.items()):
        q = pd.qcut(sc, 4, labels=False)
        for j in range(4):
            msk = q == j
            tempos, sv = kaplan_meier(t[msk], e[msk])
            ax.step(np.r_[0, tempos], np.r_[1, sv], where="post", color=cores[j], label=f"Q{j + 1} (n={msk.sum()})")
            lin = dict(modelo=k, quartil=f"Q{j + 1}", n=int(msk.sum()), obitos=int(e[msk].sum()),
                       idade_media=float(va.loc[msk, "age"].mean()), nota_min=float(sc[msk].min()), nota_max=float(sc[msk].max()))
            lin.update({f"sobrevida_{a}a": sobrevida_em(tempos, sv, a) for a in ANOS})
            linhas.append(lin)
        ax.set_title(f"{k}: quartis da nota de risco (validação)")
        ax.set_xlabel("anos desde o exame")
        ax.axvline(HORIZONTE, color="gray", lw=0.8, ls=":")
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(alpha=0.3)
    eixos[0].set_ylabel("fração viva (Kaplan-Meier)")
    fig.suptitle(f"Parte B, validação {tag}: {len(va)} pacientes, {e.sum()} óbitos em todo o acompanhamento", fontsize=10)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"parte_b_km_{tag}.png", dpi=130)
    res = pd.DataFrame(linhas)
    res["partes"] = tag
    res.to_csv(SAIDA / f"parte_b_km_{tag}.csv", index=False, float_format="%.4f")
    print(f"validação: {len(va)} pacientes ({(~comum).sum()} censurados antes de {HORIZONTE} anos), {e.sum()} óbitos")
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--partes", default=",".join(map(str, PARTES_B)))
    main([int(p) for p in ap.parse_args().partes.split(",")])
