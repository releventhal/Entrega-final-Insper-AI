"""Robustez, tarefa 2: teste de estresse do A3 (seed 42, congelado) na VALIDAÇÃO. SÓ INFERÊNCIA; nada é treinado, limiares e modelo não mudam.
Cada perturbação é aplicada no sinal CRU (float32 do HDF5) e DEPOIS vem o pré-processamento oficial do A3 (ecg_dados.normalizar_exame:
média e desvio por exame e derivação, só na região com sinal). Perturbações só atuam na região com sinal e em derivações não zeradas
(o preenchimento de zeros e as derivações zeradas continuam zero, como nos dados reais). Ruído, deriva e recorte usam um rng por
(exame, perturbação), então não dependem do tamanho do lote. "Relativo ao desvio" = desvio da própria derivação naquele exame.

  python robustez_02_estresse.py prever --n 500   # amostra de 500 exames: mede tempo e RAM, estima o total, NÃO grava nada
  python robustez_02_estresse.py prever           # validação inteira -> resultados/robustez_02_previsoes.npz (novo; aborta se já existe)
  python robustez_02_estresse.py analisar         # métricas, IC (bootstrap por paciente, pareado com o original), figura
O teste (interno, PTB-XL, CODE-TEST) NUNCA é lido: o índice vem de a1_cnn.carregar_indices (só treino e validação).
"""
import argparse
import time

import numpy as np
import pandas as pd

from ecg_dados import ROTULOS, GeradorECG, normalizar_exame
from robustez_comum import FIG, MODELOS, RES, auc_ap, carregar_val, conferir_modelo, ic, limiares_json, reamostras, salvar_novo

FS = 400  # Hz
F_DERIVA = 0.3  # Hz, deriva de linha de base (respiração)
ARQ_PREV = RES / "robustez_02_previsoes.npz"


def ctx(x):
    """região com sinal (amostras com alguma derivação != 0), derivações não zeradas e desvio de cada uma nessa região"""
    ix = np.where(np.abs(x).max(axis=1) > 0)[0]
    ok = np.where(np.abs(x).max(axis=0) > 0)[0]
    sd = x[ix][:, ok].std(axis=0) if len(ix) else np.zeros(len(ok))
    return ix, ok, sd


# --- perturbações: f(x, ix, ok, sd, rng) -> x' (cópia) -----------------------------------------------------------------
def amplitude(f):
    return lambda x, ix, ok, sd, rng: x * np.float32(f)


def ruido(f):  # ruído gaussiano branco com desvio f * desvio da derivação
    def p(x, ix, ok, sd, rng):
        y = x.copy()
        y[np.ix_(ix, ok)] += (rng.standard_normal((len(ix), len(ok))) * sd * f).astype(np.float32)
        return y
    return p


def deriva(f):  # senoide de 0,3 Hz, fase aleatória, amplitude f * desvio da derivação
    def p(x, ix, ok, sd, rng):
        y = x.copy()
        t = (ix - ix[0]) / FS
        y[np.ix_(ix, ok)] += (f * sd * np.sin(2 * np.pi * F_DERIVA * t + rng.uniform(0, 2 * np.pi))[:, None]).astype(np.float32)
        return y
    return p


def zerar(d):
    def p(x, ix, ok, sd, rng):
        y = x.copy()
        y[:, d] = 0
        return y
    return p


def deslocar(seg):  # rotação circular da região com sinal (seg segundos)
    def p(x, ix, ok, sd, rng):
        y = x.copy()
        y[ix] = np.roll(x[ix], int(seg * FS), axis=0)
        return y
    return p


def recortar(frac):  # fica só um trecho de frac da região com sinal (início sorteado); o resto vira zero
    def p(x, ix, ok, sd, rng):
        y = np.zeros_like(x)
        n = int(len(ix) * frac)
        i0 = ix[0] + rng.integers(0, len(ix) - n + 1)
        y[i0:i0 + n] = x[i0:i0 + n]
        return y
    return p


DERIVACOES = ["DI", "DII", "DIII", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
PERTURBACOES = {"original": ("original", lambda x, ix, ok, sd, rng: x)}
PERTURBACOES |= {f"amplitude x{f}": ("amplitude", amplitude(f)) for f in (0.5, 2.0)}
PERTURBACOES |= {f"ruído {f} do desvio": ("ruído", ruido(f)) for f in (0.1, 0.3, 1.0)}
PERTURBACOES |= {f"deriva 0,3 Hz {f} do desvio": ("deriva de linha de base", deriva(f)) for f in (0.5, 1.0, 2.0)}
PERTURBACOES |= {f"derivação {n} zerada": ("derivação zerada", zerar(i)) for i, n in enumerate(DERIVACOES)}
PERTURBACOES |= {f"deslocamento circular {s} s": ("deslocamento temporal", deslocar(s)) for s in (0.5, 2.0)}
PERTURBACOES |= {f"recorte de {int(f * 100)}% do sinal": ("recorte", recortar(f)) for f in (0.5, 0.25)}


def prever(n=None):
    import keras
    import psutil
    from a1_cnn import carregar_indices
    sha = conferir_modelo()
    print("SHA256 do A3 conferido:", sha[:8], flush=True)
    _, va, _ = carregar_indices(partes=[0, 1, 2, 3])  # só treino e validação; o teste é cortado aqui
    if n:
        va = va.iloc[:n]
    g = GeradorECG(va, normalizar=False)  # sinal CRU; a normalização oficial é aplicada abaixo, depois da perturbação
    assert n or not ARQ_PREV.exists(), f"já existe, não sobrescrevo: {ARQ_PREV}"
    m = keras.models.load_model(MODELOS / "A3_melhor.keras", compile=False)
    nomes = list(PERTURBACOES)
    S = {k: [] for k in nomes}
    proc, t0 = psutil.Process(), time.perf_counter()
    for i in range(len(g)):
        X, _ = g[i]
        base = i * g.lote
        for j, (nome, (_, f)) in enumerate(PERTURBACOES.items()):
            Xp = []
            for k, x in enumerate(X):
                ix, ok, sd = ctx(x)
                rng = np.random.default_rng([42, base + k, j])
                Xp.append(normalizar_exame(f(x, ix, ok, sd, rng)) if len(ix) else x)
            S[nome].append(m.predict_on_batch(np.stack(Xp)))
        if i % 10 == 0:
            print(f"lote {i + 1}/{len(g)}  {time.perf_counter() - t0:.0f} s  RAM {proc.memory_info().rss / 2**30:.2f} GB", flush=True)
    seg = time.perf_counter() - t0
    S = {k: np.concatenate(v).astype(np.float32) for k, v in S.items()}
    pico = proc.memory_info().peak_wset / 2**30
    print(f"{len(g.info)} exames x {len(nomes)} versões em {seg:.0f} s ({seg / len(g.info) * 1000:.0f} ms por exame); pico de RAM {pico:.2f} GB")
    if n:
        total = seg / len(g.info) * 8158
        print(f"ESTIMATIVA para os 8.158 exames: {total / 60:.0f} min (estimativa linear a partir da amostra)")
        return
    salvo = np.load(RES / "A3_previsoes_validacao.npy")
    print(f"original recalculado vs salvo: diferença máxima {np.abs(S['original'] - salvo).max():.2e}")
    np.savez_compressed(ARQ_PREV, nomes=np.array(nomes), **{f"S{j}": S[k] for j, k in enumerate(nomes)})
    print("gravado", ARQ_PREV)


def _boot(Yv, So, Sp, boots, k_ref):
    """quedas (orig - pert) de AUROC e AUPRC macro e por alteração, em cada reamostra"""
    out = []
    for i in boots:
        ao, po = np.array([auc_ap(Yv[i, k], So[i, k]) for k in range(6)]).T
        ap_, pp = np.array([auc_ap(Yv[i, k], Sp[i, k]) for k in range(6)]).T
        out.append(np.concatenate([ao - ap_, po - pp]))  # (12,)
    return np.array(out)


def analisar(n_boot):
    from joblib import Parallel, delayed
    S0, Y, age, male, pid = carregar_val()
    d = np.load(ARQ_PREV)
    nomes = list(d["nomes"])
    S = {n: d[f"S{j}"].astype(np.float64) for j, n in enumerate(nomes)}
    print("original vs salvo:", np.abs(S["original"] - S0).max())
    lim = np.array([limiares_json()[r] for r in ROTULOS])
    boots = reamostras(pid, n_boot)
    nomes_p = [n for n in nomes if n != "original"]
    res = [_boot(Y, S["original"], S[n], boots, 0) for n in nomes_p]  # pareado: mesmas reamostras (sem joblib: os workers travaram)
    f1 = lambda s: np.mean([2 * ((s[:, k] >= lim[k]) & (Y[:, k] == 1)).sum() / max(((s[:, k] >= lim[k]).sum() + Y[:, k].sum()), 1) for k in range(6)])
    ao, po = np.array([auc_ap(Y[:, k], S["original"][:, k]) for k in range(6)]).T
    linhas = []
    for n in nomes:
        a, p = np.array([auc_ap(Y[:, k], S[n][:, k]) for k in range(6)]).T
        q = res[nomes_p.index(n)] if n != "original" else np.zeros((n_boot, 12))
        fam = PERTURBACOES[n][0]
        dec = int(((S[n] >= lim) != (S["original"] >= lim)).sum())
        for k, r in enumerate(ROTULOS + ["MACRO"]):
            kk = slice(None) if r == "MACRO" else k
            qa, qp = (q[:, :6].mean(1), q[:, 6:].mean(1)) if r == "MACRO" else (q[:, k], q[:, 6 + k])
            linhas.append(dict(perturbacao=n, familia=fam, rotulo=r, auroc=a.mean() if r == "MACRO" else a[k], auprc=p.mean() if r == "MACRO" else p[k],
                               queda_auroc=(ao.mean() - a.mean()) if r == "MACRO" else ao[k] - a[k], queda_auroc_ic_inf=ic(qa)[0], queda_auroc_ic_sup=ic(qa)[1],
                               queda_auprc=(po.mean() - p.mean()) if r == "MACRO" else po[k] - p[k], queda_auprc_ic_inf=ic(qp)[0], queda_auprc_ic_sup=ic(qp)[1],
                               f1_macro_limiares_do_json=f1(S[n]) if r == "MACRO" else np.nan, decisoes_trocadas_vs_original=dec if r == "MACRO" else np.nan,
                               dif_max_nota_vs_original=np.abs(S[n] - S["original"]).max() if r == "MACRO" else np.nan, n_boot=n_boot))
    t = pd.DataFrame(linhas)
    salvar_novo(t, RES / "robustez_02_estresse.csv")
    m = t[t.rotulo == "MACRO"].set_index("perturbacao")
    pd.set_option("display.width", 220)
    print(m[["auroc", "queda_auroc", "queda_auroc_ic_inf", "queda_auroc_ic_sup", "auprc", "queda_auprc", "f1_macro_limiares_do_json", "decisoes_trocadas_vs_original", "dif_max_nota_vs_original"]].round(4).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mm = m.drop("original")
    fig, ax = plt.subplots(figsize=(9, 9))
    y = np.arange(len(mm))[::-1]
    cor = {f: c for f, c in zip(dict.fromkeys(mm.familia), plt.cm.tab10.colors)}
    ax.barh(y, mm.queda_auroc, color=[cor[f] for f in mm.familia])
    ax.errorbar(mm.queda_auroc, y, xerr=[mm.queda_auroc - mm.queda_auroc_ic_inf, mm.queda_auroc_ic_sup - mm.queda_auroc], fmt="none", ecolor="k", lw=1)
    ax.set_yticks(y, mm.index)
    ax.axvline(0, c="k", lw=.8)
    ax.set_xlabel(f"queda do AUROC macro (original − perturbado), IC 95% (bootstrap por paciente, pareado, {n_boot} reamostras)")
    ax.set_title("A3 (validação, 8.158 exames): estresse no sinal cru, depois a normalização oficial")
    ax.handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cor.values()]
    ax.legend(ax.handles, list(cor), loc="lower right", fontsize=8)
    fig.tight_layout()
    salvar_novo(fig, FIG / "robustez_02_estresse.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=["prever", "analisar"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=200)
    a = ap.parse_args()
    prever(a.n) if a.modo == "prever" else analisar(a.n_boot)

