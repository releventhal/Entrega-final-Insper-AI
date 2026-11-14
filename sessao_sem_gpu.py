"""Sessão sem GPU (14/11): análises leves SÓ na validação (8.158 exames do A3). O teste (interno, PTB-XL, CODE-TEST, Parte B) nunca é lido.
Não carrega modelo nem TensorFlow: só previsões salvas em resultados/*.npy e atributos em resultados/atributos*.csv.gz.
Não altera nenhum arquivo existente: toda saída é nova, com prefixo resultados/sgpu_ e figuras/sgpu_ (salvar_novo aborta se já existir).

  python sessao_sem_gpu.py operacao   # (a) pontos de operação por alteração, com IC por bootstrap de pacientes
  python sessao_sem_gpu.py ensemble   # (b) ensemble A3+A6 (média das probabilidades), diferença pareada vs A3
  python sessao_sem_gpu.py rf         # (d) floresta A0e-1 com GroupKFold por paciente (5 folds) em treino+validação
  python sessao_sem_gpu.py checar     # self-check do cálculo de ponto de operação com dados sintéticos
"""
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_curve

from a0_baselines import melhor_limiar_f1
from ecg_dados import ROTULOS, SEED
from robustez_comum import FIG, RES, carregar_val, ic, limiares_json, macro_auc_ap, reamostras, salvar_novo

N_BOOT = 1000
ESPS, SENS = (0.90, 0.95, 0.99), (0.90, 0.95)


def modelos_val(a5=("A5_completo", "A5_completo_parcial")):
    """Previsões de validação alinhadas com carregar_val (mesma ordem; conferido pela forma e pelo AUROC macro impresso)."""
    S3, Y, _, _, pid = carregar_val()
    M = {"A3": S3, "A6_completo": np.load(RES / "A6_completo_previsoes_validacao.npy").astype(np.float64)}
    for nome in a5:  # o final, se existir; senão o parcial (rotulado como tal)
        arq = RES / f"{nome}_previsoes_validacao.npy"
        if arq.exists():
            M[nome] = np.load(arq).astype(np.float64)
            break
    assert all(s.shape == Y.shape for s in M.values())
    for n, s in M.items():
        print(f"{n}: AUROC macro {np.nanmean(macro_auc_ap(Y, s)[0]):.4f} | AUPRC macro {np.nanmean(macro_auc_ap(Y, s)[1]):.4f}", flush=True)
    return M, Y, pid


def pontos(y, s):
    """Pontos de operação de uma alteração. Limiar escolhido NOS PRÓPRIOS dados (otimista).
    sens_espX = maior sensibilidade com especificidade >= X; esp_sensX = maior especificidade com sensibilidade >= X;
    precisão no mesmo ponto; falsos alarmes por 1000 negativos para a alteração = 1000 * (1 - especificidade)."""
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    P, N = y.sum(), len(y) - y.sum()
    prec = lambda i: (tpr[i] * P) / max(tpr[i] * P + fpr[i] * N, 1e-12)
    r = {}
    for e in ESPS:
        i = np.flatnonzero(fpr <= 1 - e + 1e-12)[-1]
        r[f"sens_esp{int(e * 100)}"], r[f"prec_esp{int(e * 100)}"], r[f"fa1000_esp{int(e * 100)}"] = tpr[i], prec(i), 1000 * fpr[i]
    for t in SENS:
        i = np.flatnonzero(tpr >= t - 1e-12)[0]
        r[f"esp_sens{int(t * 100)}"], r[f"prec_sens{int(t * 100)}"], r[f"fa1000_sens{int(t * 100)}"] = 1 - fpr[i], prec(i), 1000 * fpr[i]
    return r


def checar():
    y = np.array([0] * 100 + [1] * 10)
    s = np.r_[np.linspace(0, 1, 100), np.linspace(0.5, 1.5, 10)]  # negativos 0..1; positivos 0,5..1,5
    r = pontos(y, s)
    assert abs(r["sens_esp99"] - 0.5) < 0.11, r            # com esp >= 99% (no máximo 1 FP) sobram os positivos acima de ~0,99
    assert r["esp_sens90"] >= 0.45 and r["fa1000_esp90"] <= 100 + 1e-9, r
    assert pontos(np.r_[np.zeros(50), np.ones(50)], np.r_[np.zeros(50), np.ones(50)])["sens_esp99"] == 1.0
    print("checar: ok", r)


def operacao(prefixo="sgpu", a5=("A5_completo", "A5_completo_parcial"), aviso_a5=None):
    """prefixo/a5: sessão 2 (14/11 à tarde) usa prefixo "sessao2" e o A5 mais recente (final, senão o parcial de 14/11, época 9)."""
    M, Y, pid = modelos_val(a5)
    idxs = reamostras(pid, N_BOOT, SEED)
    linhas = []
    for n, S in M.items():
        t0 = time.perf_counter()
        for k, rot in enumerate(ROTULOS):
            pt = pontos(Y[:, k], S[:, k])
            bs = pd.DataFrame([pontos(Y[ix, k], S[ix, k]) for ix in idxs if 0 < Y[ix, k].sum() < len(ix)])
            for m, v in pt.items():
                lo, hi = ic(bs[m].to_numpy())
                linhas.append(dict(modelo=n, rotulo=rot, metrica=m, valor=v, ic_inf=lo, ic_sup=hi, n_boot=len(bs), positivos=int(Y[:, k].sum()), negativos=int((Y[:, k] == 0).sum())))
        print(f"{n}: bootstrap {time.perf_counter() - t0:.0f} s", flush=True)
    tab = pd.DataFrame(linhas)
    salvar_novo(tab, RES / f"{prefixo}_pontos_operacao.csv")

    # figura: sensibilidade com especificidade 95% e 99%, por alteração e modelo, com IC
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    w = 0.8 / len(M)
    for ax, m in zip(axs, ("sens_esp95", "sens_esp99")):
        for j, n in enumerate(M):
            d = tab[(tab.modelo == n) & (tab.metrica == m)].set_index("rotulo").loc[ROTULOS]
            x = np.arange(6) + (j - (len(M) - 1) / 2) * w
            ax.bar(x, d.valor, w, label=n, yerr=[d.valor - d.ic_inf, d.ic_sup - d.valor], capsize=2)
        ax.set_xticks(range(6), ROTULOS)
        ax.set_title(f"Sensibilidade com especificidade {m[-2:]}%")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    axs[0].set_ylabel("sensibilidade (IC 95%, bootstrap por paciente)")
    # legenda fora dos eixos (na versão sgpu de 14/11 de manhã ela cobria a barra do A6 em 1dAVb)
    fig.legend(*axs[0].get_legend_handles_labels(), fontsize=8, loc="lower center", ncol=len(M), bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Pontos de operação na VALIDAÇÃO (limiar escolhido na própria validação: ilustrativo, não é número de teste)", fontsize=10)
    salvar_novo(fig, FIG / f"{prefixo}_pontos_operacao.png")

    # resumo .md
    def f(n, r, m):
        x = tab[(tab.modelo == n) & (tab.rotulo == r) & (tab.metrica == m)].iloc[0]
        return f"{x.valor:.3f} ({x.ic_inf:.3f}-{x.ic_sup:.3f})" if not m.startswith("fa") else f"{x.valor:.0f} ({x.ic_inf:.0f}-{x.ic_sup:.0f})"
    md = ["# Pontos de operação na validação (sessão sem GPU, 14/11)", "",
          "> **Aviso:** números de **validação** (8.158 exames), com o limiar de cada ponto escolhido **na própria validação**. São **ilustrativos** "
          "(otimistas), **não** são números finais de teste. O teste interno, o PTB-XL e o CODE-TEST não foram abertos.", "",
          f"- Gerado por `sessao_sem_gpu.py {'operacao' if prefixo == 'sgpu' else 'operacao2'}`; tabela completa em `resultados/{prefixo}_pontos_operacao.csv`; figura `figuras/{prefixo}_pontos_operacao.png`.",
          f"- IC 95%: bootstrap por paciente ({N_BOOT} reamostras, seed {SEED}), com o limiar **reescolhido** em cada reamostra.",
          "- `FA/1000` = falsos alarmes por 1.000 exames **negativos para aquela alteração** = 1000 × (1 − especificidade).",
          "- Modelos: " + ", ".join(f"`{n}`" for n in M) + (". **`A5_completo_parcial` = 3 épocas, sem early stopping: PARCIAL, NÃO CONCLUSIVO.**" if "A5_completo_parcial" in M else "."), ""]
    if aviso_a5:
        md.insert(-1, aviso_a5)
    for n in M:
        md += [f"## {n}", "", "| Alteração | positivos | Sens. c/ esp. 90% | Sens. c/ esp. 95% | Sens. c/ esp. 99% | Prec. c/ esp. 99% | Esp. c/ sens. 90% | Esp. c/ sens. 95% | FA/1000 c/ sens. 90% | Prec. c/ sens. 90% |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for r in ROTULOS:
            p = tab[(tab.modelo == n) & (tab.rotulo == r)].positivos.iloc[0]
            md.append(f"| {r} | {p} | " + " | ".join(f(n, r, m) for m in ("sens_esp90", "sens_esp95", "sens_esp99", "prec_esp99", "esp_sens90", "esp_sens95", "fa1000_sens90", "prec_sens90")) + " |")
        md.append("")
    arq = RES / f"{prefixo}_pontos_operacao.md"
    assert not arq.exists()
    arq.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


def ensemble():
    M, Y, pid = modelos_val()
    S3, S6 = M["A3"], M["A6_completo"]
    E = (S3 + S6) / 2
    lim3 = np.array([limiares_json()[r] for r in ROTULOS])
    cand = {"A3 (limiares do JSON)": (S3, lim3), "A6_completo (limiar F1-máx val.)": (S6, None), "ensemble A3+A6 (limiar F1-máx val.)": (E, None)}
    f1v = lambda S, lim, ix=slice(None): np.array([f1_score(Y[ix, k], S[ix, k] >= lim[k], zero_division=0) for k in range(6)])
    for n, (S, lim) in cand.items():
        if lim is None:
            cand[n] = (S, np.array([melhor_limiar_f1(Y[:, k], S[:, k]) for k in range(6)]))
    idxs = reamostras(pid, N_BOOT, SEED)
    B = {n: [] for n in cand}
    t0 = time.perf_counter()
    for ix in idxs:
        for n, (S, lim) in cand.items():
            au, ap = macro_auc_ap(Y, S, ix)
            B[n].append(np.r_[au, ap, f1v(S, lim, ix)])
    print(f"bootstrap {time.perf_counter() - t0:.0f} s", flush=True)
    B = {n: np.array(v) for n, v in B.items()}
    linhas, ref = [], "A3 (limiares do JSON)"
    pt_ref = np.r_[macro_auc_ap(Y, S3)[0], macro_auc_ap(Y, S3)[1], f1v(S3, lim3)]
    for n, (S, lim) in cand.items():
        au, ap = macro_auc_ap(Y, S)
        pt = np.r_[au, ap, f1v(S, lim)]
        for mi, met in enumerate(("auroc", "auprc", "f1")):
            for k, r in enumerate(ROTULOS + ["MACRO"]):
                sel = (lambda X: X[:, mi * 6 + k]) if k < 6 else (lambda X: np.nanmean(X[:, mi * 6:mi * 6 + 6], axis=1))
                val = pt[mi * 6 + k] if k < 6 else pt[mi * 6:mi * 6 + 6].mean()
                lo, hi = ic(sel(B[n]))
                l = dict(modelo=n, rotulo=r, metrica=met, valor=val, ic_inf=lo, ic_sup=hi)
                if n != ref:
                    d = sel(B[n]) - sel(B[ref])
                    dv = val - (pt_ref[mi * 6 + k] if k < 6 else pt_ref[mi * 6:mi * 6 + 6].mean())
                    l |= dict(dif_vs_A3=dv, dif_ic_inf=ic(d)[0], dif_ic_sup=ic(d)[1])
                linhas.append(l)
    tab = pd.DataFrame(linhas)
    salvar_novo(tab, RES / "sgpu_ensemble_A3_A6.csv")
    pd.set_option("display.width", 220)
    print(tab[tab.rotulo == "MACRO"].round(4).to_string(index=False))
    print(tab[(tab.modelo.str.startswith("ensemble")) & (tab.metrica == "auprc")].round(4).to_string(index=False))


def rf():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold

    from arvore_rf_pr import VARIANTES, ler
    cols = VARIANTES["A0e-1"]
    d = pd.concat([ler("treino"), ler("validacao")], ignore_index=True)
    assert set(d.grupo) == {"treino", "validação"}, set(d.grupo)  # nenhum exame do grupo teste
    X, Y, g = d[cols].to_numpy(), d[ROTULOS].to_numpy(int), d.patient_id.to_numpy()
    print(f"treino+validação: {len(d):,} exames, {len(np.unique(g)):,} pacientes, {len(cols)} atributos (A0e-1)", flush=True)
    linhas = []
    for f, (itr, iva) in enumerate(GroupKFold(n_splits=5).split(X, Y, g)):
        assert not set(g[itr]) & set(g[iva])
        t0 = time.perf_counter()
        for k, r in enumerate(ROTULOS):
            # hiperparâmetros do A0e-1 RF (ecg) escolhidos na validação em arvore_rf_pr.py (arvore_rf_pr_grade.csv: 300 árvores, folha mínima 5)
            m = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample", random_state=SEED, n_jobs=2).fit(X[itr], Y[itr, k])
            s = m.predict_proba(X[iva])[:, 1]
            linhas.append(dict(fold=f + 1, rotulo=r, auroc=roc_auc_score(Y[iva, k], s), auprc=average_precision_score(Y[iva, k], s),
                               n_val=len(iva), positivos=int(Y[iva, k].sum()), prevalencia=Y[iva, k].mean()))
        print(f"fold {f + 1}: {time.perf_counter() - t0:.0f} s | AUROC macro {np.mean([l['auroc'] for l in linhas[-6:]]):.4f} | AUPRC macro {np.mean([l['auprc'] for l in linhas[-6:]]):.4f}", flush=True)
    tab = pd.DataFrame(linhas)
    mac = tab.groupby("fold")[["auroc", "auprc"]].mean().reset_index().assign(rotulo="MACRO")
    tab = pd.concat([tab, mac], ignore_index=True)
    salvar_novo(tab, RES / "sgpu_rf_A0e1_groupkfold.csv")
    res = tab.groupby("rotulo", sort=False)[["auroc", "auprc"]].agg(["mean", "std"])
    salvar_novo(res.reset_index().set_axis(["rotulo", "auroc_media", "auroc_dp", "auprc_media", "auprc_dp"], axis=1), RES / "sgpu_rf_A0e1_groupkfold_resumo.csv")
    print(res.round(4).to_string())


def experimentos():
    """Acrescenta linhas NOVAS em resultados/experimentos.csv (aborta se o nome já existir; as linhas antigas ficam iguais).
    Limiar = F1 máximo na validação (como nas linhas do A3 no CSV)."""
    import json
    from a0_baselines import metricas
    from robustez_comum import RAIZ
    arq = RES / "experimentos.csv"
    antigo = pd.read_csv(arq)
    S3, Y, _, _, _ = carregar_val()
    S6 = np.load(RES / "A6_completo_previsoes_validacao.npy").astype(np.float64)

    def info_treino(nome):
        e = json.loads((RAIZ / "logs" / f"noite_{nome}_estado.json").read_text(encoding="utf-8"))
        h = e["historia"]
        return dict(epocas=len(h), melhor_epoca=1 + int(np.argmin([x["val_loss"] for x in h])), minutos_treino=sum(x["segundos"] for x in h) / 60), e["termino"]

    novos = {}
    it6, t6 = info_treino("A6_completo")
    novos[f"A6_completo CNN+LSTM (partes 0 a 3; noite 13/11; terminou por {t6})"] = (S6, it6)
    a5 = next(n for n in ("A5_completo", "A5_completo_parcial14", "A5_completo_parcial") if (RES / f"{n}_previsoes_validacao.npy").exists())
    it5, t5 = info_treino(a5)
    rot5 = "" if t5.startswith("early stopping") else " | PARCIAL, NÃO CONCLUSIVO (sem early stopping)"
    novos[f"{a5} ResNet pequena (partes 0 a 3; terminou por {t5}){rot5}"] = (np.load(RES / f"{a5}_previsoes_validacao.npy").astype(np.float64), it5)
    novos["ensemble A3+A6_completo (média das probabilidades; EXPLORATÓRIO, não oficial)"] = ((S3 + S6) / 2, dict(epocas=np.nan, melhor_epoca=np.nan, minutos_treino=np.nan))
    assert not set(novos) & set(antigo.experimento), "experimento já existe no CSV"
    linhas = []
    for n, (S, it) in novos.items():
        df = pd.DataFrame([dict(experimento=n, rotulo=r, n_val=len(Y), positivos_val=int(Y[:, k].sum()), **metricas(Y[:, k], S[:, k], melhor_limiar_f1(Y[:, k], S[:, k])))
                           for k, r in enumerate(ROTULOS)])
        mac = df[["auroc", "auprc", "sensibilidade", "precisao", "f1", "especificidade", "acuracia"]].mean()
        df = pd.concat([df, pd.DataFrame([dict(experimento=n, rotulo="MACRO", n_val=len(Y), **mac)])], ignore_index=True)
        linhas.append(df.assign(dados="partes 0 a 3", n_treino=55561, **it))
    novo = pd.concat([antigo] + linhas, ignore_index=True)
    assert novo.iloc[:len(antigo)].equals(antigo)
    novo.to_csv(arq, index=False, float_format="%.4f")
    print(pd.concat(linhas)[lambda d: d.rotulo == "MACRO"][["experimento", "auroc", "auprc", "f1", "epocas", "melhor_epoca"]].round(4).to_string(index=False))


if __name__ == "__main__":
    operacao2 = lambda: operacao("sessao2", ("A5_completo", "A5_completo_parcial14"),
                                 "- **`A5_completo_parcial14` = melhor checkpoint (época 9 de 10 fechadas) da retomada de 14/11 de manhã, cortada à mão: "
                                 "early stopping NÃO concluído, PARCIAL, NÃO CONCLUSIVO.** (Se existir `A5_completo`, é o A5 com early stopping completo e ele é usado no lugar.)")
    {"operacao": operacao, "operacao2": operacao2, "ensemble": ensemble, "rf": rf, "checar": checar, "experimentos": experimentos}[sys.argv[1]]()
