"""A0e (13/11): árvore e floresta COM o intervalo PR (p_r_peak do neurokit2), em duas variantes. Estrutura, grade e critérios de `arvore_rf.py` (Fase 1):
hiperparâmetros (um conjunto por modelo/versão/variante, igual para as 6 alterações) = maior AUPRC macro na validação; limiar por alteração = F1 máximo na
validação. Isso torna a validação OTIMISTA (como A0c, A0d e A3). Só treino -> validação; o teste NÃO é aberto.
  A0e-1 = os 22 atributos da Fase 1 (NOMES) + p_r_peak
  A0e-2 = A0e-1 + qrs_dwt + qt_dwt  (mesma delineação ainda não validada do pr_dwt, por isso é uma variante à parte)
NÃO usa pr_dwt. Lê resultados/atributos_{treino,validacao}.csv.gz (Fase 1) e resultados/atributos_pr_{treino,validacao}.csv.gz (atributos_pr.py).
NÃO grava em experimentos.csv, não salva modelos e não mexe em nenhum arquivo da Fase 1/2 nem no A3. Saídas novas (resultados/arvore_rf_pr_*):
  grade.csv, metricas.csv (ponto, por modelo e alteração), ic.csv (IC por bootstrap de pacientes; diferenças pareadas), regras.txt, importancia_*.csv, previsoes_validacao.npz
Uso: python arvore_rf_pr.py
"""
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text

from a0_baselines import melhor_limiar_f1
from atributos_sinal import NOMES
from ecg_dados import ROTULOS, SEED
from robustez_comum import carregar_val, reamostras, salvar_novo

RAIZ = Path(__file__).resolve().parent
RES = RAIZ / "resultados"
VARIANTES = {"A0e-1": NOMES + ["p_r_peak"], "A0e-2": NOMES + ["p_r_peak", "qrs_dwt", "qt_dwt"]}
VERSOES = {"ecg": [], "ecg+idade+sexo": ["age", "is_male"]}
GRADES = {"arvore": [dict(max_depth=d) for d in (3, 4, 5)],  # as mesmas de arvore_rf.py
          "rf": [dict(n_estimators=n, min_samples_leaf=l) for n, l in product((100, 300, 500), (1, 5, 20))]}


def novo(fam, cfg):
    if fam == "arvore":
        return DecisionTreeClassifier(**cfg, random_state=SEED)
    return RandomForestClassifier(**cfg, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)


def ajustar(fam, cfg, Xtr, Ytr, Xva):
    modelos = {r: novo(fam, cfg).fit(Xtr, Ytr[:, k]) for k, r in enumerate(ROTULOS)}
    return modelos, np.column_stack([modelos[r].predict_proba(Xva)[:, 1] for r in ROTULOS])


def ler(nome):
    a = pd.read_csv(RES / f"atributos_{nome}.csv.gz")
    b = pd.read_csv(RES / f"atributos_pr_{nome}.csv.gz")
    assert (a.exam_id.to_numpy() == b.exam_id.to_numpy()).all(), "ordem/conjunto de exames difere entre Fase 1 e atributos_pr"
    return pd.concat([a, b[["p_r_peak", "qrs_dwt", "qt_dwt"]]], axis=1)


def boot_metricas(Y, S, lim, idxs):
    """AUROC e AUPRC (n_boot, 6) para um modelo; reamostras de pacientes."""
    au, ap = np.full((len(idxs), 6), np.nan), np.full((len(idxs), 6), np.nan)
    for b, ix in enumerate(idxs):
        for k in range(6):
            y = Y[ix, k]
            if y.min() != y.max():
                au[b, k], ap[b, k] = roc_auc_score(y, S[ix, k]), average_precision_score(y, S[ix, k])
    return au, ap


def main():
    tr, va = ler("treino"), ler("validacao")
    assert not set(tr.patient_id) & set(va.patient_id) and set(tr.grupo) == {"treino"} and set(va.grupo) == {"validação"}
    Ytr, Yva = tr[ROTULOS].to_numpy(int), va[ROTULOS].to_numpy(int)
    print(f"treino {len(tr):,} | validação {len(va):,} exames (teste não aberto)", flush=True)
    print("NaN de p_r_peak / qrs_dwt / qt_dwt (treino, validação):", tr[["p_r_peak", "qrs_dwt", "qt_dwt"]].isna().mean().round(4).tolist(), va[["p_r_peak", "qrs_dwt", "qt_dwt"]].isna().mean().round(4).tolist(), flush=True)

    grade, finais = [], {}
    for var, (fam, (versao, extra)) in product(VARIANTES, product(GRADES, VERSOES.items())):
        cols = VARIANTES[var] + extra
        melhor = None
        for cfg in GRADES[fam]:
            t0 = time.perf_counter()
            modelos, S = ajustar(fam, cfg, tr[cols].to_numpy(), Ytr, va[cols].to_numpy())
            ap = [average_precision_score(Yva[:, k], S[:, k]) for k in range(6)]
            grade.append(dict(variante=var, familia=fam, atributos=versao, **cfg, auprc_macro=np.mean(ap), segundos=round(time.perf_counter() - t0, 1), **{f"auprc_{r}": a for r, a in zip(ROTULOS, ap)}))
            print(f"{var} | {fam} | {versao} | {cfg} | AUPRC macro {np.mean(ap):.4f} | {grade[-1]['segundos']} s", flush=True)
            if melhor is None or np.mean(ap) > melhor[0]:
                melhor = (np.mean(ap), cfg, modelos, S)
        finais[(var, fam, versao)] = (melhor[1], melhor[2], melhor[3], cols)
    salvar_novo(pd.DataFrame(grade), RES / "arvore_rf_pr_grade.csv")

    # ---- previsões: A0e (aqui) + A0c/A0d (Fase 1, salvas) + A3 (salvas), alinhadas por exam_id
    f1 = np.load(RES / "arvore_rf_previsoes_validacao.npz")
    assert (f1["exam_id"] == va.exam_id.to_numpy()).all(), "ordem da validação da Fase 1 difere"
    P = {"A0c árvore": f1["A0c_arvore_ecg"], "A0d RF": f1["A0d_rf_ecg"], "A0c árvore +idade+sexo": f1["A0c_arvore_ecg+idade+sexo"], "A0d RF +idade+sexo": f1["A0d_rf_ecg+idade+sexo"]}
    nomes_fam = {"arvore": "árvore", "rf": "RF"}
    for (var, fam, versao), (cfg, _, S, _) in finais.items():
        P[f"{var} {nomes_fam[fam]}" + ("" if versao == "ecg" else " +idade+sexo")] = S
    S3, Y3, _, _, pid3 = carregar_val()
    assert (pid3 == va.patient_id.to_numpy()).all() and (Y3 == Yva).all(), "A3 salvo não alinha com a validação"
    P["A3 (CNN)"] = S3
    np.savez_compressed(RES / "arvore_rf_pr_previsoes_validacao.npz", exam_id=va.exam_id.to_numpy(), **{k.replace(" ", "_"): v for k, v in P.items()})

    # ---- métricas pontuais (limiar F1 máximo na validação; A3 com o do JSON não é necessário aqui: só AUROC/AUPRC/F1 com o limiar F1-máx de cada um)
    lims = {m: [melhor_limiar_f1(Yva[:, k], S[:, k]) for k in range(6)] for m, S in P.items()}
    linhas = []
    for m, S in P.items():
        for k, r in enumerate(ROTULOS):
            linhas.append(dict(modelo=m, rotulo=r, auroc=roc_auc_score(Yva[:, k], S[:, k]), auprc=average_precision_score(Yva[:, k], S[:, k]),
                               f1=f1_score(Yva[:, k], S[:, k] >= lims[m][k]), limiar=lims[m][k]))
    pt = pd.DataFrame(linhas)
    salvar_novo(pt, RES / "arvore_rf_pr_metricas.csv")

    # ---- bootstrap por paciente (1.000 reamostras, seed 42), pareado entre modelos
    comp = ["A0c árvore", "A0d RF", "A0e-1 árvore", "A0e-1 RF", "A0e-2 árvore", "A0e-2 RF", "A3 (CNN)"]
    idxs = reamostras(va.patient_id.to_numpy(), 1000, 42)
    B = {}
    for m in comp:
        t0 = time.perf_counter()
        B[m] = boot_metricas(Yva, P[m], None, idxs)
        print(f"bootstrap {m}: {time.perf_counter() - t0:.0f} s", flush=True)
    ic_l = []
    pares = [("A0e-1 árvore", "A0c árvore"), ("A0e-2 árvore", "A0c árvore"), ("A0e-2 árvore", "A0e-1 árvore"),
             ("A0e-1 RF", "A0d RF"), ("A0e-2 RF", "A0d RF"), ("A0e-2 RF", "A0e-1 RF"), ("A3 (CNN)", "A0e-1 RF"), ("A3 (CNN)", "A0e-2 RF")]
    for m in comp:
        for k, r in enumerate(ROTULOS + ["macro"]):
            for mi, nm in enumerate(("auroc", "auprc")):
                v = B[m][mi][:, k] if k < 6 else np.nanmean(B[m][mi], axis=1)
                ponto = pt[(pt.modelo == m) & (pt.rotulo == r)][nm].iloc[0] if k < 6 else pt[(pt.modelo == m)][nm].mean()
                ic_l.append(dict(tipo="modelo", a=m, b="", rotulo=r, metrica=nm, valor=ponto, ic_inf=np.nanpercentile(v, 2.5), ic_sup=np.nanpercentile(v, 97.5)))
    for a, b in pares:
        for k, r in enumerate(ROTULOS + ["macro"]):
            for mi, nm in enumerate(("auroc", "auprc")):
                va_, vb_ = (B[a][mi][:, k], B[b][mi][:, k]) if k < 6 else (np.nanmean(B[a][mi], axis=1), np.nanmean(B[b][mi], axis=1))
                d = va_ - vb_
                ponto = (pt[(pt.modelo == a) & (pt.rotulo == r)][nm].iloc[0] - pt[(pt.modelo == b) & (pt.rotulo == r)][nm].iloc[0]) if k < 6 else (pt[pt.modelo == a][nm].mean() - pt[pt.modelo == b][nm].mean())
                ic_l.append(dict(tipo="diferença a-b", a=a, b=b, rotulo=r, metrica=nm, valor=ponto, ic_inf=np.nanpercentile(d, 2.5), ic_sup=np.nanpercentile(d, 97.5)))
    ic = pd.DataFrame(ic_l)
    salvar_novo(ic, RES / "arvore_rf_pr_ic.csv")

    # ---- regras das árvores (texto) e importâncias das florestas
    regras = []
    for (var, fam, versao), (cfg, modelos, S, cols) in finais.items():
        if fam == "arvore":
            for r in ROTULOS:
                regras.append(f"=== {var} árvore | {versao} | {r} | {cfg} (esquerda se atributo <= corte) ===\n{export_text(modelos[r], feature_names=cols, decimals=2)}")
        else:
            imp = pd.DataFrame({r: modelos[r].feature_importances_ for r in ROTULOS}, index=cols)
            salvar_novo(imp.reset_index().rename(columns={"index": "atributo"}), RES / f"arvore_rf_pr_importancia_{var}_{versao.replace('+', '_')}.csv")
            print(f"importância (Gini) 1dAVb | {var} | {versao} | {cfg}: " + ", ".join(f"{a} {b:.3f}" for a, b in imp["1dAVb"].nlargest(5).items()))
    arq = RES / "arvore_rf_pr_regras.txt"
    assert not arq.exists()
    arq.write_text("\n\n".join(regras), encoding="utf-8")
    print("hiperparâmetros escolhidos:", {k: v[0] for k, v in finais.items()})
    pd.set_option("display.width", 220)
    print("\nMACRO e 1dAVb (ponto; IC 95% por bootstrap de pacientes):")
    for m in comp:
        for r in ("1dAVb", "macro"):
            s = ic[(ic.tipo == "modelo") & (ic.a == m) & (ic.rotulo == r)].set_index("metrica")
            f = pt[(pt.modelo == m) & (pt.rotulo == r)].f1.iloc[0] if r != "macro" else pt[pt.modelo == m].f1.mean()
            print(f"{m:14s} {r:6s} AUROC {s.loc['auroc','valor']:.4f} ({s.loc['auroc','ic_inf']:.4f}-{s.loc['auroc','ic_sup']:.4f})  AUPRC {s.loc['auprc','valor']:.4f} ({s.loc['auprc','ic_inf']:.4f}-{s.loc['auprc','ic_sup']:.4f})  F1 {f:.4f}")
    print("\nDIFERENÇAS pareadas a-b, 1dAVb e macro:")
    for a, b in pares:
        for r in ("1dAVb", "macro"):
            s = ic[(ic.tipo == "diferença a-b") & (ic.a == a) & (ic.b == b) & (ic.rotulo == r)].set_index("metrica")
            print(f"{a:12s} - {b:12s} {r:6s} AUROC {s.loc['auroc','valor']:+.4f} ({s.loc['auroc','ic_inf']:+.4f} a {s.loc['auroc','ic_sup']:+.4f})  AUPRC {s.loc['auprc','valor']:+.4f} ({s.loc['auprc','ic_inf']:+.4f} a {s.loc['auprc','ic_sup']:+.4f})")


if __name__ == "__main__":
    main()
