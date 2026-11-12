"""Fase 1 de robustez (12/11): A0c = árvore de decisão e A0d = random forest, sobre os atributos de `atributos_sinal.py`.
Degraus novos da escadinha: A0b (idade+sexo) < A0c árvore < A0d floresta < A1-A3 (CNN). NÃO altera o A3. (O nome A4 já é a CNN com
perda ponderada, nunca treinada; por isso A0c/A0d.) TÉCNICA FORA DAS AULAS: árvore, floresta, export_text, plot_tree, importância de atributos.
Só TREINO -> VALIDAÇÃO (tabelas de atributos de resultados/); o teste não é aberto. Limiar por alteração = F1 máximo na validação (como o A3).
Hiperparâmetros (um conjunto por modelo e versão de atributos, igual para as 6 alterações) = o de maior AUPRC macro na validação.
Isto torna a validação otimista (escolhe hiperparâmetros e limiares nela), como no A3.

Uso (.venv, na raiz; precisa de resultados/atributos_{treino,validacao}.csv.gz, gerados por `atributos_sinal.py tudo`):
  python arvore_rf.py
"""
import hashlib
import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from a0_baselines import PASTA_RESULTADOS, melhor_limiar_f1
from a1_cnn import gravar_no_csv, tabela_metricas
from atributos_sinal import NOMES
from avaliar_externo import bootstrap
from ecg_dados import ROTULOS, SEED

RAIZ = Path(__file__).parent
PASTA_MODELOS, PASTA_FIGURAS = RAIZ / "modelos", RAIZ / "figuras"
VERSOES = {"ecg": NOMES, "ecg+idade+sexo": NOMES + ["age", "is_male"]}
GRADES = {"arvore": [dict(max_depth=d) for d in (3, 4, 5)],
          "rf": [dict(n_estimators=n, min_samples_leaf=l) for n, l in product((100, 300, 500), (1, 5, 20))]}
EXPERIMENTO = {"arvore": "A0c árvore de decisão (atributos do sinal", "rf": "A0d random forest (atributos do sinal"}
ARQ = {"arvore": "A0c_arvore", "rf": "A0d_rf"}


def novo(fam, cfg):
    if fam == "arvore":
        return DecisionTreeClassifier(**cfg, random_state=SEED)
    return RandomForestClassifier(**cfg, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)


def ajustar(fam, cfg, Xtr, Ytr, Xva):
    """Um modelo por alteração. Devolve ({rótulo: modelo}, notas (n_val, 6))."""
    modelos = {r: novo(fam, cfg).fit(Xtr, Ytr[:, k]) for k, r in enumerate(ROTULOS)}
    return modelos, np.column_stack([modelos[r].predict_proba(Xva)[:, 1] for r in ROTULOS])


def sha256(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def divisoes_fc(m, nomes):
    """Cortes da árvore sobre FC/RR, já em bpm (RR em ms -> 60000/RR). Regra: vai para a esquerda se atributo <= corte."""
    t, saida = m.tree_, []
    for i in range(t.node_count):
        f = nomes[t.feature[i]] if t.feature[i] >= 0 else None
        if f in ("fc_bpm", "rr_media_ms", "rr_mediana_ms"):
            saida.append((f, round(float(t.threshold[i]), 1), round(float(t.threshold[i]) if f == "fc_bpm" else 60000 / t.threshold[i], 1)))
    return saida


def main():
    tr, va = (pd.read_csv(PASTA_RESULTADOS / f"atributos_{n}.csv.gz") for n in ("treino", "validacao"))
    assert not set(tr.patient_id) & set(va.patient_id) and set(tr.grupo) == {"treino"} and set(va.grupo) == {"validação"}
    Ytr, Yva = tr[ROTULOS].to_numpy(int), va[ROTULOS].to_numpy(int)
    print(f"treino {len(tr):,} | validação {len(va):,} exames (teste não aberto) | sklearn {sklearn.__version__}", flush=True)

    grade, finais, partes_csv, ic_linhas, regras = [], {}, [], [], []
    for fam, (versao, cols) in product(GRADES, VERSOES.items()):
        melhor = None
        for cfg in GRADES[fam]:
            t0 = time.perf_counter()
            modelos, S = ajustar(fam, cfg, tr[cols].to_numpy(), Ytr, va[cols].to_numpy())
            ap = [average_precision_score(Yva[:, k], S[:, k]) for k in range(6)]
            grade.append(dict(familia=fam, atributos=versao, **cfg, auprc_macro=np.mean(ap), segundos=round(time.perf_counter() - t0, 1),
                              **{f"auprc_{r}": a for r, a in zip(ROTULOS, ap)}))
            print(f"{fam} | {versao} | {cfg} | AUPRC macro {np.mean(ap):.4f} | {grade[-1]['segundos']} s", flush=True)
            if melhor is None or np.mean(ap) > melhor[0]:
                melhor = (np.mean(ap), cfg, modelos, S)
        _, cfg, modelos, S = melhor
        finais[(fam, versao)] = (cfg, modelos, S, cols)
    pd.DataFrame(grade).to_csv(PASTA_RESULTADOS / "arvore_rf_grade.csv", index=False, float_format="%.4f")

    PASTA_MODELOS.mkdir(exist_ok=True)
    cand = {"status": "candidato", "aviso": "CANDIDATO, NAO congelado. Nenhum conjunto de teste foi aberto. Congelar = decisao da dupla; avaliar_externo.py NAO conhece este tipo de modelo (ver relatorio).",
            "gerado_em": time.strftime("%Y-%m-%d"), "commit_base": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=RAIZ).stdout.strip(),
            "pacotes": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__, "scipy": scipy.__version__, "numpy": np.__version__,
                        "pandas": pd.__version__, "joblib": joblib.__version__},
            "rotulos": ROTULOS, "seed": SEED,
            "entrada": "atributos_sinal.atributos(sinal CRU 4096x12, sem normalizar, ordem DI,DII,DIII,AVR,AVL,AVF,V1-V6); NaN aceito; versao 'ecg+idade+sexo' soma age e is_male (is_male 0/1)",
            "dados_de_treino": {"partes": [0, 1, 2, 3], "n_treino": len(tr), "n_validacao": len(va),
                                "regra_de_exclusao": "so os exames com as 12 derivacoes zeradas (mesma do A3); parciais ficam, com NaN onde o atributo nao existe"},
            "limiar": "por alteracao, F1 maximo na validacao (a0_baselines.melhor_limiar_f1)", "modelos": {}}
    for (fam, versao), (cfg, modelos, S, cols) in finais.items():
        nome = f"{EXPERIMENTO[fam]}{'' if versao == 'ecg' else ' + idade + sexo'})"
        lims = {r: melhor_limiar_f1(Yva[:, k], S[:, k]) for k, r in enumerate(ROTULOS)}
        df = tabela_metricas(nome, Yva, S, lims)
        df["dados"] = f"partes [0, 1, 2, 3] (provisório); atributos: {versao}; {cfg}"
        df["n_treino"] = len(tr)
        partes_csv.append(df)
        arq = PASTA_MODELOS / f"{ARQ[fam]}_{versao.replace('+', '_')}.joblib"
        joblib.dump({"modelos": modelos, "atributos": cols}, arq, compress=3)
        cand["modelos"][f"{ARQ[fam]}_{versao}"] = dict(arquivo=str(arq.relative_to(RAIZ)).replace("\\", "/"), sha256=sha256(arq), tamanho_mb=round(arq.stat().st_size / 2**20, 1),
                                                       hiperparametros=cfg, atributos=cols, limiares=lims)
        ic, n_ok = bootstrap(Yva.astype(float), S, lims, np.ones(Yva.shape, bool), va.patient_id.to_numpy())
        for r, m in df[["rotulo"] + ["auroc", "auprc", "sensibilidade", "precisao", "f1", "especificidade"]].set_index("rotulo").iterrows():
            for c, v in m.items():
                ic_linhas.append(dict(experimento=nome, rotulo=r, metrica=c, valor=v, ic_inf=ic[(r, c)][0], ic_sup=ic[(r, c)][1]))
        print(f"\n{nome} | {cfg} | macro AUROC {df.iloc[-1].auroc:.4f} AUPRC {df.iloc[-1].auprc:.4f} F1 {df.iloc[-1].f1:.4f} | bootstrap por paciente: {n_ok} reamostras válidas", flush=True)

        if fam == "arvore":
            for r in ROTULOS:
                m = modelos[r]
                txt = export_text(m, feature_names=cols, decimals=2)
                regras.append(f"=== {nome} | {r} | max_depth={cfg['max_depth']} (vai à esquerda se atributo <= corte) ===\n{txt}")
                fig, ax = plt.subplots(figsize=(24, 10))
                plot_tree(m, feature_names=cols, class_names=["não", "sim"], filled=True, impurity=False, proportion=True, fontsize=7, ax=ax)
                ax.set_title(f"{nome} | {r} | max_depth={cfg['max_depth']}")
                fig.savefig(PASTA_FIGURAS / f"arvore_{r}_{versao.replace('+', '_')}.png", dpi=90, bbox_inches="tight")
                plt.close(fig)
            print(f"cortes de FC/RR na árvore de SB ({versao}) (atributo, corte, bpm equivalente):", divisoes_fc(modelos["SB"], cols))
        else:
            imp = pd.DataFrame({r: modelos[r].feature_importances_ for r in ROTULOS}, index=cols)
            imp.to_csv(PASTA_RESULTADOS / f"rf_importancia_{versao.replace('+', '_')}.csv", float_format="%.4f")
            print("importância (Gini) — 5 maiores por alteração:")
            for r in ROTULOS:
                print(f"  {r}: " + ", ".join(f"{a} {b:.3f}" for a, b in imp[r].nlargest(5).items()))

    (PASTA_RESULTADOS / "arvore_regras.txt").write_text("\n\n".join(regras), encoding="utf-8")
    print("\n" + "\n\n".join(regras))
    pd.DataFrame(ic_linhas).to_csv(PASTA_RESULTADOS / "arvore_rf_ic.csv", index=False, float_format="%.4f")
    np.savez_compressed(PASTA_RESULTADOS / "arvore_rf_previsoes_validacao.npz", exam_id=va.exam_id.to_numpy(),
                        **{f"{ARQ[f]}_{v}": S for (f, v), (_, _, S, _) in finais.items()})
    gravar_no_csv(pd.concat(partes_csv, ignore_index=True))
    (RAIZ / "congelado_candidato_arvore_rf.json").write_text(json.dumps(cand, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nok: resultados/experimentos.csv (linhas novas), arvore_rf_{grade,ic}.csv, arvore_regras.txt, rf_importancia_*.csv, figuras/arvore_*.png, congelado_candidato_arvore_rf.json")


if __name__ == "__main__":
    main()
