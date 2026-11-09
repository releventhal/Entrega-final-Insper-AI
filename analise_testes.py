"""Depois do teste (08/11): tabelas, análise POSTERIOR (descritiva, não muda nada) e figuras.
Só lê resultados/bruto_<conjunto>_*.npz (notas salvas pelo avaliar_externo.py), resultados/avaliacao_*.csv e o JSON congelado.
Nenhum modelo é rodado. O item (b) lê os sinais das partes 0 a 3 só para calcular o MD5 de cada exame (cache em code15/, fora do git).

Uso (.venv):
  python analise_testes.py tabela interno     -> tabela markdown do conjunto, lado a lado com a validação do A3
  python analise_testes.py posterior          -> (a) positivos previstos vs reais, (b) teste interno com/sem sinais idênticos
  python analise_testes.py figuras            -> figuras/i_testes_auroc.png, j_testes_auprc.png, k_roc_pr_ptbxl.png
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).parent
R, F = RAIZ / "resultados", RAIZ / "figuras"
ROTULOS = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF"]
CONJUNTOS = {"validacao_A3": "Validação A3", "interno": "Teste interno", "ptbxl": "PTB-XL", "code_test": "CODE-TEST"}
METRICAS = ("auroc", "auprc", "sensibilidade", "precisao", "f1", "especificidade")
LIMIARES = json.loads((RAIZ / "congelado_candidato_A3.json").read_text(encoding="utf-8"))["limiares"]["valores"]


def bruto(c):
    b = np.load(sorted(R.glob(f"bruto_{c}_*.npz"))[-1], allow_pickle=False)
    return {k: b[k] for k in b.files}


def br(x):
    return f"{x:.3f}".replace(".", ",")


def tabela(c):
    """Markdown: por alteração, métrica [IC 95%]; macro do conjunto e macro da validação do A3."""
    t = pd.read_csv(R / f"avaliacao_{c}.csv").set_index("rotulo")
    v = pd.read_csv(R / "avaliacao_validacao_A3.csv").set_index("rotulo")
    cab = "| Alteração | Positivos | " + " | ".join(["AUROC", "AUPRC", "Sensibilidade", "Precisão", "F1", "Especificidade"]) + " |"
    out = [cab, "|" + "---|" * 8]
    for r in ROTULOS + ["MACRO"]:
        x = t.loc[r]
        pos = "" if r == "MACRO" else f"{int(x.positivos)}"
        out.append(f"| {'Média macro' if r == 'MACRO' else r} | {pos} | " + " | ".join(
            f"{br(x[m])} [{br(x[m + '_ic_inf'])}-{br(x[m + '_ic_sup'])}]" for m in METRICAS) + " |")
    x = v.loc["MACRO"]
    out.append("| Validação A3, média macro (8.158 exames) | | " + " | ".join(
        f"{br(x[m])} [{br(x[m + '_ic_inf'])}-{br(x[m + '_ic_sup'])}]" for m in METRICAS) + " |")
    print("\n".join(out))


def comparacao():
    """Markdown: AUROC, AUPRC e F1 por alteração, validação vs cada teste disponível (diferença teste - validação)."""
    disp = [c for c in CONJUNTOS if (R / f"avaliacao_{c}.csv").exists()]
    d = {c: pd.read_csv(R / f"avaliacao_{c}.csv").set_index("rotulo") for c in disp}
    for m in ("auroc", "auprc", "f1"):
        print(f"\n{m.upper()} (diferença em relação à validação entre parênteses)\n")
        print("| Alteração | " + " | ".join(CONJUNTOS[c] for c in disp) + " |\n|" + "---|" * (len(disp) + 1))
        for r in ROTULOS + ["MACRO"]:
            base = d["validacao_A3"].loc[r, m]
            cel = [br(base)] + [f"{br(d[c].loc[r, m])} ({'+' if d[c].loc[r, m] - base >= 0 else ''}{br(d[c].loc[r, m] - base)})" for c in disp[1:]]
            print(f"| {'Média macro' if r == 'MACRO' else r} | " + " | ".join(cel) + " |")


def metricas_pontuais(Y, S, mask):
    from a0_baselines import metricas
    linhas = {r: metricas(Y[mask[:, k], k].astype(int), S[mask[:, k], k], LIMIARES[r]) for k, r in enumerate(ROTULOS)}
    linhas["MACRO"] = {m: float(np.mean([linhas[r][m] for r in ROTULOS])) for m in METRICAS}
    return pd.DataFrame(linhas).T.loc[:, list(METRICAS)]


def a_positivos_previstos():
    linhas = []
    for c in CONJUNTOS:
        if c == "validacao_A3":  # sem saída bruta no mesmo formato; a validação entra pelos CSVs
            continue
        b = bruto(c)
        for k, r in enumerate(ROTULOS):
            m = b["mask"][:, k]
            y, s = b["Y"][m, k], b["S"][m, k]
            prev = int((s >= LIMIARES[r]).sum())
            linhas.append(dict(conjunto=CONJUNTOS[c], rotulo=r, avaliados=int(m.sum()), positivos_reais=int(y.sum()),
                               prevalencia=y.mean(), positivos_previstos=prev, razao_previstos_reais=prev / max(y.sum(), 1)))
    df = pd.DataFrame(linhas)
    df.to_csv(R / "posterior_a_positivos_previstos.csv", index=False, float_format="%.4f")
    print("\n(a) positivos previstos com o limiar congelado vs positivos reais\n")
    print("| Conjunto | Alteração | Avaliados | Positivos reais (prevalência) | Previstos | Previstos / reais |\n|---|---|---|---|---|---|")
    for x in df.itertuples():
        print(f"| {x.conjunto} | {x.rotulo} | {x.avaliados} | {x.positivos_reais} ({100 * x.prevalencia:.1f}%) | {x.positivos_previstos} | "
              f"{x.razao_previstos_reais:.2f}".replace(".", ",") + " |")


def md5_partes():
    """MD5 do sinal bruto (float32 como gravado) de todos os exames das partes 0 a 3. Cache em code15/md5_p0-3.csv (fora do git)."""
    cache = RAIZ / "code15" / "md5_p0-3.csv"
    if cache.exists():
        return pd.read_csv(cache)
    import h5py
    out = []
    for p in (0, 1, 2, 3):
        with h5py.File(RAIZ / "code15" / f"exams_part{p}.hdf5", "r") as f:
            ds, ids = f["tracings"], f["exam_id"][:]
            for ini in range(0, len(ids), 500):
                for k, x in enumerate(ds[ini:ini + 500]):
                    out.append((int(ids[ini + k]), hashlib.md5(x.tobytes()).hexdigest()))
        print(f"md5: parte {p} pronta", flush=True)
    df = pd.DataFrame(out, columns=["exam_id", "md5"])
    df = df[df.exam_id != 0]  # linha de preenchimento
    df.to_csv(cache, index=False)
    return df


def b_identicos():
    from ecg_dados import tabela_com_grupo
    b = bruto("interno")
    h = md5_partes().merge(tabela_com_grupo()[["exam_id", "patient_id", "grupo"]], on="exam_id", how="left")
    cont = h.md5.value_counts()
    h["n_iguais"] = h.md5.map(cont)
    hi = h.set_index("exam_id").loc[b["ids"]]
    ident = hi.n_iguais.to_numpy() > 1
    # idêntico a algum exame de treino/validação (de outro grupo)
    md5_trva = set(h.loc[h.grupo != "teste", "md5"])
    ident_trva = hi.md5.isin(md5_trva).to_numpy()
    # idêntico a outro exame de OUTRO paciente (em qualquer grupo)
    pac_por_md5 = h.groupby("md5").patient_id.nunique()
    ident_outro_pac = (hi.md5.map(pac_por_md5).to_numpy() > 1)
    Y, S, M = b["Y"], b["S"], b["mask"].astype(bool)
    print(f"\n(b) teste interno: {len(Y)} exames; {ident.sum()} com sinal idêntico a outro exame das partes 0 a 3 "
          f"({ident_trva.sum()} idênticos a algum exame de treino/validação; {ident_outro_pac.sum()} cujo sinal também aparece em outro paciente); "
          f"com alguma das 6 alterações: {int((Y[ident].sum(1) > 0).sum())}")
    com, sem = metricas_pontuais(Y, S, M), metricas_pontuais(Y[~ident], S[~ident], M[~ident])
    sem_trva = metricas_pontuais(Y[~ident_trva], S[~ident_trva], M[~ident_trva])
    df = pd.concat({"todos": com, "sem_identicos": sem, "sem_identicos_a_treino_validacao": sem_trva}, names=["versao", "rotulo"])
    df.to_csv(R / "posterior_b_interno_identicos.csv", float_format="%.4f")
    print("\n| Alteração | AUROC com / sem | AUPRC com / sem | F1 com / sem |\n|---|---|---|---|")
    for r in ROTULOS + ["MACRO"]:
        print(f"| {'Média macro' if r == 'MACRO' else r} | " + " | ".join(f"{br(com.loc[r, m])} / {br(sem.loc[r, m])}" for m in ("auroc", "auprc", "f1")) + " |")
    print(f"\nmaior diferença absoluta (com - sem) em qualquer métrica e alteração: {np.abs(com.values - sem.values).max():.4f}; "
          f"sem os idênticos a treino/validação: {np.abs(com.values - sem_trva.values).max():.4f}")


def d_codetest_anotadores():
    """Contexto descritivo: F1 contra o gold_standard de cada anotação que vem no zip do CODE-TEST e do A3 (limiar congelado)."""
    from ecg_dados import CODE_TEST
    b = bruto("code_test")
    linhas = b["ids"]
    gold = pd.read_csv(CODE_TEST / "annotations" / "gold_standard.csv")[ROTULOS].to_numpy()[linhas]
    assert (gold == b["Y"]).all()
    quem = {"A3 (este projeto)": (b["S"] >= np.array([LIMIARES[r] for r in ROTULOS])).astype(int)}
    for nome in ("dnn", "cardiologist1", "cardiologist2", "cardiology_residents", "emergency_residents", "medical_students"):
        quem[nome] = pd.read_csv(CODE_TEST / "annotations" / f"{nome}.csv")[ROTULOS].to_numpy()[linhas]
    out = {}
    for nome, P in quem.items():
        tp, fp, fn = ((P == 1) & (gold == 1)).sum(0), ((P == 1) & (gold == 0)).sum(0), ((P == 0) & (gold == 1)).sum(0)
        out[nome] = list(2 * tp / np.maximum(2 * tp + fp + fn, 1)) + [float(np.mean(2 * tp / np.maximum(2 * tp + fp + fn, 1)))]
    df = pd.DataFrame(out, index=ROTULOS + ["MACRO"]).T
    df.to_csv(R / "posterior_d_codetest_anotadores_f1.csv", float_format="%.4f")
    print("\n(d) CODE-TEST, F1 contra o gold_standard (sem IC; 13 a 37 positivos por alteração)\n")
    print("| Quem | " + " | ".join(ROTULOS) + " | Média macro |\n|" + "---|" * 8)
    for nome, x in df.iterrows():
        print(f"| {nome} | " + " | ".join(br(v) for v in x) + " |")


def figuras():
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    sys.path.insert(0, str(RAIZ))
    from figuras import CORES, salvar  # mesmo estilo e cores por alteração das figuras da validação
    cores_conj = {"validacao_A3": "#d55e00", "interno": "#0072b2", "ptbxl": "#009e73", "code_test": "#cc79a7"}
    disp = [c for c in CONJUNTOS if (R / f"avaliacao_{c}.csv").exists()]
    d = {c: pd.read_csv(R / f"avaliacao_{c}.csv").set_index("rotulo") for c in disp}
    for met, nome, arq in (("auroc", "AUROC", "i_testes_auroc.png"), ("auprc", "AUPRC", "j_testes_auprc.png")):
        fig, ax = plt.subplots(figsize=(14, 6))
        x, larg = np.arange(7), 0.8 / len(disp)
        for i, c in enumerate(disp):
            v = d[c].loc[ROTULOS + ["MACRO"]]
            err = np.vstack([v[met] - v[f"{met}_ic_inf"], v[f"{met}_ic_sup"] - v[met]])
            ax.bar(x + (i - (len(disp) - 1) / 2) * larg, v[met], larg, yerr=err, capsize=3, color=cores_conj[c], label=CONJUNTOS[c])
        ax.set_xticks(x, ROTULOS + ["Média macro"])
        ax.set_ylim(0, 1.12 if met == "auprc" else 1.05)
        if met == "auroc":
            ax.set_ylim(0.5, 1.02)
        ax.set_ylabel(nome + (" (eixo começa em 0,5)" if met == "auroc" else ""))
        ax.set_title(f"{nome} por alteração: validação do A3 contra cada teste (barras = IC 95%)")
        ax.legend(ncol=len(disp), loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
        salvar(fig, arq)
    if "ptbxl" not in disp:
        return
    b = bruto("ptbxl")
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.5))
    for k, r in enumerate(ROTULOS):
        m = b["mask"][:, k].astype(bool)
        y, s = b["Y"][m, k], b["S"][m, k]
        fpr, tpr, _ = roc_curve(y, s)
        pr, rc, _ = precision_recall_curve(y, s)
        p = s >= LIMIARES[r]
        axs[0].plot(fpr, tpr, color=CORES[r], lw=2, label=f"{r} (AUROC {br(d['ptbxl'].loc[r, 'auroc'])})")
        axs[0].plot(1 - (~p[y == 0]).mean(), p[y == 1].mean(), "o", color=CORES[r], ms=9, mec="black")
        axs[1].plot(rc, pr, color=CORES[r], lw=2, label=f"{r} (AUPRC {br(d['ptbxl'].loc[r, 'auprc'])})")
        if p.any():
            axs[1].plot(p[y == 1].mean(), y[p].mean(), "o", color=CORES[r], ms=9, mec="black")
    axs[0].plot([0, 1], [0, 1], "--", color="#999999", lw=1)
    axs[0].set(xlabel="1 - especificidade", ylabel="sensibilidade", title="PTB-XL: curvas ROC")
    axs[1].set(xlabel="sensibilidade (recall)", ylabel="precisão", title="PTB-XL: curvas Precisão-Recall", ylim=(0, 1.02))
    for ax in axs:
        ax.legend(loc="lower right" if ax is axs[0] else "lower left", fontsize=11)
    fig.suptitle("Ponto = limiar congelado (escolhido na validação)", fontsize=13, y=0.0)
    salvar(fig, "k_roc_pr_ptbxl.png")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "tabela":
        tabela(sys.argv[2])
    elif cmd == "comparacao":
        comparacao()
    elif cmd == "posterior":
        a_positivos_previstos()
        d_codetest_anotadores()
        b_identicos()
    elif cmd == "figuras":
        figuras()
