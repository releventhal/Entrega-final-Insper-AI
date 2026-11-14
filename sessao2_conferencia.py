"""Sessão 2 (14/11, tarde): confere os números da Parte A citados em roteiro_slides.md (slides 7 a 10, e o acréscimo de 14/11 do slide 12)
e no CLAUDE.md contra os arquivos de resultado. Só LÊ resultados já salvos (validação e as saídas dos testes já abertos em 08/11);
não abre teste nenhum, não roda modelo. Saída nova: resultados/sessao2_conferencia_numeros_parte_a.csv (aborta se já existe).

  python sessao2_conferencia.py
"""
import numpy as np
import pandas as pd

from robustez_comum import RES, carregar_val, salvar_novo
from sklearn.metrics import precision_recall_curve

EXP = pd.read_csv(RES / "experimentos.csv")
A3 = "A3 provisório (partes 0 a 3; normalização por exame e derivação)"
NOMES = {"A0a": "A0a sempre negativo", "A0b": "A0b idade+sexo", "A0b_A1": "A0b idade+sexo (mesmo conjunto do A1)",
         "A0b_A3": "A0b idade+sexo (mesmo conjunto do A3)", "A1": "A1 CNN (provisório, parte 0)",
         "A2": "A2 provisório (normalização por exame e derivação)", "A3": A3,
         "A3_2069": A3 + " | só os 2069 exames de validação da parte 0 (limiar da validação completa)",
         "A3_2069r": A3 + " | só os 2069 exames de validação da parte 0 (limiar reescolhido neles, como no A1/A2)",
         "A0c": "A0c árvore de decisão (atributos do sinal)", "A0d": "A0d random forest (atributos do sinal)",
         "A0e1": "A0e-1 random forest (atributos do sinal + p_r_peak)",
         "A6": "A6_completo CNN+LSTM (partes 0 a 3; noite 13/11; terminou por early stopping (paciência 3))",
         "ens": "ensemble A3+A6_completo (média das probabilidades; EXPLORATÓRIO, não oficial)"}
F_EXP = "resultados/experimentos.csv"


def e(m, rot, met):
    return F_EXP, EXP[(EXP.experimento == NOMES[m]) & (EXP.rotulo == rot)][met].item()


def seeds(met):
    v = [EXP[(EXP.experimento == n) & (EXP.rotulo == "MACRO")][met].item() for n in (A3, A3 + " | seed 43", A3 + " | seed 44")]
    return np.mean(v), np.std(v, ddof=1)


def t(conj, rot, met):
    f = f"resultados/avaliacao_{conj}_20261108_466e7a3.csv"
    d = pd.read_csv(RES.parent / f)
    return f, d[d.rotulo == rot][met].item()


def ic_rf(a, rot, met, col="valor"):
    f = "resultados/arvore_rf_pr_ic.csv"
    d = pd.read_csv(RES.parent / f)
    return f, d[(d.tipo == "modelo") & (d.a == a) & (d.rotulo == rot) & (d.metrica == met)][col].item()


def op(rot, met, f="resultados/sgpu_pontos_operacao.csv"):
    d = pd.read_csv(RES.parent / f)
    return f, d[(d.modelo == "A3") & (d.rotulo == rot) & (d.metrica == met)].valor.item()


def a5(met):
    f = "resultados/noite_A5_completo_parcial14_avaliacao.csv"
    d = pd.read_csv(RES.parent / f)
    return f, d[(d.rotulo == "MACRO") & (d.metrica == met)].novo.item()


def kfold(col):
    f = "resultados/sgpu_rf_A0e1_groupkfold_resumo.csv"
    d = pd.read_csv(RES.parent / f)
    return f, d[d.rotulo == "MACRO"][col].item()


def prec_max_1davb():
    S, Y, *_ = carregar_val()
    p, r, _ = precision_recall_curve(Y[:, 0], S[:, 0])
    return "resultados/erros_A3/previsoes_e_rotulos.npy (recalculado: precisão máxima com sensibilidade >= 0,5)", p[r >= 0.5].max()


R = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF"]
L = []  # (onde, item, citado, função que devolve (arquivo, valor))
add = lambda onde, item, citado, f: L.append((onde, item, citado, f))

# --- slides
S7, S8, S9, S10, S12 = (f"roteiro_slides.md, slide {n}" for n in (7, 8, 9, 10, "12 (acréscimo de 14/11)"))
add(S7, "A0a n_val", "8.180", lambda: e("A0a", "MACRO", "n_val"))
add(S7, "A0a acurácia macro", "0,980", lambda: e("A0a", "MACRO", "acuracia"))
add(S7, "A0a sensibilidade macro", "0", lambda: e("A0a", "MACRO", "sensibilidade"))
add(S7, "A0a AUROC macro", "0,500", lambda: e("A0a", "MACRO", "auroc"))
add(S8, "A3 n_val", "8.158", lambda: e("A3", "MACRO", "n_val"))
add(S8, "A3 AUROC macro", "0,989", lambda: e("A3", "MACRO", "auroc"))
add(S8, "A0b AUROC macro (8.158)", "0,705", lambda: e("A0b_A3", "MACRO", "auroc"))
add(S8, "A3 AUPRC macro", "0,691", lambda: e("A3", "MACRO", "auprc"))
add(S8, "A0b AUPRC macro (8.158)", "0,046", lambda: e("A0b_A3", "MACRO", "auprc"))
add(S8, "A3 AUPRC 1dAVb", "0,44", lambda: e("A3", "1dAVb", "auprc"))
for m, au, ap in (("A0c", "0,911", "0,402"), ("A0d", "0,943", "0,531"), ("A0e1", "0,976", "0,567")):
    add(S8, f"{m} AUROC macro", au, lambda m=m: e(m, "MACRO", "auroc"))
    add(S8, f"{m} AUPRC macro", ap, lambda m=m: e(m, "MACRO", "auprc"))
add(S8, "A0d AUROC 1dAVb", "0,749", lambda: e("A0d", "1dAVb", "auroc"))
add(S8, "A0e-1 floresta AUROC 1dAVb", "0,942", lambda: ic_rf("A0e-1 RF", "1dAVb", "auroc"))
add(S8, "A0e-1 floresta AUROC 1dAVb IC inf", "0,922", lambda: ic_rf("A0e-1 RF", "1dAVb", "auroc", "ic_inf"))
add(S8, "A0e-1 floresta AUROC 1dAVb IC sup", "0,957", lambda: ic_rf("A0e-1 RF", "1dAVb", "auroc", "ic_sup"))
add(S8, "A0d AUPRC 1dAVb", "0,053", lambda: e("A0d", "1dAVb", "auprc"))
add(S8, "A0e-1 floresta AUPRC 1dAVb", "0,262", lambda: ic_rf("A0e-1 RF", "1dAVb", "auprc"))
add(S8, "A0e-1 floresta AUPRC 1dAVb IC inf", "0,204", lambda: ic_rf("A0e-1 RF", "1dAVb", "auprc", "ic_inf"))
add(S8, "A0e-1 floresta AUPRC 1dAVb IC sup", "0,337", lambda: ic_rf("A0e-1 RF", "1dAVb", "auprc", "ic_sup"))
add(S8, "A3 AUROC 1dAVb", "0,972", lambda: e("A3", "1dAVb", "auroc"))
add(S8, "A3 AUPRC 1dAVb", "0,436", lambda: e("A3", "1dAVb", "auprc"))
for m, au, ap in (("A1", "0,976", "0,656"), ("A2", "0,985", "0,682"), ("A3_2069", "0,990", "0,721")):
    add(S9, f"{m} AUROC macro (2.069)", au, lambda m=m: e(m, "MACRO", "auroc"))
    add(S9, f"{m} AUPRC macro (2.069)", ap, lambda m=m: e(m, "MACRO", "auprc"))
for met, md, dp in (("auroc", "0,9897", "0,0004"), ("auprc", "0,697", "0,007"), ("f1", "0,713", "0,004")):
    add(S9, f"3 seeds do A3: média {met}", md, lambda met=met: (F_EXP + " (seeds 42, 43, 44; desvio ddof=1)", seeds(met)[0]))
    add(S9, f"3 seeds do A3: desvio {met}", dp, lambda met=met: (F_EXP + " (seeds 42, 43, 44; desvio ddof=1)", seeds(met)[1]))
add(S10, "1dAVb: precisão máxima com sensibilidade >= 0,5", "0,45", prec_max_1davb)
add(S12, "A6_completo AUPRC macro", "0,445", lambda: e("A6", "MACRO", "auprc"))
add(S12, "A5 parcial (14/11) AUPRC macro", "0,705", lambda: a5("auprc"))
add(S12, "A3 sens. com esp. 99%, 1dAVb", "0,48", lambda: op("1dAVb", "sens_esp99"))
add(S12, "A3 sens. com esp. 99%, AF", "0,75", lambda: op("AF", "sens_esp99"))
add(S12, "A3 sens. com esp. 95%, 1dAVb", "0,88", lambda: op("1dAVb", "sens_esp95"))
add(S12, "A3 sens. com esp. 95%, ST", "1,00", lambda: op("ST", "sens_esp95"))
add(S12, "floresta A0e-1 5 folds AUPRC média", "0,558", lambda: kfold("auprc_media"))
add(S12, "floresta A0e-1 5 folds AUPRC dp", "0,020", lambda: kfold("auprc_dp"))
add(S12, "ensemble A3+A6 AUPRC macro", "0,676", lambda: e("ens", "MACRO", "auprc"))

# --- CLAUDE.md (Parte A)
C = "CLAUDE.md, seção 5.3 / 0 / 9"
for m, vals in (("A0b", ("0,705", "0,046", "0,095")), ("A0b_A1", ("0,697", "0,046", "0,103")), ("A1", ("0,976", "0,656", "0,696")),
                ("A2", ("0,985", "0,682", "0,717")), ("A3", ("0,989", "0,691", "0,709")), ("A0b_A3", ("0,705", "0,046", "0,095")),
                ("A3_2069", ("0,990", "0,721", "0,717")), ("A3_2069r", (None, None, "0,739"))):
    for met, v in zip(("auroc", "auprc", "f1"), vals):
        if v:
            add(C, f"{m} {met} macro", v, lambda m=m, met=met: e(m, "MACRO", met))
for m, vals in (("A1", ("0,749", "0,658", "0,993")), ("A3", ("0,746", "0,681", "0,993")), ("A3_2069r", ("0,817", "0,680", None))):
    for met, v in zip(("sensibilidade", "precisao", "especificidade"), vals):
        if v:
            add(C, f"{m} {met} macro", v, lambda m=m, met=met: e(m, "MACRO", met))
for m, vals in (("A1", "0,919 0,997 0,994 0,992 0,996 0,959"), ("A2", "0,952 0,993 0,994 0,994 0,996 0,983"),
                ("A3_2069", "0,969 0,995 0,996 0,996 0,996 0,988")):
    for r, v in zip(R, vals.split()):
        add(C, f"{m} AUROC {r}", v, lambda m=m, r=r: e(m, r, "auroc"))
a3 = {"1dAVb": "0,972 0,436 0,485", "RBBB": "0,995 0,851 0,826", "LBBB": "0,994 0,844 0,798", "SB": "0,993 0,618 0,658",
      "ST": "0,995 0,695 0,783", "AF": "0,986 0,700 0,701"}
for r, vs in a3.items():
    for met, v in zip(("auroc", "auprc", "f1"), vs.split()):
        add(C, f"A3 {met} {r}", v, lambda r=r, met=met: e("A3", r, met))
add(C, "A3 n_treino", "55.561", lambda: e("A3", "MACRO", "n_treino"))
add(C, "A1 n_treino (parte 0)", "13.866", lambda: e("A1", "MACRO", "n_treino"))
add(C, "A1 n_val (parte 0)", "2.069", lambda: e("A1", "MACRO", "n_val"))
# testes (já abertos em 08/11; só leitura das saídas)
T = "CLAUDE.md, seção 0 (Testes do A3 congelado)"
for conj, vs in (("interno", "0,989 0,987 0,991 0,710 0,686 0,733 0,705 0,687 0,722"),
                 ("ptbxl", "0,970 0,968 0,973 0,788 0,776 0,799 0,746 0,735 0,756"),
                 ("code_test", "0,998 0,996 0,999 0,916 0,870 0,960 0,812 0,747 0,865")):
    v = vs.split()
    for i, met in enumerate(("auroc", "auprc", "f1")):
        add(T, f"{conj} {met} macro", v[3 * i], lambda c=conj, m=met: t(c, "MACRO", m))
        add(T, f"{conj} {met} macro IC inf", v[3 * i + 1], lambda c=conj, m=met: t(c, "MACRO", m + "_ic_inf"))
        add(T, f"{conj} {met} macro IC sup", v[3 * i + 2], lambda c=conj, m=met: t(c, "MACRO", m + "_ic_sup"))
add(T, "PTB-XL AUROC SB", "0,920", lambda: t("ptbxl", "SB", "auroc"))
add(T, "PTB-XL AUROC 1dAVb", "0,924", lambda: t("ptbxl", "1dAVb", "auroc"))
for r in ("RBBB", "LBBB", "ST", "AF"):
    add(T, f"PTB-XL AUROC {r} (citado: >= 0,990)", ">=0,990", lambda r=r: t("ptbxl", r, "auroc"))
for c in ("interno", "ptbxl", "code_test"):
    add(T, f"{c} sensibilidade 1dAVb (citado: ~0,50)", "~0,50", lambda c=c: t(c, "1dAVb", "sensibilidade"))
for r, p in zip(R, "281 442 290 308 353 310".split()):
    add("CLAUDE.md, seção 9 (teste interno)", f"positivos {r} teste interno", p, lambda r=r: t("interno", r, "positivos"))
for r, p in zip(R, "28 34 30 16 37 13".split()):
    add("CLAUDE.md, seção 4 (CODE-TEST)", f"positivos {r} CODE-TEST", p, lambda r=r: t("code_test", r, "positivos"))
add(T, "PTB-XL positivos RBBB (com exclusão IRBBB)", "540", lambda: t("ptbxl", "RBBB", "positivos"))
add(T, "PTB-XL positivos AF (com exclusão AFLT)", "1.497", lambda: t("ptbxl", "AF", "positivos"))


def status(citado, v):
    if citado.startswith(">="):
        return "ok" if v >= float(citado[2:].replace(",", ".")) - 5e-4 else "divergente"
    if citado.startswith("~"):
        return "ok" if abs(v - float(citado[1:].replace(",", "."))) <= 0.05 else "divergente"
    if "," in citado:
        casas = len(citado.split(",")[1])
        return "ok" if abs(v - float(citado.replace(",", "."))) <= 0.5 * 10 ** -casas + 1e-9 else "divergente"
    return "ok" if abs(v - float(citado.replace(".", ""))) < 0.5 else "divergente"


linhas = []
for onde, item, citado, f in L:
    arq, v = f()
    linhas.append(dict(onde=onde, item=item, valor_citado=citado, valor_arquivo=round(float(v), 6), arquivo_origem=arq, status=status(citado, float(v))))
tab = pd.DataFrame(linhas)
salvar_novo(tab, RES / "sessao2_conferencia_numeros_parte_a.csv")
print(tab.status.value_counts().to_string())
pd.set_option("display.width", 250)
print(tab[tab.status != "ok"].to_string(index=False))
