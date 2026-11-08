"""Parte B, só tabela (Bloco 1 de 08/11): B0 (idade + sexo) e B1 (B0 + idade do ECG). NENHUM sinal é lido.

PROVISÓRIO, a confirmar com a Renata:
  - horizonte principal 2 anos (declarado ANTES de rodar); 1 e 3 anos só como sensibilidade;
  - só treino -> validação; o grupo teste é cortado logo na carga e nunca é usado.
Regra do horizonte (seção 3.6 do CLAUDE.md): entra quem morreu dentro do prazo OU foi acompanhado o prazo inteiro.
Quem ficou vivo e foi acompanhado menos que o prazo SAI (censurado): não sabemos se morreu depois, então nunca vira "sobreviveu".
`nn_predicted_age` ser a saída do modelo de Lima et al. 2021 é provável, não documentado.

Uso: .venv\\Scripts\\python parte_b_tabular.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from a0_baselines import PASTA_RESULTADOS
from ecg_dados import SEED, tabela_com_grupo

HORIZONTES = [2, 1, 3]  # o primeiro é o principal
FAIXAS = [(0, 40, "<40"), (40, 60, "40-59"), (60, 70, "60-69"), (70, 200, "70+")]
N_BOOT = 1000


def carregar():
    """Primeiro exame de cada paciente (o único com death/timey), só treino e validação."""
    t = tabela_com_grupo()
    t = t[t.grupo != "teste"]  # o teste nunca passa daqui
    p = t[t.death.notna()].copy()
    assert p.patient_id.is_unique, "mais de um exame com death por paciente"
    p["death"] = p.death.astype(bool)
    p["delta"] = p.nn_predicted_age - p.age  # idade do ECG - idade real
    return t, p


def rotular(p, h):
    """Aplica a regra do horizonte h (anos). Devolve (elegíveis com coluna y, nº censurados)."""
    morreu_no_prazo = p.death & (p.timey <= h)
    acompanhado = p.timey >= h
    censurado = ~p.death & (p.timey < h)  # vivo e acompanhado menos que h: sai
    assert ((morreu_no_prazo | acompanhado) == ~censurado).all()
    e = p[~censurado].copy()
    e["y"] = morreu_no_prazo[~censurado].astype(int)
    return e, censurado


def features(d, esc, modelo):
    """B0: idade (padronizada no treino) + sexo. B1: B0 + idade do ECG + diferença (padronizadas no treino)."""
    cols = ["age"] if modelo == "B0" else ["age", "nn_predicted_age", "delta"]
    return np.column_stack([esc[modelo].transform(d[cols]), d.is_male.astype(float)])


def ajustar(tr):
    esc = {"B0": StandardScaler().fit(tr[["age"]]), "B1": StandardScaler().fit(tr[["age", "nn_predicted_age", "delta"]])}
    mods = {m: LogisticRegression(max_iter=1000, random_state=SEED).fit(features(tr, esc, m), tr.y) for m in ("B0", "B1")}
    return esc, mods


def bootstrap(y, s0, s1, n=N_BOOT):
    """IC 95% por bootstrap POR PACIENTE (aqui 1 linha = 1 paciente). Inclui a diferença pareada B1 - B0."""
    rng = np.random.default_rng(SEED)
    out = {k: [] for k in ("auroc_B0", "auroc_B1", "auprc_B0", "auprc_B1", "d_auroc", "d_auprc")}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        a0, a1 = roc_auc_score(y[i], s0[i]), roc_auc_score(y[i], s1[i])
        p0, p1 = average_precision_score(y[i], s0[i]), average_precision_score(y[i], s1[i])
        for k, v in zip(out, (a0, a1, p0, p1, a1 - a0, p1 - p0)):
            out[k].append(v)
    return {k: np.percentile(v, [2.5, 97.5]) for k, v in out.items()}


def main():
    t, p = carregar()
    print(f"pacientes (treino+validação, tabela inteira): {t.patient_id.nunique():,}; com death/timey: {len(p):,}")
    # é o primeiro exame? (menor idade do paciente)
    menor = t.groupby("patient_id").age.min()
    print(f"exame com death tem a menor idade do paciente: {(p.age.values == menor.loc[p.patient_id].values).mean():.1%}")
    print("timey (descrição):", p.timey.describe().round(2).to_dict())
    print("timey dos óbitos:", p.loc[p.death, "timey"].describe().round(2).to_dict())

    resumo, sens, faixas, calib = [], [], [], []
    for h in HORIZONTES:
        e, cens = rotular(p, h)
        for g in ("treino", "validação"):
            resumo.append(dict(horizonte=h, grupo=g, pacientes_com_death=int((p.grupo == g).sum()),
                               elegiveis=int((e.grupo == g).sum()), obitos=int(e.loc[e.grupo == g, "y"].sum()),
                               censurados_sairam=int(cens[p.grupo == g].sum())))
        tr, va = e[e.grupo == "treino"], e[e.grupo == "validação"]
        assert not set(tr.patient_id) & set(va.patient_id)
        esc, mods = ajustar(tr)
        y = va.y.to_numpy()
        s = {m: mods[m].predict_proba(features(va, esc, m))[:, 1] for m in mods}
        ic = bootstrap(y, s["B0"], s["B1"])
        lin = dict(horizonte=h, principal=(h == 2), n_val=len(y), obitos_val=int(y.sum()))
        for m in ("B0", "B1"):
            lin[f"auroc_{m}"], lin[f"auprc_{m}"] = roc_auc_score(y, s[m]), average_precision_score(y, s[m])
            lin[f"auroc_{m}_ic"], lin[f"auprc_{m}_ic"] = ic[f"auroc_{m}"].round(4).tolist(), ic[f"auprc_{m}"].round(4).tolist()
        lin["d_auroc_ic"], lin["d_auprc_ic"] = ic["d_auroc"].round(4).tolist(), ic["d_auprc"].round(4).tolist()
        lin["coef_B1"] = dict(zip(["age", "nn_predicted_age", "delta", "is_male"], mods["B1"].coef_[0].round(3).tolist()))
        sens.append(lin)
        if h != 2:
            continue
        # por faixa de idade (só horizonte principal)
        for lo, hi, nome in FAIXAS:
            k = ((va.age >= lo) & (va.age < hi)).to_numpy()
            d = dict(faixa=nome, n=int(k.sum()), obitos=int(y[k].sum()))
            if 0 < y[k].sum() < k.sum():
                ic_f = bootstrap(y[k], s["B0"][k], s["B1"][k])
                for m in ("B0", "B1"):
                    d[f"auroc_{m}"], d[f"auprc_{m}"] = roc_auc_score(y[k], s[m][k]), average_precision_score(y[k], s[m][k])
                    d[f"auroc_{m}_ic"] = ic_f[f"auroc_{m}"].round(3).tolist()
                d["d_auroc_ic"] = ic_f["d_auroc"].round(3).tolist()
            faixas.append(d)
        # calibração: 10 faixas de risco previsto (decis), risco médio previsto vs taxa observada
        for m in ("B0", "B1"):
            dec = pd.qcut(s[m], 10, labels=False, duplicates="drop")
            for b in np.unique(dec):
                k = dec == b
                calib.append(dict(modelo=m, decil=int(b) + 1, n=int(k.sum()), previsto=s[m][k].mean(), observado=y[k].mean()))
        np.save(PASTA_RESULTADOS / "parte_b_previsoes_validacao_2anos.npy",
                np.column_stack([va.patient_id, y, s["B0"], s["B1"], va.age]).astype(np.float64))

    resumo, sens, faixas, calib = map(pd.DataFrame, (resumo, sens, faixas, calib))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n== regra do horizonte: elegíveis, óbitos, censurados (só treino e validação) ==\n", resumo.to_string(index=False))
    print("\n== B0 vs B1 na validação (principal = 2 anos; 1 e 3 = sensibilidade) ==\n", sens.round(4).to_string(index=False))
    print("\n== por faixa de idade (2 anos, validação) ==\n", faixas.round(4).to_string(index=False))
    print("\n== calibração por decil de risco (2 anos, validação) ==\n", calib.round(4).to_string(index=False))

    # --- paradoxo de Simpson (mesma definição do notebook, seção 4: morte em qualquer tempo, só o 1º exame) ---
    # diferença: aqui SÓ treino + validação (o notebook usou a tabela inteira, que inclui pacientes do teste)
    grupo = pd.cut(p.delta, [-np.inf, -8, 8, np.inf], labels=["ECG 8+ mais jovem", "semelhante (±8)", "ECG 8+ mais velho"])
    direto = p.groupby(grupo, observed=True).agg(pacientes=("death", "size"), obitos=("death", "sum"),
                                                taxa=("death", "mean"), idade_media=("age", "mean"))
    faixa = pd.cut(p.age, [16, 40, 50, 60, 70, 80, 101], right=False, labels=["16-39", "40-49", "50-59", "60-69", "70-79", "80+"])
    por_faixa = p.groupby([faixa, grupo], observed=True).death.agg(["mean", "size"]).unstack()
    print("\n== Simpson, sem controle (treino+validação) ==\n", direto.round(4).to_string())
    print("\n== Simpson, dentro de cada faixa de idade (taxa; n) ==\n", por_faixa.round(4).to_string())

    resumo.to_csv(PASTA_RESULTADOS / "parte_b_horizonte.csv", index=False)
    sens.to_csv(PASTA_RESULTADOS / "parte_b_B0_B1.csv", index=False, float_format="%.4f")
    faixas.to_csv(PASTA_RESULTADOS / "parte_b_faixas_2anos.csv", index=False, float_format="%.4f")
    calib.to_csv(PASTA_RESULTADOS / "parte_b_calibracao_2anos.csv", index=False, float_format="%.4f")
    direto.to_csv(PASTA_RESULTADOS / "parte_b_simpson_direto.csv", float_format="%.4f")
    por_faixa.to_csv(PASTA_RESULTADOS / "parte_b_simpson_faixas.csv", float_format="%.4f")


if __name__ == "__main__":
    main()
