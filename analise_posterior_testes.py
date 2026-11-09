"""POSTERIOR AO TESTE, DESCRITIVO, EXPLORATÓRIO (08/11, madrugada). Nada aqui muda modelo, limiares, normalização, exclusões,
limite de likelihood ou o JSON congelado. Nenhum predict: as notas vêm de resultados/bruto_*.npz (testes) e
resultados/erros_A3/previsoes_e_rotulos.npy (validação). Sinal do PTB-XL e da validação só é lido para estimar a frequência cardíaca.

Uso (.venv, na raiz):
  python analise_posterior_testes.py checar      -> confere o estimador de FC em sinais sintéticos de FC conhecida
  python analise_posterior_testes.py tudo        -> A (SB), B (1dAVb), C (LBBB no PTB-XL), D (CODE-TEST), figuras l, m
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt

RAIZ = Path(__file__).parent
R, F = RAIZ / "resultados", RAIZ / "figuras"
SAIDA = R / "posterior_testes"
ROTULOS = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF"]
import json
LIMIARES = json.loads((RAIZ / "congelado_candidato_A3.json").read_text(encoding="utf-8"))["limiares"]["valores"]
FAIXAS = [(0, 40, "<40"), (40, 60, "40-59"), (60, 70, "60-69"), (70, 1000, "70+")]  # PTB-XL: idade 300 (= mais de 89) cai em 70+
MARCA = "posterior ao teste, descritivo"


def br(x, c=3):
    return f"{x:.{c}f}".replace(".", ",")


# ------------------------------------------------------------------------------------------------ frequência cardíaca
def fc_estimada(x, fs):
    """FC (bpm) de UMA derivação. Método (tipo Pan-Tompkins simplificado): passa-banda 5-15 Hz (realça o QRS, atenua onda T e
    linha de base), derivada ao quadrado, média móvel de 150 ms; picos com distância mínima de 0,25 s (no máximo 240 bpm) e altura
    >= 30% do percentil 99 do sinal integrado. FC = 60 / mediana dos intervalos RR. Menos de 3 picos -> NaN.
    Devolve (fc, n_picos)."""
    if len(x) < 2 * fs or not np.any(x):
        return np.nan, 0
    y = sosfiltfilt(butter(2, [5, 15], btype="band", fs=fs, output="sos"), x)
    w = int(0.15 * fs)
    e = np.convolve(np.diff(y) ** 2, np.ones(w) / w, mode="same")
    pk, _ = find_peaks(e, distance=int(0.25 * fs), height=0.3 * np.percentile(e, 99))
    if len(pk) < 3:
        return np.nan, len(pk)
    return 60 / np.median(np.diff(pk) / fs), len(pk)


def ecg_sintetico(bpm, fs, seg=10, seed=0):
    """QRS estreito (20 ms) + onda T larga (amplitude 0,35, 250 ms depois) + ruído e linha de base: só para conferir o estimador."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seg * fs)) / fs
    x = 0.2 * np.sin(2 * np.pi * 0.3 * t) + 0.03 * rng.standard_normal(len(t))
    for b in np.arange(0.3, seg, 60 / bpm):
        x += np.exp(-((t - b) / 0.01) ** 2) + 0.35 * np.exp(-((t - b - 0.25) / 0.06) ** 2)
    return x


def checar():
    for fs in (400, 500):
        for bpm in (40, 50, 58, 75, 110, 150):
            est, _ = fc_estimada(ecg_sintetico(bpm, fs, seg=7.3 if fs == 400 else 10), fs)
            assert abs(est - bpm) < 1.5, (fs, bpm, est)
    print("checar: estimador de FC ok (erro < 1,5 bpm em 40-150 bpm, 400 e 500 Hz, sinais sintéticos)")


def fc_registro_ptbxl(caminho):
    import wfdb
    x = wfdb.rdrecord(str(RAIZ / "ptbxl" / caminho)).p_signal
    return [*fc_estimada(x[:, 1], 500), fc_estimada(x[:, 10], 500)[0]]  # DII, nº de picos em DII, V5


def fc_exame_code(ds, linha):
    x = ds[linha]
    ativo = np.where(np.abs(x).max(axis=1) > 0)[0]
    x = x[ativo.min():ativo.max() + 1] if len(ativo) else x
    return [*fc_estimada(x[:, 1].astype(float), 400), fc_estimada(x[:, 10].astype(float), 400)[0]]


# ------------------------------------------------------------------------------------------------ dados (só notas salvas)
def bruto(c):
    b = np.load(sorted(R.glob(f"bruto_{c}_*.npz"))[-1], allow_pickle=False)
    return {k: b[k] for k in b.files}


def validacao():
    """Notas, rótulos, idade, sexo da validação do A3 (salvos pela análise de erros) + parte/linha para ler o sinal."""
    from a1_cnn import carregar_indices
    from ecg_dados import GeradorECG
    a = np.load(R / "erros_A3" / "previsoes_e_rotulos.npy")
    _, va, _ = carregar_indices(partes=[0, 1, 2, 3])
    info = GeradorECG(va).info  # mesma ordem das previsões salvas (parte, linha)
    assert (info[ROTULOS].to_numpy() == a[:, 6:12]).all() and (info.patient_id.to_numpy() == a[:, 14]).all()
    return dict(S=a[:, :6], Y=a[:, 6:12], idade=a[:, 12], homem=a[:, 13].astype(bool), info=info)


def ptbxl_info():
    from ecg_dados import ptbxl_indice
    info = ptbxl_indice()
    b = bruto("ptbxl")
    assert (info.index.to_numpy() == b["ids"]).all()
    return info, b


# ------------------------------------------------------------------------------------------------ A) SB
def tarefa_a():
    k = ROTULOS.index("SB")
    info, b = ptbxl_info()
    v = validacao()
    lim = LIMIARES["SB"]
    print(f"\n== A) SB ({MARCA})\n(i) notas do modelo para SB (limiar congelado {br(lim, 3)})")
    q = [0.05, 0.25, 0.5, 0.75, 0.95]
    for nome, S, Y in (("PTB-XL", b["S"][:, k], b["Y"][:, k]), ("Validação CODE-15%", v["S"][:, k], v["Y"][:, k])):
        for grupo, s in (("positivos", S[Y == 1]), ("negativos", S[Y == 0])):
            print(f"  {nome} {grupo} (n={len(s)}): quantis 5/25/50/75/95% = " + " / ".join(br(x, 3) for x in np.quantile(s, q))
                  + f"; nota < 0,05: {(s < 0.05).sum()} ({100 * (s < 0.05).mean():.1f}%); >= limiar: {(s >= lim).sum()} ({100 * (s >= lim).mean():.1f}%)")

    # (ii) e (iii): FC estimada. Amostras (seed 42). Positivos SB: todos (PTB-XL só tem 637; validação 118).
    cache = SAIDA / "a_fc_estimada.csv"
    if cache.exists():  # FC já estimada numa rodada anterior: não lê os sinais de novo
        df = pd.read_csv(cache)
        resumo_fc(df)
        return df
    rng = np.random.default_rng(42)
    sb_pos = np.where(b["Y"][:, k] == 1)[0]
    sb_neg = rng.choice(np.where(b["Y"][:, k] == 0)[0], 1000, replace=False)
    st_pos = rng.choice(np.where(b["Y"][:, ROTULOS.index("ST")] == 1)[0], 200, replace=False)  # controle: taquicardia deve dar > 100
    linhas = []
    for grupo, idx in (("SB positivo", sb_pos), ("SB negativo (amostra)", sb_neg), ("ST positivo (controle)", st_pos)):
        for i in idx:
            fc, n, fc5 = fc_registro_ptbxl(info["filename_hr"].iloc[i])
            linhas.append(dict(base="PTB-XL", grupo=grupo, id=int(b["ids"][i]), nota_SB=float(b["S"][i, k]), fc_II=fc, picos_II=n, fc_V5=fc5,
                               idade=info["age"].iloc[i], homem=bool(info["sex"].iloc[i] == 0)))
        print(f"  FC PTB-XL: {grupo} pronto ({len(idx)})", flush=True)
    import h5py
    vi = v["info"]
    vpos = np.where(v["Y"][:, k] == 1)[0]
    vneg = rng.choice(np.where(v["Y"][:, k] == 0)[0], 1000, replace=False)
    vst = np.where(v["Y"][:, ROTULOS.index("ST")] == 1)[0]
    arqs = {p: h5py.File(RAIZ / "code15" / f"exams_part{p}.hdf5", "r") for p in (0, 1, 2, 3)}
    for grupo, idx in (("SB positivo", vpos), ("SB negativo (amostra)", vneg), ("ST positivo (controle)", vst)):
        for i in idx:
            fc, n, fc5 = fc_exame_code(arqs[vi.parte.iloc[i]]["tracings"], vi.linha.iloc[i])
            linhas.append(dict(base="Validação CODE-15%", grupo=grupo, id=int(vi.exam_id.iloc[i]), nota_SB=float(v["S"][i, k]), fc_II=fc,
                               picos_II=n, fc_V5=fc5, idade=v["idade"][i], homem=bool(v["homem"][i])))
        print(f"  FC validação: {grupo} pronto ({len(idx)})", flush=True)
    df = pd.DataFrame(linhas)
    df.to_csv(SAIDA / "a_fc_estimada.csv", index=False, float_format="%.3f")
    resumo_fc(df)
    return df


def resumo_fc(df):
    print("\n(ii)/(iii) FC estimada pela derivação II (bpm)")
    df["concorda_V5"] = (df.fc_II - df.fc_V5).abs() <= 5
    for (base, grupo), d in df.groupby(["base", "grupo"], sort=False):
        f = d.fc_II.dropna()
        print(f"  {base} | {grupo}: n={len(d)}, sem estimativa {d.fc_II.isna().sum()}; mediana {br(f.median(), 1)} (p10 {br(f.quantile(.1), 1)}, "
              f"p90 {br(f.quantile(.9), 1)}); < 50: {(f < 50).sum()} ({100 * (f < 50).mean():.1f}%); 50-59,9: {((f >= 50) & (f < 60)).sum()} "
              f"({100 * ((f >= 50) & (f < 60)).mean():.1f}%); < 60: {(f < 60).sum()} ({100 * (f < 60).mean():.1f}%); > 100: {(f > 100).sum()}; "
              f"DII e V5 a até 5 bpm: {100 * d.concorda_V5[d.fc_II.notna() & d.fc_V5.notna()].mean():.1f}%")
    p = df[(df.base == "PTB-XL") & (df.grupo == "SB positivo")].dropna(subset=["fc_II"])
    print("\n  PTB-XL, SB positivos: nota do modelo por faixa de FC estimada")
    for lo, hi, nome in ((0, 50, "< 50"), (50, 55, "50-54,9"), (55, 60, "55-59,9"), (60, 1000, ">= 60")):
        d = p[(p.fc_II >= lo) & (p.fc_II < hi)]
        if len(d):
            print(f"    FC {nome}: n={len(d)}; nota < 0,05: {(d.nota_SB < 0.05).sum()} ({100 * (d.nota_SB < 0.05).mean():.1f}%); "
                  f">= limiar: {(d.nota_SB >= LIMIARES['SB']).sum()} ({100 * (d.nota_SB >= LIMIARES['SB']).mean():.1f}%)")
    q = df[(df.base == "Validação CODE-15%") & (df.grupo == "SB negativo (amostra)")].dropna(subset=["fc_II"])
    print(f"  Validação CODE-15%, SB NEGATIVOS com FC estimada 50-59,9: {((q.fc_II >= 50) & (q.fc_II < 60)).sum()} de {len(q)}; < 50: {(q.fc_II < 50).sum()}")


# ------------------------------------------------------------------------------------------------ B) 1dAVb
def conjuntos_com_idade_sexo():
    from ecg_dados import CODE_TEST, tabela_com_grupo
    v = validacao()
    out = {"Validação": (v["S"], v["Y"], np.ones_like(v["Y"], bool), v["idade"], v["homem"])}
    b = bruto("interno")
    t = tabela_com_grupo().set_index("exam_id").loc[b["ids"]]
    out["Teste interno"] = (b["S"], b["Y"], b["mask"], t.age.to_numpy(), t.is_male.to_numpy().astype(bool))
    info, b = ptbxl_info()
    out["PTB-XL"] = (b["S"], b["Y"], b["mask"], info.age.to_numpy(), info.sex.to_numpy() == 0)  # PTB-XL: 0 = homem
    b = bruto("code_test")
    at = pd.read_csv(CODE_TEST / "attributes.csv").iloc[b["ids"]]
    out["CODE-TEST"] = (b["S"], b["Y"], b["mask"], at.age.to_numpy(), at.sex.to_numpy() == "M")
    return out


def tarefa_b():
    k, lim = ROTULOS.index("1dAVb"), LIMIARES["1dAVb"]
    print(f"\n== B) 1dAVb ({MARCA}); limiar congelado {br(lim)}")
    linhas = []
    for nome, (S, Y, M, idade, homem) in conjuntos_com_idade_sexo().items():
        m = M[:, k]
        s, y, idade, homem = S[m, k], Y[m, k], idade[m], homem[m]
        for g, sg in (("positivos", s[y == 1]), ("negativos", s[y == 0])):
            print(f"  {nome} {g} (n={len(sg)}): quantis 5/25/50/75/95% = " + " / ".join(br(x) for x in np.quantile(sg, [.05, .25, .5, .75, .95])))
        p = s >= lim
        sub = [(f"idade {n}", (idade >= lo) & (idade < hi)) for lo, hi, n in FAIXAS] + [("homens", homem), ("mulheres", ~homem)]
        for rot, sel in sub:
            pos, neg = sel & (y == 1), sel & (y == 0)
            linhas.append(dict(conjunto=nome, subgrupo=rot, positivos=int(pos.sum()), falsos_negativos=int((pos & ~p).sum()),
                               sensibilidade=p[pos].mean() if pos.any() else np.nan, negativos=int(neg.sum()),
                               falsos_positivos=int((neg & p).sum()), taxa_falso_positivo=p[neg].mean() if neg.any() else np.nan))
    df = pd.DataFrame(linhas)
    df.to_csv(SAIDA / "b_1davb_subgrupos.csv", index=False, float_format="%.4f")
    print("\n  erros por subgrupo (sensibilidade = positivos achados / positivos; taxa de falso positivo = negativos marcados / negativos)")
    for x in df.itertuples():
        print(f"  {x.conjunto} | {x.subgrupo}: positivos {x.positivos}, FN {x.falsos_negativos}, sensibilidade "
              f"{br(x.sensibilidade) if x.positivos else 'n/d'} | negativos {x.negativos}, FP {x.falsos_positivos}, taxa FP {br(x.taxa_falso_positivo, 4)}")


# ------------------------------------------------------------------------------------------------ C) LBBB no PTB-XL
def tarefa_c():
    k, lim = ROTULOS.index("LBBB"), LIMIARES["LBBB"]
    info, b = ptbxl_info()
    scp = info["scp_codes"].apply(ast.literal_eval).to_numpy()
    st = pd.read_csv(RAIZ / "ptbxl" / "scp_statements.csv", index_col=0)
    cd = set(st.index[st.diagnostic_class == "CD"])  # "conduction disturbance" segundo o scp_statements.csv do PTB-XL
    m, y, p = b["mask"][:, k], b["Y"][:, k], b["S"][:, k] >= lim
    fp, vn = np.where(m & (y == 0) & p)[0], np.where(m & (y == 0) & ~p)[0]
    print(f"\n== C) LBBB no PTB-XL ({MARCA}): {len(fp)} falsos positivos (de {int((m & (y == 0)).sum())} negativos avaliados)")
    cont = lambda idx: pd.Series([c for i in idx for c in scp[i]]).value_counts()
    c_fp = cont(fp)
    com_cd = np.array([bool(set(scp[i]) & cd) for i in fp])
    print(f"  com algum código de distúrbio de condução (classe CD do scp_statements): {com_cd.sum()} ({100 * com_cd.mean():.1f}%); "
          f"entre os verdadeiros negativos: {100 * np.mean([bool(set(scp[i]) & cd) for i in vn]):.1f}%")
    for c in ("IRBBB", "ILBBB", "IVCD", "LAFB", "LPFB", "CRBBB", "1AVB", "WPW"):
        print(f"    {c}: {int(c_fp.get(c, 0))} dos {len(fp)} FP")
    print("  códigos mais frequentes entre os FP (um registro pode ter vários): " + ", ".join(f"{c} {n}" for c, n in c_fp.head(12).items()))
    sem = fp[~com_cd]
    print(f"  FP SEM código CD ({len(sem)}): códigos mais frequentes: " + ", ".join(f"{c} {n}" for c, n in cont(sem).head(10).items()))
    il = np.array([("ILBBB" in d) for d in scp])
    print(f"  (fora da avaliação de LBBB) registros com ILBBB: {il.sum()}; marcados como LBBB pelo modelo: {int(p[il].sum())}")
    pd.DataFrame({"codigo": c_fp.index, "n_fp": c_fp.values, "classe_CD": [c in cd for c in c_fp.index]}).to_csv(
        SAIDA / "c_lbbb_fp_codigos.csv", index=False)


# ------------------------------------------------------------------------------------------------ D) CODE-TEST
def tarefa_d():
    from ecg_dados import CODE_TEST
    b = bruto("code_test")
    gold = pd.read_csv(CODE_TEST / "annotations" / "gold_standard.csv")[ROTULOS].to_numpy()[b["ids"]]
    assert (gold == b["Y"]).all()
    a3 = (b["S"] >= np.array([LIMIARES[r] for r in ROTULOS])).astype(int)
    justos = {g: pd.read_csv(CODE_TEST / "annotations" / f"{g}.csv")[ROTULOS].to_numpy()[b["ids"]]
              for g in ("cardiology_residents", "emergency_residents", "medical_students")}

    def f1(P, Y):
        tp, fp, fn = ((P == 1) & (Y == 1)).sum(0), ((P == 1) & (Y == 0)).sum(0), ((P == 0) & (Y == 1)).sum(0)
        f = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        return np.append(f, f.mean())

    rng = np.random.default_rng(42)
    boot = {n: [] for n in ["A3", *justos]}
    for _ in range(1000):  # bootstrap por exame (sem patient_id), mesmas reamostras para todos
        ii = rng.integers(0, len(gold), len(gold))
        boot["A3"].append(f1(a3[ii], gold[ii]))
        for g, P in justos.items():
            boot[g].append(f1(P[ii], gold[ii]))
    boot = {n: np.array(v) for n, v in boot.items()}
    pontual = {"A3": f1(a3, gold), **{g: f1(P, gold) for g, P in justos.items()}}
    ic = lambda a: (np.percentile(a, 2.5, axis=0), np.percentile(a, 97.5, axis=0))
    av = pd.read_csv(R / "avaliacao_code_test.csv").set_index("rotulo")
    print(f"\n== D) CODE-TEST ({MARCA}). Gold standard = concordância de cardiologist1 e cardiologist2; nas divergências, um terceiro "
          "especialista, que via as duas anotações, decidiu (README do zip). Comparar os cardiologistas com o gold standard é circular.")
    print("  F1 reportado antes: 0,902 = cardiologist1; 0,935 = cardiologist2 (circulares, fora da comparação).")
    linhas = []
    for j, r in enumerate(ROTULOS + ["MACRO"]):
        lin = dict(rotulo=r, n_positivos=int(gold[:, j].sum()) if j < 6 else None,
                   A3_auroc=av.loc[r, "auroc"], A3_auroc_ic=f"{av.loc[r, 'auroc_ic_inf']:.3f}-{av.loc[r, 'auroc_ic_sup']:.3f}")
        for n in boot:
            lo, hi = ic(boot[n])
            lin[f"{n}_f1"], lin[f"{n}_f1_ic"] = pontual[n][j], f"{lo[j]:.3f}-{hi[j]:.3f}"
            if n != "A3":
                d = boot["A3"][:, j] - boot[n][:, j]
                lin[f"A3_menos_{n}"], lin[f"A3_menos_{n}_ic"] = pontual["A3"][j] - pontual[n][j], f"{np.percentile(d, 2.5):.3f}-{np.percentile(d, 97.5):.3f}"
        linhas.append(lin)
    df = pd.DataFrame(linhas)
    df.to_csv(SAIDA / "d_codetest_grupos_justos.csv", index=False, float_format="%.4f")
    for x in linhas:
        print(f"  {x['rotulo']} (n={x['n_positivos']}): A3 AUROC {br(x['A3_auroc'])} [{x['A3_auroc_ic']}] | F1 A3 {br(x['A3_f1'])} [{x['A3_f1_ic']}]; "
              + "; ".join(f"{g} {br(x[g + '_f1'])} [{x[g + '_f1_ic']}] (A3 - {g}: {br(x['A3_menos_' + g])} [{x['A3_menos_' + g + '_ic']}])" for g in justos))


# ------------------------------------------------------------------------------------------------ E) figuras
def figuras(df_fc):
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(RAIZ))
    from figuras import salvar
    k = ROTULOS.index("SB")
    _, b = ptbxl_info()
    v = validacao()
    cores = {"PTB-XL": "#009e73", "Validação CODE-15%": "#d55e00"}
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    for ax, y_alvo, tit in ((axs[0], 1, "SB positivos"), (axs[1], 0, "SB negativos")):
        for nome, S, Y in (("PTB-XL", b["S"][:, k], b["Y"][:, k]), ("Validação CODE-15%", v["S"][:, k], v["Y"][:, k])):
            s = np.sort(S[Y == y_alvo])
            ax.plot(s, np.arange(1, len(s) + 1) / len(s), color=cores[nome], lw=2.5, label=f"{nome} (n={len(s)})")
        ax.axvline(LIMIARES["SB"], color="#555555", ls="--", lw=1.5, label="limiar congelado")
        ax.set(xscale="log", xlim=(1e-6, 1), xlabel="nota do modelo para SB (escala log)", ylabel="fração acumulada dos exames", title=tit)
        ax.legend(loc="upper left", fontsize=12)
    fig.suptitle(f"Notas de SB: PTB-XL vs validação ({MARCA})", fontsize=16)
    salvar(fig, "l_sb_notas_ptbxl_validacao.png")

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    pos = df_fc[df_fc.grupo == "SB positivo"]
    bins = np.arange(30, 101, 2.5)
    for nome, d in pos.groupby("base"):
        axs[0].hist(d.fc_II.dropna().clip(30, 100), bins=bins, density=True, alpha=0.55, color=cores[nome], label=f"{nome} (n={d.fc_II.notna().sum()})")
    for x in (50, 60):
        axs[0].axvline(x, color="#555555", ls="--", lw=1.5)
    axs[0].set(xlabel="FC estimada, derivação II (bpm; cortada em 30 e 100)", ylabel="densidade", title="SB positivos: FC estimada")
    axs[0].legend(fontsize=12)
    p = pos[pos.base == "PTB-XL"]
    axs[1].scatter(p.fc_II, p.nota_SB.clip(1e-6), s=18, alpha=0.6, color=cores["PTB-XL"])
    axs[1].axhline(LIMIARES["SB"], color="#555555", ls="--", lw=1.5, label="limiar congelado")
    axs[1].axvline(50, color="#999999", ls=":", lw=1.5)
    axs[1].axvline(60, color="#999999", ls=":", lw=1.5)
    axs[1].set(yscale="log", xlim=(30, 100), xlabel="FC estimada, derivação II (bpm)", ylabel="nota do modelo para SB (log)",
               title="PTB-XL, SB positivos: nota vs FC")
    axs[1].legend(fontsize=12, loc="lower left")
    fig.suptitle(f"Frequência cardíaca estimada pelo sinal ({MARCA}; estimativa, não medida clínica)", fontsize=16)
    salvar(fig, "m_fc_sb_ptbxl_validacao.png")


if __name__ == "__main__":
    SAIDA.mkdir(exist_ok=True)
    if sys.argv[1] == "checar":
        checar()
    elif sys.argv[1] == "tudo":
        checar()
        tarefa_d()
        tarefa_c()
        tarefa_b()
        figuras(tarefa_a())
