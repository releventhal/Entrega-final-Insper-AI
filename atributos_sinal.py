"""Fase 1 de robustez (12/11): atributos calculados do sinal, para árvore de decisão e random forest (A0c/A0d).
NÃO treina nada e NÃO altera o A3. Só lê TREINO e VALIDAÇÃO (o grupo teste é cortado em `carregar_indices`; o teste não é aberto).

Atributos (só na região sem o preenchimento de zeros, como no A2; sinal CRU, sem normalização, unidade da CODE-15% "não verificada"):
  - Ritmo, na DII (índice 1; ordem DI, DII, DIII, AVR, AVL, AVF, V1-V6 medida por Einthoven em 07/11): picos R pelo mesmo método de
    `analise_posterior_testes.fc_estimada` (passa-banda 5-15 Hz, derivada², média móvel 150 ms, scipy.signal.find_peaks; erro < 1,5 bpm
    em sinais sintéticos). A DII foi escolhida porque o QRS costuma ser positivo e bem definido nela, e porque DII e V5 já mostraram
    concordar a até 5 bpm em >= 96,9% dos exames (08/11).
    fc_bpm, rr_media_ms, rr_mediana_ms, rr_desvio_ms, rmssd_ms, rr_cv, n_picos (n_picos conta sempre; o resto precisa de >= 3 picos).
  - Amplitude pico a pico (máx - mín) nas 12 derivações: ptp_<deriv>. Sensível a picos isolados (não tratado).
  - Largura do QRS, APROXIMAÇÃO GROSSEIRA (não é delineação): em torno de cada pico da DII (±100 ms), na derivação filtrada 3-40 Hz, o
    trecho em que |derivada| > 15% do máximo da janela; mediana entre os batimentos. Calculada em DII, V1 e V6: qrs_ms_<deriv>.
  - Intervalo PR: NÃO calculado (precisa achar o início da onda P; sem biblioteca de delineação). Isto limita o 1dAVb (que é um PR longo).
Atributo impossível de calcular (derivação zerada/constante, menos de 3 picos) vira NaN; árvore e floresta do scikit-learn aceitam NaN.
Exames com as 12 derivações zeradas já saem em `carregar_indices` (mesma regra do A3); os 239 parciais ficam, com NaN onde couber.

Uso (.venv, na raiz):
  python atributos_sinal.py checar   # sinais sintéticos de FC e largura conhecidas
  python atributos_sinal.py medir    # 500 exames de treino: tempo e RAM, estimativa do total (não grava tabela)
  python atributos_sinal.py tudo     # treino e validação -> resultados/atributos_{treino,validacao}.csv.gz
"""
import sys
import time

import h5py
import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt

FS = 400
DERIV = ["DI", "DII", "DIII", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LIDER = 1                     # DII
QRS_DERIV = (1, 6, 11)        # DII, V1, V6
RITMO = ["n_picos", "fc_bpm", "rr_media_ms", "rr_mediana_ms", "rr_desvio_ms", "rmssd_ms", "rr_cv"]
NOMES = RITMO + [f"ptp_{d}" for d in DERIV] + [f"qrs_ms_{DERIV[k]}" for k in QRS_DERIV]
_B_QRS = butter(2, [5, 15], btype="band", fs=FS, output="sos")
_B_FORMA = butter(2, [3, 40], btype="band", fs=FS, output="sos")


def picos_r(x, fs=FS):
    """Posições (amostras) dos picos do QRS numa derivação; array vazio se não der. Mesmo método de fc_estimada (ver cabeçalho)."""
    if len(x) < 2 * fs or np.ptp(x) == 0:
        return np.array([], int)
    y = sosfiltfilt(_B_QRS, x)
    w = int(0.15 * fs)
    e = np.convolve(np.diff(y) ** 2, np.ones(w) / w, mode="same")
    return find_peaks(e, distance=int(0.25 * fs), height=0.3 * np.percentile(e, 99))[0]


def largura_qrs_ms(x, pk, fs=FS):
    """APROXIMAÇÃO (ver cabeçalho). NaN se não houver batimento com janela completa."""
    if len(pk) == 0 or np.ptp(x) == 0:
        return np.nan
    s = np.abs(np.diff(sosfiltfilt(_B_FORMA, x)))
    h, larg = int(0.1 * fs), []
    for p in pk:
        if p - h < 0 or p + h > len(s):
            continue
        j = np.flatnonzero(s[p - h:p + h] > 0.15 * s[p - h:p + h].max())
        larg.append((j[-1] - j[0]) / fs * 1000)
    return float(np.median(larg)) if larg else np.nan


def atributos(x):
    """x: (4096, 12) CRU. Devolve dict com NOMES (NaN onde não dá para calcular)."""
    out = dict.fromkeys(NOMES, np.nan)
    ativo = np.flatnonzero(np.abs(x).max(axis=1) > 0)
    if len(ativo) == 0:
        return out
    x = x[ativo[0]:ativo[-1] + 1].astype(np.float64)
    for k, d in enumerate(DERIV):
        p = np.ptp(x[:, k])
        out[f"ptp_{d}"] = p if p > 0 else np.nan   # derivação zerada ou constante: NaN
    pk = picos_r(x[:, LIDER])
    if np.ptp(x[:, LIDER]) > 0 and len(x) >= 2 * FS:
        out["n_picos"] = len(pk)
    if len(pk) >= 3:
        rr = np.diff(pk) / FS * 1000
        out.update(fc_bpm=60000 / np.median(rr), rr_media_ms=rr.mean(), rr_mediana_ms=np.median(rr), rr_desvio_ms=rr.std(ddof=1),
                   rmssd_ms=np.sqrt(np.mean(np.diff(rr) ** 2)), rr_cv=rr.std(ddof=1) / rr.mean())
        for k in QRS_DERIV:
            out[f"qrs_ms_{DERIV[k]}"] = largura_qrs_ms(x[:, k], pk)
    return out


def tabela_atributos(info, passo=64):
    """info: DataFrame com `parte` e `linha` (como o de construir_indice). Lê o HDF5 em lotes e devolve DataFrame alinhado com info (mesma ordem)."""
    DADOS = _dados()
    linhas = []
    for p, d in info.groupby("parte", sort=True):
        with h5py.File(DADOS / f"exams_part{p}.hdf5", "r") as f:
            ds, ls = f["tracings"], d["linha"].to_numpy()
            for i in range(0, len(ls), passo):
                X = ds[np.sort(ls[i:i + passo])]  # índices crescentes (h5py); d deve estar ordenado por linha dentro da parte
                linhas += [atributos(x) for x in X]
    return pd.DataFrame(linhas, index=info.index, columns=NOMES)


def _dados():
    from pathlib import Path
    return Path(__file__).parent / "code15"


def _paralelo(info, n_jobs):
    """Divide `info` (já ordenado por parte, linha) em blocos contíguos e calcula em processos separados."""
    from joblib import Parallel, delayed
    blocos = np.array_split(np.arange(len(info)), n_jobs * 4)
    partes = Parallel(n_jobs=n_jobs)(delayed(tabela_atributos)(info.iloc[b]) for b in blocos if len(b))
    return pd.concat(partes)


# ---------------------------------------------------------------------------------------------- uso
def ecg_sintetico(bpm, a, seg=7.3, seed=0, fs=FS):
    """QRS gaussiano exp(-(t/a)^2), onda T larga, linha de base e ruído. Só para conferir."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seg * fs)) / fs
    x = 0.2 * np.sin(2 * np.pi * 0.3 * t) + 0.03 * rng.standard_normal(len(t))
    for b in np.arange(0.3, seg, 60 / bpm):
        x += np.exp(-((t - b) / a) ** 2) + 0.35 * np.exp(-((t - b - 0.25) / 0.06) ** 2)
    return x


def checar():
    larg = {}
    for bpm in (45, 75, 110):
        for a in (0.01, 0.03):
            x = np.zeros((4096, 12)); s = ecg_sintetico(bpm, a); x[100:100 + len(s)] = s[:, None]  # DII = s; zeros nas pontas
            r = atributos(x.astype(np.float32))
            assert abs(r["fc_bpm"] - bpm) < 1.5, (bpm, r["fc_bpm"])
            larg[a] = r["qrs_ms_DII"]
    assert 2.0 < larg[0.03] / larg[0.01] < 4.0, larg  # QRS 3x mais largo -> medida bem maior (aproximação: só a ordem e a ordem de grandeza)
    z = atributos(np.zeros((4096, 12), np.float32))
    assert all(np.isnan(v) for v in z.values()), "exame zerado deve dar tudo NaN"
    print(f"checar ok: FC com erro < 1,5 bpm (45-110); largura sintética 0,01 s -> {larg[0.01]:.0f} ms, 0,03 s -> {larg[0.03]:.0f} ms; zerado -> NaN")


def medir(n=500):
    import psutil
    from a1_cnn import carregar_indices
    tr, va, _ = carregar_indices(partes=[0, 1, 2, 3])
    amostra = tr.sample(n, random_state=42).sort_values(["parte", "linha"])
    proc = psutil.Process()
    t = time.perf_counter()
    tab = tabela_atributos(amostra)
    seg = time.perf_counter() - t
    pico = getattr(proc.memory_info(), "peak_wset", proc.memory_info().rss) / 2**30
    total = len(tr) + len(va)
    print(f"{n} exames de TREINO (sorteio seed 42): {seg:.1f} s = {seg / n * 1000:.1f} ms/exame; pico de RAM do processo (com TensorFlow carregado): {pico:.2f} GiB")
    print(f"estimado a partir da medição: {total:,} exames de treino+validação em 1 processo = {seg / n * total / 60:.1f} min; "
          f"com 8 processos ~{seg / n * total / 60 / 8:.1f} min se escalar linearmente (não medido)")
    print("NaN por atributo (nesta amostra):")
    print(tab.isna().mean().round(3).to_string())


def tudo(n_jobs=8):
    from a1_cnn import carregar_indices
    from ecg_dados import ROTULOS
    tr, va, rel = carregar_indices(partes=[0, 1, 2, 3])
    print("descartes (só exames todo-zero, só treino e validação):", rel, flush=True)
    for nome, d in (("treino", tr), ("validacao", va)):
        d = d.sort_values(["parte", "linha"])
        t = time.perf_counter()
        tab = _paralelo(d, n_jobs)
        assert (tab.index == d.index).all()
        saida = pd.concat([d[["exam_id", "patient_id", "parte", "linha", "grupo", "age", "is_male"] + ROTULOS], tab], axis=1)
        arq = _dados().parent / "resultados" / f"atributos_{nome}.csv.gz"
        saida.to_csv(arq, index=False, float_format="%.6g")
        print(f"{nome}: {len(saida):,} exames em {(time.perf_counter() - t) / 60:.1f} min -> {arq.name}", flush=True)


if __name__ == "__main__":
    {"checar": checar, "medir": medir, "tudo": tudo}[sys.argv[1]]()
