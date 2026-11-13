"""A0e (13/11): extrai com `ecg_medidas.medidas_exame` (neurokit2) as medidas FC, p_r_peak, qrs_dwt, qt_dwt (e algumas de apoio) de TODOS os exames
de TREINO e VALIDAÇÃO das partes 0 a 3 (a mesma lista de exames de `atributos_sinal.py`; o teste NÃO é aberto: o índice vem de a1_cnn.carregar_indices).
Não altera `ecg_medidas.py` nem nenhum arquivo da Fase 1/2. Caches NOVOS (abortam se já existirem):
  resultados/atributos_pr_treino.csv.gz, resultados/atributos_pr_validacao.csv.gz
Colunas: exam_id, patient_id, parte, linha, as medidas, e `ausente_<medida>` = 1 quando `medidas_exame` nem devolveu a chave (falha/curto demais),
em oposição a NaN (chave devolvida, valor NaN). `pr_dwt` fica gravado SÓ para diagnóstico (não é usado no A0e: valores fisiologicamente impossíveis).

  python atributos_pr.py fumaca   # 200 exames de treino, 2 processos, confere contra 1 processo; NÃO grava nada
  python atributos_pr.py tudo     # extração completa (8 processos, blocos de 200 exames)
"""
import os
import sys
import time
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(v, "1")  # 1 thread por processo, para 8 processos não brigarem por núcleos

import h5py
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
DADOS, RES = RAIZ / "code15", RAIZ / "resultados"
MEDIDAS = ["fc_media", "fc_mediana", "n_batimentos", "p_r_peak", "frac_p_peak", "pr_dwt", "qrs_dwt", "qt_dwt", "qtc_dwt", "qrs_largura_mediano"]
BLOCO = 200


def trabalho(tarefa):
    """tarefa = (parte, linhas ordenadas, exam_ids). Roda num processo separado; devolve lista de dicts."""
    from ecg_medidas import medidas_exame
    parte, linhas, ids = tarefa
    with h5py.File(DADOS / f"exams_part{parte}.hdf5", "r") as f:
        X = f["tracings"][np.asarray(linhas)]
    res = []
    for x, i in zip(X, ids):
        t0 = time.perf_counter()
        m = medidas_exame(x)
        r = {"exam_id": int(i), "segundos": time.perf_counter() - t0}
        for k in MEDIDAS:
            r[k] = float(m[k]) if k in m else np.nan
            r[f"ausente_{k}"] = int(k not in m)
        res.append(r)
    return res


def tarefas(info):
    info = info.sort_values(["parte", "linha"])
    out = []
    for p, d in info.groupby("parte", sort=True):
        for i in range(0, len(d), BLOCO):
            b = d.iloc[i:i + BLOCO]
            out.append((int(p), b["linha"].to_numpy(), b["exam_id"].to_numpy()))
    return out


def extrair(info, n_proc, rotulo=""):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ts = tarefas(info)
    linhas, t0 = [], time.perf_counter()
    if n_proc <= 1:
        for k, t in enumerate(ts):
            linhas += trabalho(t)
    else:
        with ProcessPoolExecutor(max_workers=n_proc) as ex:
            fut = [ex.submit(trabalho, t) for t in ts]
            for k, f in enumerate(as_completed(fut)):
                linhas += f.result()
                if k % 20 == 0:
                    print(f"{rotulo} bloco {k + 1}/{len(ts)}  {len(linhas):,} exames  {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return pd.DataFrame(linhas)


def fumaca():
    from a1_cnn import carregar_indices
    tr, _, _ = carregar_indices(partes=[0, 1, 2, 3])
    amostra = tr.sample(200, random_state=42)
    global BLOCO
    BLOCO = 100
    t0 = time.perf_counter(); d1 = extrair(amostra, 1).set_index("exam_id").sort_index(); s1 = time.perf_counter() - t0
    t0 = time.perf_counter(); d2 = extrair(amostra, 2).set_index("exam_id").sort_index(); s2 = time.perf_counter() - t0
    print(f"1 processo: {s1:.1f} s ({s1 / 200 * 1000:.0f} ms/exame, inclui leitura do HDF5) | 2 processos: {s2:.1f} s")
    cols = [c for c in d1.columns if c != "segundos"]
    iguais = np.allclose(d1[cols].to_numpy(float), d2[cols].to_numpy(float), equal_nan=True)
    print("resultado de 2 processos == 1 processo (200 exames, todas as medidas):", iguais)
    print("sem valor (NaN ou ausente) por medida, de 200:", d2[MEDIDAS].isna().sum().to_dict())
    print("segundos por exame dentro do worker: média %.0f ms" % (d2.segundos.mean() * 1000))


def tudo(n_proc=8):
    from a1_cnn import carregar_indices
    tr, va, rel = carregar_indices(partes=[0, 1, 2, 3])
    print("descartes (só exames todo-zero, só treino e validação):", rel, flush=True)
    for nome, d in (("treino", tr), ("validacao", va)):
        arq = RES / f"atributos_pr_{nome}.csv.gz"
        assert not arq.exists(), f"já existe, não sobrescrevo: {arq}"
    for nome, d in (("treino", tr), ("validacao", va)):
        arq = RES / f"atributos_pr_{nome}.csv.gz"
        t0 = time.perf_counter()
        tab = extrair(d, n_proc, nome)
        d = d.sort_values(["parte", "linha"])[["exam_id", "patient_id", "parte", "linha"]]
        saida = d.merge(tab.drop(columns="segundos"), on="exam_id", how="left", validate="1:1")
        assert len(saida) == len(d) and saida[MEDIDAS].shape[0] == len(d)
        assert not arq.exists(), f"já existe, não sobrescrevo: {arq}"
        saida.to_csv(arq, index=False, float_format="%.6g")
        print(f"{nome}: {len(saida):,} exames em {(time.perf_counter() - t0) / 60:.1f} min; segundos médios por exame no worker {tab.segundos.mean() * 1000:.0f} ms -> {arq}", flush=True)


if __name__ == "__main__":
    {"fumaca": fumaca, "tudo": tudo}[sys.argv[1]]()
