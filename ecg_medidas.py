"""Medidas "de médico" extraídas de cada ECG, para a random forest da Parte A (fora das aulas: neurokit2 e random forest).

Sem TensorFlow aqui de propósito: este módulo roda em vários processos ao mesmo tempo.
Tudo é calculado no sinal CRU (mV, sem a normalização do A2), só na janela com sinal (sem os zeros das pontas).

Medidas (NaN quando não dá para medir; a random forest do scikit-learn aceita NaN):
  ritmo (derivação II, picos R pelo neurokit2): FC média e mediana, RR (desvio, CV, RMSSD, irregularidade entre
      batimentos seguidos), nº de batimentos -> bradicardia (SB), taquicardia (ST), fibrilação atrial (AF, RR irregular);
  intervalos (delineamento do neurokit2, métodos "peak" e "dwt"): P->R, P->Q, Q->S, QRS (dwt), QT e QTc (Bazett),
      fração de batimentos com onda P achada -> 1dAVb (PR > 200 ms) e AF (sem onda P);
  forma (batimento mediano de cada uma das 12 derivações, alinhado nos picos R): no QRS (-60 a +120 ms do R) máximo,
      mínimo, área, posição do máximo e do mínimo; na onda T (+150 a +400 ms) máximo e mínimo; amplitude da onda P
      (-250 a -80 ms) na II e na V1; largura do QRS pelo batimento mediano das 12 derivações juntas -> bloqueios de ramo
      (RBBB: R' tardio em V1; LBBB: QRS largo, S profundo em V1, R largo em V6).
"""
import warnings

import h5py
import numpy as np

FS = 400
MS = 1000 / FS
DERIV = ["DI", "DII", "DIII", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _mediana(v):
    v = np.asarray(v, float)
    return float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan


def _amostras(ms):
    return int(round(ms / MS))


def medidas_exame(x):
    """x: (4096, 12) sinal cru. Devolve dict nome -> valor."""
    import neurokit2 as nk
    out = {}
    nz = np.nonzero(np.abs(x).max(1))[0]
    if len(nz) < 2 * FS:
        return out
    x = x[nz[0]:nz[-1] + 1].astype(np.float64)
    # derivação de referência para os picos R: a II, ou a de maior desvio se a II estiver zerada
    ref = 1 if np.abs(x[:, 1]).max() > 0 else int(np.argmax(x.std(0)))
    out["deriv_ref_II"] = float(ref == 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            c = nk.ecg_clean(x[:, ref], sampling_rate=FS)
            _, info = nk.ecg_peaks(c, sampling_rate=FS)
            R = np.asarray(info["ECG_R_Peaks"], int)
        except Exception:
            return out
        if len(R) < 3:
            out["n_batimentos"] = float(len(R))
            return out
        rr = np.diff(R) * MS
        out.update(n_batimentos=float(len(R)), fc_media=60000 / rr.mean(), fc_mediana=60000 / np.median(rr),
                   rr_desvio=rr.std(), rr_cv=rr.std() / rr.mean(), rmssd=np.sqrt(np.mean(np.diff(rr) ** 2)) if len(rr) > 1 else np.nan,
                   rr_irreg=np.median(np.abs(np.diff(rr))) / rr.mean() if len(rr) > 1 else np.nan,
                   rr_min_max=rr.min() / rr.max())
        # intervalos pelo delineamento
        for metodo in ("peak", "dwt"):
            try:
                _, w = nk.ecg_delineate(c, R, sampling_rate=FS, method=metodo)
                g = {k: np.asarray(v, float)[:len(R)] for k, v in w.items()}
                Rf = R[:len(g["ECG_P_Peaks"])].astype(float)
                if metodo == "peak":
                    out["p_r_peak"] = _mediana((Rf - g["ECG_P_Peaks"]) * MS)
                    out["q_s_peak"] = _mediana((g["ECG_S_Peaks"] - g["ECG_Q_Peaks"]) * MS)
                    out["frac_p_peak"] = float(np.isfinite(g["ECG_P_Peaks"]).mean())
                    out["r_t_peak"] = _mediana((g["ECG_T_Peaks"] - Rf) * MS)
                else:
                    out["p_q_dwt"] = _mediana((g["ECG_Q_Peaks"] - g["ECG_P_Onsets"]) * MS)
                    out["pr_dwt"] = _mediana((g["ECG_R_Onsets"] - g["ECG_P_Onsets"]) * MS)
                    out["qrs_dwt"] = _mediana((g["ECG_R_Offsets"] - g["ECG_R_Onsets"]) * MS)
                    qt = _mediana((g["ECG_T_Offsets"] - g["ECG_R_Onsets"]) * MS)
                    out["qt_dwt"] = qt
                    out["qtc_dwt"] = qt / np.sqrt(np.median(rr) / 1000) if np.isfinite(qt) else np.nan
                    out["frac_p_dwt"] = float(np.isfinite(g["ECG_P_Onsets"]).mean())
            except Exception:
                pass
    # batimento mediano das 12 derivações (-250 a +450 ms do pico R)
    a, b = _amostras(250), _amostras(450)
    Rv = R[(R - a >= 0) & (R + b < len(x))]
    if len(Rv) < 2:
        return out
    bat = np.stack([x[r - a:r + b] for r in Rv])                       # (n_bat, janela, 12)
    bat = bat - np.median(bat[:, :_amostras(60)], axis=1, keepdims=True)  # linha de base: início da janela (antes da P)
    t = np.median(bat, axis=0)                                         # (janela, 12)
    z = a                                                             # índice do pico R na janela
    q0, q1 = z - _amostras(60), z + _amostras(120)
    t0, t1 = z + _amostras(150), z + _amostras(400)
    p0, p1 = z - _amostras(250) + _amostras(60), z - _amostras(80)
    for j, d in enumerate(DERIV):
        qrs = t[q0:q1, j]
        out[f"{d}_qrs_max"] = qrs.max()
        out[f"{d}_qrs_min"] = qrs.min()
        out[f"{d}_qrs_area"] = qrs.sum() * MS / 1000
        out[f"{d}_qrs_tmax"] = (np.argmax(qrs) + q0 - z) * MS
        out[f"{d}_qrs_tmin"] = (np.argmin(qrs) + q0 - z) * MS
        out[f"{d}_t_max"] = t[t0:t1, j].max()
        out[f"{d}_t_min"] = t[t0:t1, j].min()
    for j, d in ((1, "DII"), (6, "V1")):
        out[f"{d}_p_amp"] = t[p0:p1, j].max() - t[p0:p1, j].min()
    # largura do QRS: onde a "energia" (|derivada| somada nas 12 derivações) passa de 20% do pico, perto do R
    e = np.abs(np.diff(t, axis=0)).sum(1)
    janela = e[z - _amostras(120):z + _amostras(200)]
    acima = np.nonzero(janela > 0.2 * janela.max())[0]
    out["qrs_largura_mediano"] = (acima[-1] - acima[0]) * MS if len(acima) else np.nan
    out["r_amp_ref"] = t[z, ref]
    return out


def medidas_lote(tarefa):
    """tarefa = (caminho do hdf5, linhas, exam_ids). Roda num processo separado."""
    caminho, linhas, ids = tarefa
    ordem = np.argsort(linhas)
    linhas, ids = np.asarray(linhas)[ordem], np.asarray(ids)[ordem]
    res = []
    with h5py.File(caminho, "r") as f:
        X = f["tracings"][linhas]                                     # h5py lê índices crescentes
    for x, i in zip(X, ids):
        todo_zero = bool((np.abs(x).max(axis=0) == 0).all())          # mesma regra de ecg_dados.todo_zero (pelo sinal)
        m = {} if todo_zero else medidas_exame(x)
        m["exam_id"], m["todo_zero"] = int(i), todo_zero
        res.append(m)
    return res
