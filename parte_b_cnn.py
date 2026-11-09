"""Parte B com CNN (T7): o ECG prevê óbito em até 2 anos além da idade e do sexo? SÓ TREINO E VALIDAÇÃO.

Decisões da Renata (09/11): 8 partes de sinal (0 a 7), transfer learning a partir do A3 E uma CNN do zero para comparar,
tudo treinado no computador da Renata. Horizonte de 2 anos (declarado antes, em 08/11).

Modelos, todos avaliados nos MESMOS pacientes de validação:
  B0  regressão logística: idade + sexo                                  (como em parte_b_tabular.py)
  B1  B0 + idade do ECG (nn_predicted_age) + diferença                   (como em parte_b_tabular.py)
  B2z CNN do zero, só o sinal (mesma arquitetura do A3, 1 saída sigmoid)
  B2t transfer learning: as camadas de convolução do A3 CONGELADAS viram um extrator de 128 características do ECG;
      só a camada final (1 sigmoid) é treinada sobre elas. Uma camada sigmoid sobre características fixas
      é exatamente uma regressão logística, então ela é ajustada com o scikit-learn.
  B3  B2t + idade + sexo (as 128 características do ECG mais idade e sexo na mesma camada final)
Uma mudança por vez: B2z -> B2t (de onde a rede começa) e B2t -> B3 (entra idade e sexo).

Regras (CLAUDE.md): split por paciente de sempre; o grupo teste é cortado logo na carga e NUNCA é lido;
exames com as 12 derivações zeradas saem (pelo sinal, nunca pelos rótulos); normalização por exame do A2 (a mesma do A3).
Sem vazamento pelo A3: ele foi treinado só com pacientes do grupo treino (mesmo split), então nunca viu a validação.

Uso (raiz do repositório, Python do .venv; as etapas gravam em resultados/ e modelos/ e podem ser rodadas separadas):
  python parte_b_cnn.py preparar            # índice, regra do horizonte, exames zerados, contagens
  python parte_b_cnn.py caracteristicas     # passa os exames pelo A3 congelado e guarda as 128 características
  python parte_b_cnn.py bench               # tempo por lote da CNN do zero
  python parte_b_cnn.py treinar_zero        # B2z (o mais demorado)
  python parte_b_cnn.py avaliar             # B0, B1, B2t, B3 e (se existir) B2z; IC por bootstrap
  Para um teste rápido do encanamento: --partes 0 --saida <pasta temporária> --max-epocas 1
"""
import argparse
import json
import time
from pathlib import Path

import h5py
import keras
import numpy as np
import pandas as pd
from keras import layers
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import ecg_dados as E
from a1_cnn import LimiteDeTempo
from parte_b_tabular import ajustar, features, rotular

RAIZ = Path(__file__).parent
HORIZONTE = 2
PARTES_B = list(range(8))
MODELO_A3 = RAIZ / "modelos" / "A3_melhor.keras"
SHA_A3 = "bae64fa994934072434853ce6c5f6dfffd485a1f126336be1d6e8d526ca12a89"  # o do congelamento (congelado_candidato_A3.json)
N_BOOT = 1000


def nome_partes(partes):
    return f"p{partes[0]}-{partes[-1]}" if len(partes) > 1 else f"p{partes[0]}"


# ------------------------------------------------------------------------------------------------ dados
def todo_zero_por_sinal(info, cache):
    """exam_id dos exames de `info` com as 12 derivações zeradas, lendo o SINAL (nunca os rótulos). Guarda em cache."""
    if cache.exists():
        return set(np.load(cache).tolist())
    achados = []
    for p, d in info.groupby("parte"):
        with h5py.File(E.DADOS / f"exams_part{p}.hdf5", "r") as f:
            linhas = np.sort(d["linha"].to_numpy())
            ids = f["exam_id"][:]
            for ini in range(0, len(linhas), 500):
                bloco = linhas[ini:ini + 500]
                zero = E.todo_zero(f["tracings"][bloco])           # h5py lê índices crescentes
                achados += ids[bloco[zero]].tolist()
    np.save(cache, np.array(achados, dtype=np.int64))
    return set(achados)


def carregar(partes, saida):
    """Primeiro exame de cada paciente (o único com death/timey) nas `partes`, SÓ treino e validação,
    sem exames todo-zero, com a regra do horizonte. Devolve (elegíveis com coluna y, relatório)."""
    info = E.construir_indice(partes)
    info = info[info["grupo"].isin(["treino", "validação"])]           # o grupo teste para aqui e nunca é usado
    info = info[info["death"].notna()].copy()
    assert info["patient_id"].is_unique, "mais de um exame com death por paciente"
    info["death"] = info["death"].astype(bool)
    info["delta"] = info["nn_predicted_age"] - info["age"]
    zerados = todo_zero_por_sinal(info, saida / f"parte_b_todo_zero_{nome_partes(partes)}.npy")
    sai = info["exam_id"].isin(zerados)
    elegiveis, censurado = rotular(info[~sai], HORIZONTE)
    rel = {g: dict(primeiros_exames=int((info.grupo == g).sum()), todo_zero=int((sai & (info.grupo == g)).sum()),
                   censurados=int((censurado & (info[~sai].grupo == g)).sum()), elegiveis=int((elegiveis.grupo == g).sum()),
                   obitos=int(elegiveis.loc[elegiveis.grupo == g, "y"].sum())) for g in ("treino", "validação")}
    assert not set(elegiveis.loc[elegiveis.grupo == "treino", "patient_id"]) & set(elegiveis.loc[elegiveis.grupo == "validação", "patient_id"])
    return elegiveis, rel


class GeradorObito(E.GeradorECG):
    """GeradorECG com o rótulo de óbito (y) no lugar das 6 alterações. Mesma leitura e mesma normalização do A3."""

    def __init__(self, info, **kw):
        super().__init__(info, normalizar=True, **kw)
        self.Y = self.info["y"].to_numpy(np.float32)[:, None]      # self.info já está na ordem (parte, linha)


# ------------------------------------------------------------------------------------------------ modelos
def extrator_a3():
    """O A3 congelado até o GlobalAveragePooling: entra o ECG, saem 128 números (as 'características' que ele aprendeu)."""
    import hashlib
    assert hashlib.sha256(MODELO_A3.read_bytes()).hexdigest() == SHA_A3, "o A3 não é o congelado"
    a3 = keras.models.load_model(MODELO_A3, compile=False)
    x = entrada = keras.Input((4096, 12))
    for camada in a3.layers[:-1]:                                    # tudo menos a camada final de 6 saídas
        x = camada(x)
    return keras.Model(entrada, x)


def cnn_do_zero(seed=E.SEED):
    """B2z: a mesma arquitetura do A1/A3, com 1 saída sigmoid (morreu em 2 anos?). Pesos iniciais sorteados (do zero)."""
    keras.utils.set_random_seed(seed)
    m = keras.Sequential([layers.Input((4096, 12))])
    for i, f in enumerate([32, 64, 128, 128]):
        m.add(layers.Conv1D(f, 15, padding="same", activation="relu"))
        if i < 3:
            m.add(layers.MaxPooling1D(4))
    m.add(layers.GlobalAveragePooling1D())
    m.add(layers.Dense(1, activation="sigmoid"))
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=[keras.metrics.AUC(name="auroc")])
    return m


# ------------------------------------------------------------------------------------------------ etapas
def preparar(a):
    el, rel = carregar(a.partes, a.saida)
    print(json.dumps(rel, indent=1, ensure_ascii=False))
    return el


def caracteristicas(a):
    el, rel = carregar(a.partes, a.saida)
    g = GeradorObito(el)
    t0 = time.time()
    F = extrator_a3().predict(g, verbose=0)
    np.savez(a.saida / f"parte_b_caracteristicas_A3_{nome_partes(a.partes)}.npz", F=F, exam_id=g.info["exam_id"].to_numpy())
    print(f"características do A3: {F.shape} em {(time.time() - t0) / 60:.1f} min")


def bench(a, n=10):
    el, _ = carregar(a.partes, a.saida)
    g = GeradorObito(el[el.grupo == "treino"], embaralhar=True)
    m = cnn_do_zero()
    m.train_on_batch(*g[0])
    t_ler = t_cal = 0.0
    for i in range(1, n + 1):
        t = time.perf_counter(); X, y = g[i]; t_ler += time.perf_counter() - t
        t = time.perf_counter(); m.train_on_batch(X, y); t_cal += time.perf_counter() - t
    lote = (t_ler + t_cal) / n
    print(f"lote de 64: leitura {t_ler / n * 1000:.0f} ms + treino {t_cal / n * 1000:.0f} ms | {len(g)} lotes por época "
          f"-> ~{len(g) * lote / 60:.0f} min por época só de treino (estimativa)")


def treinar_zero(a):
    el, rel = carregar(a.partes, a.saida)
    print("dados:", rel, flush=True)
    g_tr = GeradorObito(el[el.grupo == "treino"], embaralhar=True)
    g_va = GeradorObito(el[el.grupo == "validação"])
    m = cnn_do_zero()
    caminho = a.modelos / f"B2z_{nome_partes(a.partes)}_melhor.keras"
    limite = LimiteDeTempo(a.max_min)
    t0 = time.time()
    hist = m.fit(g_tr, validation_data=g_va, epochs=a.max_epocas, verbose=2, callbacks=[
        # para pelo AUROC de validação (a perda quase não muda com 2,6% de óbitos); paciência 3, como no A3
        keras.callbacks.EarlyStopping("val_auroc", mode="max", patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(caminho, monitor="val_auroc", mode="max", save_best_only=True),
        keras.callbacks.CSVLogger(a.saida / f"parte_b_B2z_{nome_partes(a.partes)}_log_por_epoca.csv"),
        limite])
    h = hist.history
    melhor = int(np.argmax(h["val_auroc"]))
    print(f"B2z: {len(h['loss'])} épocas em {(time.time() - t0) / 60:.1f} min (limite de tempo: {limite.estourou}); "
          f"melhor época {melhor + 1} (val_auroc {h['val_auroc'][melhor]:.4f})", flush=True)
    m = keras.models.load_model(caminho, compile=False)              # o melhor salvo, mesmo se parar pelo tempo
    S = m.predict(g_va, verbose=0)[:, 0]
    np.savez(a.saida / f"parte_b_B2z_{nome_partes(a.partes)}_previsoes_validacao.npz", S=S, exam_id=g_va.info["exam_id"].to_numpy(),
             epocas=len(h["loss"]), melhor_epoca=melhor + 1, minutos=(time.time() - t0) / 60)


def bootstrap(y, notas, pares, n=N_BOOT, seed=E.SEED):
    """IC 95% (percentis 2,5 e 97,5) do AUROC e da AUPRC de cada modelo e das diferenças pareadas.
    1 linha = 1 paciente (primeiro exame), então reamostrar linhas = reamostrar pacientes."""
    rng = np.random.default_rng(seed)
    amostras = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        r = {}
        for m, s in notas.items():
            r[("auroc", m)], r[("auprc", m)] = roc_auc_score(y[i], s[i]), average_precision_score(y[i], s[i])
        for m1, m0 in pares:
            for met in ("auroc", "auprc"):
                r[(met, f"{m1} - {m0}")] = r[(met, m1)] - r[(met, m0)]
        amostras.append(r)
    df = pd.DataFrame(amostras)
    return {k: tuple(np.percentile(df[k], [2.5, 97.5])) for k in df.columns}


def avaliar(a):
    tag = nome_partes(a.partes)
    el, rel = carregar(a.partes, a.saida)
    tr, va = el[el.grupo == "treino"], el[el.grupo == "validação"].sort_values(["parte", "linha"], ignore_index=True)
    y = va["y"].to_numpy()

    # B0 e B1: as mesmas funções de parte_b_tabular.py, ajustadas nos MESMOS pacientes de treino
    esc, mods = ajustar(tr)
    notas = {m: mods[m].predict_proba(features(va, esc, m))[:, 1] for m in ("B0", "B1")}

    # B2t e B3: camada final sobre as características do A3 congelado
    c = np.load(a.saida / f"parte_b_caracteristicas_A3_{tag}.npz")
    F = pd.DataFrame(c["F"], index=c["exam_id"])
    Ftr, Fva = F.loc[tr["exam_id"]].to_numpy(), F.loc[va["exam_id"]].to_numpy()
    tab = lambda d: np.column_stack([d["age"], d["is_male"].astype(float)])
    for nome, Xtr, Xva in (("B2t", Ftr, Fva), ("B3", np.hstack([Ftr, tab(tr)]), np.hstack([Fva, tab(va)]))):
        e = StandardScaler().fit(Xtr)                                  # escala só do treino
        lr = LogisticRegression(max_iter=2000, random_state=E.SEED).fit(e.transform(Xtr), tr["y"])
        notas[nome] = lr.predict_proba(e.transform(Xva))[:, 1]

    # B2z (CNN do zero), se já foi treinada
    arq = a.saida / f"parte_b_B2z_{tag}_previsoes_validacao.npz"
    if arq.exists():
        z = np.load(arq)
        notas["B2z"] = pd.Series(z["S"], index=z["exam_id"]).loc[va["exam_id"]].to_numpy()

    ordem = [m for m in ("B0", "B1", "B2z", "B2t", "B3") if m in notas]
    pares = [(m, "B0") for m in ordem if m != "B0"] + [("B3", "B1"), ("B2t", "B2z")] * ("B2z" in notas)
    pares = list(dict.fromkeys(pares + [("B3", "B1")]))
    ic = bootstrap(y, notas, pares)
    linhas = []
    for m in ordem + [f"{m1} - {m0}" for m1, m0 in pares]:
        lin = dict(modelo=m, n_val=len(y), obitos_val=int(y.sum()))
        for met in ("auroc", "auprc"):
            if m in notas:
                lin[met] = (roc_auc_score if met == "auroc" else average_precision_score)(y, notas[m])
            else:
                m1, m0 = m.split(" - ")
                lin[met] = (roc_auc_score if met == "auroc" else average_precision_score)(y, notas[m1]) - \
                           (roc_auc_score if met == "auroc" else average_precision_score)(y, notas[m0])
            lin[f"{met}_ic_inf"], lin[f"{met}_ic_sup"] = ic[(met, m)]
        linhas.append(lin)
    res = pd.DataFrame(linhas)
    res["partes"], res["n_treino"], res["obitos_treino"] = tag, len(tr), int(tr["y"].sum())
    res.to_csv(a.saida / f"parte_b_cnn_{tag}.csv", index=False, float_format="%.4f")
    pd.DataFrame({"exam_id": va["exam_id"], "patient_id": va["patient_id"], "age": va["age"], "is_male": va["is_male"], "y": y, **notas}) \
        .to_csv(a.saida / f"parte_b_cnn_{tag}_previsoes_validacao.csv", index=False, float_format="%.6f")
    pd.set_option("display.width", 200)
    print("dados:", rel)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=["preparar", "caracteristicas", "bench", "treinar_zero", "avaliar"])
    ap.add_argument("--partes", default=",".join(map(str, PARTES_B)))
    ap.add_argument("--saida", default=str(RAIZ / "resultados"))
    ap.add_argument("--modelos", default=str(RAIZ / "modelos"))
    ap.add_argument("--max-epocas", type=int, default=15)
    ap.add_argument("--max-min", type=float, default=240)
    a = ap.parse_args()
    a.partes = [int(p) for p in a.partes.split(",")]
    a.saida, a.modelos = Path(a.saida), Path(a.modelos)
    a.saida.mkdir(exist_ok=True)
    a.modelos.mkdir(exist_ok=True)
    {"preparar": preparar, "caracteristicas": caracteristicas, "bench": bench, "treinar_zero": treinar_zero, "avaliar": avaliar}[a.etapa](a)
