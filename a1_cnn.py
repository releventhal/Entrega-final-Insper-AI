"""A1 (T4): CNN 1D base da seção 5.3 do CLAUDE.md. Treina no TREINO, avalia na VALIDAÇÃO.

PROVISÓRIO, A CONFIRMAR COM A RENATA (tudo marcado com [PROV] no código):
  [PROV] só a parte 0 (desenvolvimento);
  [PROV] descarta só os exames com as 12 derivações zeradas, SÓ do treino e da validação (regra pelo sinal, nunca pelos rótulos);
  [PROV] limiar por rótulo = o que maximiza o F1 na validação;
  [PROV] sem normalização no A1 (A2 = normalização por exame e derivação, também provisório).
O teste interno, o PTB-XL e o CODE-TEST NÃO são abertos aqui: o índice é filtrado para treino/validação logo na carga.

Uso (na raiz, com o Python do .venv):
  python a1_cnn.py bench                      # mede o tempo por lote e estima a época
  python a1_cnn.py treinar A1 [--n-treino N]  # treina e avalia na validação
  python a1_cnn.py treinar A2 --normalizar    # A2 provisório (uma mudança: normalização)
"""
import argparse
import time

import keras
import numpy as np
import pandas as pd
from keras import layers

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from a0_baselines import PASTA_RESULTADOS, melhor_limiar_f1, metricas
from ecg_dados import DADOS, PARTES_DEV, ROTULOS, SEED, GeradorECG, construir_indice, exames_todo_zero

PASTA_MODELOS = DADOS.parent / "modelos"  # fora do git
ROTULO_EXPERIMENTO = {"A1": "A1 CNN (provisório, parte 0)", "A2": "A2 provisório (normalização por exame e derivação)",
                      "A3": "A3 provisório (partes 0 a 3; normalização por exame e derivação)"}  # nome na tabela de experimentos


def carregar_indices(n_treino=None, partes=PARTES_DEV):
    """[PROV] parte 0. Descarta SÓ os exames todo-zero (12/12 derivações), SÓ de treino e validação.
    Devolve (treino, validação, relatório). O grupo teste é cortado aqui e nunca chega ao resto do código."""
    info = construir_indice(partes)
    todo_zero = exames_todo_zero(partes)  # lista code15/exames_zerados_p0-3.npy; o arquivo NÃO é alterado
    rel = {}
    saida = {}
    for g in ("treino", "validação"):
        d = info[info.grupo == g]
        sai = d.exam_id.isin(todo_zero)
        rel[g] = dict(antes=len(d), todo_zero_descartados=int(sai.sum()), depois=int((~sai).sum()))
        saida[g] = d[~sai].reset_index(drop=True)
    assert not set(saida["treino"].patient_id) & set(saida["validação"].patient_id), "paciente em treino e validação!"
    if n_treino and n_treino < len(saida["treino"]):  # subconjunto do treino sorteado POR PACIENTE (seed 42): paciente entra com todos os exames
        t = saida["treino"]
        sorteados = np.random.default_rng(SEED).permutation(np.sort(t.patient_id.unique()))
        conta = t.groupby("patient_id").size().reindex(sorteados).cumsum()
        manter = conta.index[: int((conta < n_treino).sum()) + 1]
        saida["treino"] = t[t.patient_id.isin(manter)].reset_index(drop=True)
        rel["treino"]["subconjunto"] = len(saida["treino"])
    return saida["treino"], saida["validação"], rel


def construir_modelo():
    """Arquitetura A1 da seção 5.3, sem mudanças: 4 x Conv1D(32,64,128,128; kernel 15; same; ReLU),
    MaxPool 4 nas três primeiras, GlobalAveragePooling, Dense 6 sigmoid. ~406 mil parâmetros."""
    keras.utils.set_random_seed(SEED)
    m = keras.Sequential([layers.Input((4096, 12))])
    for i, f in enumerate([32, 64, 128, 128]):
        m.add(layers.Conv1D(f, 15, padding="same", activation="relu"))
        if i < 3:
            m.add(layers.MaxPooling1D(4))
    m.add(layers.GlobalAveragePooling1D())
    m.add(layers.Dense(len(ROTULOS), activation="sigmoid"))
    m.compile(optimizer="adam", loss="binary_crossentropy")
    return m


def bench(n_lotes=15, normalizar=False, partes=PARTES_DEV):
    """(b) Tempo por lote de treino (leitura e cálculo separados) e estimativa de uma época completa."""
    tr, va, rel = carregar_indices(partes=partes)
    g = GeradorECG(tr, embaralhar=True, normalizar=normalizar)
    m = construir_modelo()
    print("parâmetros:", m.count_params(), "| treino:", len(tr), "exames =", len(g), "lotes | validação:", len(va), "exames")
    X, y = g[0]
    m.train_on_batch(X, y)  # aquecimento (compilação)
    t_ler = t_cal = 0.0
    for i in range(1, n_lotes + 1):
        t = time.perf_counter(); X, y = g[i]; t_ler += time.perf_counter() - t
        t = time.perf_counter(); m.train_on_batch(X, y); t_cal += time.perf_counter() - t
    gv = GeradorECG(va, normalizar=normalizar)
    t = time.perf_counter()
    for i in range(5):
        Xv, yv = gv[i]; m.predict_on_batch(Xv)
    t_val = (time.perf_counter() - t) / 5
    lote = (t_ler + t_cal) / n_lotes
    print(f"por lote de treino: leitura {t_ler / n_lotes * 1000:.0f} ms + treino {t_cal / n_lotes * 1000:.0f} ms = {lote * 1000:.0f} ms")
    print(f"por lote de validação (leitura + previsão): {t_val * 1000:.0f} ms")
    ep = len(g) * lote + len(gv) * t_val
    print(f"ESTIMATIVA de 1 época completa: treino {len(g) * lote / 60:.1f} min + validação {len(gv) * t_val / 60:.1f} min = {ep / 60:.1f} min")


class LimiteDeTempo(keras.callbacks.Callback):
    """Para o treino ao passar de `minutos` (checa a cada lote). O melhor modelo já foi salvo pelo ModelCheckpoint."""

    def __init__(self, minutos):
        super().__init__()
        self.limite, self.t0, self.estourou = minutos * 60, time.perf_counter(), False

    def on_train_batch_end(self, batch, logs=None):
        if time.perf_counter() - self.t0 > self.limite:
            self.estourou, self.model.stop_training = True, True


def treinar(nome, normalizar=False, n_treino=None, max_min=60, max_epocas=15, paciencia=3, partes=PARTES_DEV):
    PASTA_MODELOS.mkdir(exist_ok=True)
    PASTA_RESULTADOS.mkdir(exist_ok=True)
    tr, va, rel = carregar_indices(n_treino, partes)
    print("descartes (só exames todo-zero, só treino e validação):", rel, flush=True)
    g_tr = GeradorECG(tr, embaralhar=True, normalizar=normalizar)
    g_va = GeradorECG(va, normalizar=normalizar)
    m = construir_modelo()
    caminho = PASTA_MODELOS / f"{nome}_melhor.keras"
    limite = LimiteDeTempo(max_min)
    t0 = time.perf_counter()
    hist = m.fit(g_tr, validation_data=g_va, epochs=max_epocas, verbose=2, callbacks=[
        # early stopping na perda de validação (como na fase 1); paciência 3 = [PROV]
        keras.callbacks.EarlyStopping("val_loss", patience=paciencia, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(caminho, monitor="val_loss", save_best_only=True),
        keras.callbacks.CSVLogger(PASTA_RESULTADOS / f"{nome}_log_por_epoca.csv"),
        limite])
    minutos = (time.perf_counter() - t0) / 60
    h = hist.history
    melhor = int(np.argmin(h["val_loss"]))
    print(f"\n{nome}: {len(h['loss'])} épocas em {minutos:.1f} min (limite de tempo estourou: {limite.estourou}); "
          f"melhor época = {melhor + 1} (val_loss {h['val_loss'][melhor]:.4f})", flush=True)

    # --- avaliação na VALIDAÇÃO com o MELHOR modelo salvo (o EarlyStopping só restaura pesos se disparar;
    # se o treino acabar pelo limite de épocas ou de tempo, o checkpoint é que guarda a melhor época)
    m = keras.models.load_model(caminho)
    S = m.predict(g_va, verbose=0)
    Y = g_va.Y
    assert S.shape == Y.shape == (len(va), len(ROTULOS))  # previsões alinhadas com g_va.info / g_va.Y
    exp = ROTULO_EXPERIMENTO.get(nome, nome)
    df = tabela_metricas(exp, Y, S)
    df["dados"] = f"partes {partes} (provisório, a confirmar com a Renata)"
    df["n_treino"], df["epocas"], df["melhor_epoca"], df["minutos_treino"] = len(tr), len(h["loss"]), melhor + 1, round(minutos, 1)
    partes_df = [df]
    sub = (g_va.info.parte == 0).to_numpy()
    if len(partes) > 1 and sub.any():
        # comparação justa com o A1/A2: os MESMOS exames de validação da parte 0 (2.069), em duas versões do limiar
        lim_completo = {r: melhor_limiar_f1(Y[:, k].astype(int), S[:, k]) for k, r in enumerate(ROTULOS)}  # limiar da validação completa
        for sufixo, lims in ((f" | só os {sub.sum()} exames de validação da parte 0 (limiar da validação completa)", lim_completo),
                             (f" | só os {sub.sum()} exames de validação da parte 0 (limiar reescolhido neles, como no A1/A2)", None)):
            d = tabela_metricas(exp + sufixo, Y[sub], S[sub], lims)
            d["dados"] = f"partes {partes} (provisório, a confirmar com a Renata)"
            d["n_treino"], d["epocas"], d["melhor_epoca"], d["minutos_treino"] = len(tr), len(h["loss"]), melhor + 1, round(minutos, 1)
            partes_df.append(d)
    df = pd.concat(partes_df, ignore_index=True)
    np.save(PASTA_RESULTADOS / f"{nome}_previsoes_validacao.npy", S.astype(np.float32))
    print(df.round(3).to_string(index=False), flush=True)
    return pd.concat([df, a0b_mesmo_conjunto(tr, va, nome, partes)], ignore_index=True), S, g_va


def tabela_metricas(experimento, Y, S, limiares=None):
    """Métricas por rótulo + média macro. `limiares` = dict rótulo -> limiar; None = maximiza o F1 nestes mesmos exames."""
    linhas = []
    for k, r in enumerate(ROTULOS):
        y, s = Y[:, k].astype(int), S[:, k]
        lim = melhor_limiar_f1(y, s) if limiares is None else limiares[r]
        linhas.append(dict(experimento=experimento, rotulo=r, n_val=len(y), positivos_val=int(y.sum()), **metricas(y, s, lim)))
    df = pd.DataFrame(linhas)
    cols = ["auroc", "auprc", "sensibilidade", "precisao", "f1", "especificidade", "acuracia"]
    macro = df[cols].mean().to_dict() | dict(experimento=experimento, rotulo="MACRO", n_val=len(Y), positivos_val=np.nan, limiar=np.nan)
    return pd.concat([df, pd.DataFrame([macro])], ignore_index=True)


def a0b_mesmo_conjunto(tr, va, nome="A1", partes=PARTES_DEV):
    """A0b (idade + sexo) treinado e avaliado EXATAMENTE nos mesmos exames do A1, para comparação justa
    (o A0b do CSV original usa as partes 0 a 3; este usa o mesmo conjunto do A1)."""
    esc = StandardScaler().fit(tr[["age"]])
    f = lambda d: np.column_stack([esc.transform(d[["age"]]), d["is_male"].astype(float)])
    linhas = []
    for r in ROTULOS:
        s = LogisticRegression(max_iter=1000, random_state=SEED).fit(f(tr), tr[r]).predict_proba(f(va))[:, 1]
        y = va[r].to_numpy()
        linhas.append(dict(experimento=f"A0b idade+sexo (mesmo conjunto do {nome})", rotulo=r, n_val=len(y), positivos_val=int(y.sum()),
                           **metricas(y, s, melhor_limiar_f1(y, s))))
    df = pd.DataFrame(linhas)
    cols = ["auroc", "auprc", "sensibilidade", "precisao", "f1", "especificidade", "acuracia"]
    macro = df[cols].mean().to_dict() | dict(experimento=df.experimento[0], rotulo="MACRO", n_val=len(va), positivos_val=np.nan, limiar=np.nan)
    df = pd.concat([df, pd.DataFrame([macro])], ignore_index=True)
    df["dados"] = f"partes {partes} (provisório, a confirmar com a Renata)"
    df["n_treino"] = len(tr)
    return df


def gravar_no_csv(df):
    """Acrescenta as linhas em resultados/experimentos.csv (troca as do mesmo experimento, mantém as outras)."""
    arq = PASTA_RESULTADOS / "experimentos.csv"
    antigo = pd.read_csv(arq) if arq.exists() else pd.DataFrame()
    if len(antigo):
        antigo = antigo[~antigo.experimento.isin(df.experimento.unique())]
    pd.concat([antigo, df], ignore_index=True).to_csv(arq, index=False, float_format="%.4f")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=["bench", "treinar"])
    ap.add_argument("nome", nargs="?", default="A1")
    ap.add_argument("--normalizar", action="store_true")
    ap.add_argument("--n-treino", type=int, default=None)
    ap.add_argument("--max-min", type=float, default=60)
    ap.add_argument("--max-epocas", type=int, default=15)
    ap.add_argument("--partes", default="0")  # ex.: 0,1,2,3
    a = ap.parse_args()
    if a.modo == "bench":
        bench(normalizar=a.normalizar, partes=[int(x) for x in a.partes.split(",")])
    else:
        df, _, _ = treinar(a.nome, a.normalizar, a.n_treino, a.max_min, a.max_epocas, partes=[int(x) for x in a.partes.split(",")])
        gravar_no_csv(df)
