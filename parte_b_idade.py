"""Parte B: B5 = B3z com PRÉ-TREINO DE IDADE. SÓ TREINO -> VALIDAÇÃO; o teste da Parte B não é aberto.
Fora das aulas no detalhe (pré-treino numa tarefa auxiliar), mas é a mesma ideia do transfer learning da aula 10.

Ideia (sugestão da intermediária de 11/11: "modelo próprio de idade pelo ECG"): óbitos são raros (2,6%), mas TODO
paciente tem idade. Então primeiro a CNN aprende a estimar a idade pelo ECG (muitos exemplos), e só depois aprende o
óbito, partindo do que aprendeu sobre "envelhecimento" do ECG.

Etapa 1  idade   CNN com a MESMA arquitetura do B2z/B3z (4 Conv1D + GlobalAveragePooling), saída linear (idade
                 padronizada), perda MSE. Dados: o primeiro exame de cada paciente de TREINO, inclusive os censurados
                 (a idade não depende do seguimento), sem exames todo-zero. Early stopping no MAE da validação
                 (primeiros exames dos pacientes de validação). Compara o MAE com o `nn_predicted_age` (Lima et al.)
                 nos mesmos pacientes de validação.
Etapa 2  B5      O B3z (mesma arquitetura, idade e sexo juntados depois do pooling) com as 4 convoluções INICIADAS
                 pelos pesos da etapa 1, em vez de sorteados. Uma mudança em relação ao B3z: só os pesos iniciais.
                 Mesmo treino do B3z (Adam, lote 64, early stopping no AUROC de validação, paciência 3).
Saídas: modelos/idade_<tag>_melhor.keras, modelos/B5_<tag>_melhor.keras, resultados/parte_b_idade_<tag>.csv,
        resultados/parte_b_idade_<tag>_log_por_epoca.csv, resultados/parte_b_B5_<tag>_log_por_epoca.csv,
        resultados/parte_b_B5_<tag>_previsoes_validacao.npz (mesmo formato do B3z, lido por parte_b_comparar.py).

Uso: python parte_b_idade.py idade --partes 0,...,17   e depois   python parte_b_idade.py b5 --partes 0,...,17
(precisa do cache de exames todo-zero: python parte_b_cnn.py preparar --partes ...)
"""
import argparse
import time
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from keras import layers

import ecg_dados as E
from a1_cnn import LimiteDeTempo
from parte_b_cnn import GeradorObito, carregar, cnn_do_zero_com_idade, nome_partes, todo_zero_por_sinal

RAIZ = Path(__file__).parent


def primeiros_exames(partes, saida):
    """Primeiro exame (o que tem death) de cada paciente de treino e validação, sem os todo-zero. Sem regra do horizonte."""
    info = E.construir_indice(partes)
    info = info[info["grupo"].isin(["treino", "validação"]) & info["death"].notna()].copy()
    cache = saida / f"parte_b_todo_zero_{nome_partes(partes)}.npy"
    assert cache.exists(), "rode antes: python parte_b_cnn.py preparar"
    return info[~info["exam_id"].isin(todo_zero_por_sinal(info, cache))]


def cnn_idade(seed=E.SEED):
    keras.utils.set_random_seed(seed)
    m = keras.Sequential([layers.Input((4096, 12))])
    for i, f in enumerate([32, 64, 128, 128]):                       # as mesmas convoluções do B2z/B3z
        m.add(layers.Conv1D(f, 15, padding="same", activation="relu"))
        if i < 3:
            m.add(layers.MaxPooling1D(4))
    m.add(layers.GlobalAveragePooling1D())
    m.add(layers.Dense(1))                                           # idade padronizada (regressão)
    m.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return m


def etapa_idade(a):
    tag = nome_partes(a.partes)
    info = primeiros_exames(a.partes, a.saida)
    tr, va = info[info.grupo == "treino"].copy(), info[info.grupo == "validação"].copy()
    media, desvio = tr["age"].mean(), tr["age"].std()                # só do treino
    for d in (tr, va):
        d["y"] = (d["age"] - media) / desvio
    print(f"idade: treino {len(tr)} | validação {len(va)} | média {media:.2f} desvio {desvio:.2f}", flush=True)
    g_tr, g_va = GeradorObito(tr, embaralhar=True), GeradorObito(va)
    m = cnn_idade()
    caminho = a.modelos / f"idade_{tag}_melhor.keras"
    t0 = time.time()
    hist = m.fit(g_tr, validation_data=g_va, epochs=a.max_epocas, verbose=2, callbacks=[
        keras.callbacks.EarlyStopping("val_mae", patience=2, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(caminho, monitor="val_mae", save_best_only=True),
        keras.callbacks.CSVLogger(a.saida / f"parte_b_idade_{tag}_log_por_epoca.csv"),
        LimiteDeTempo(a.max_min)])
    m = keras.models.load_model(caminho, compile=False)
    pred = m.predict(g_va, verbose=0)[:, 0] * desvio + media
    v = g_va.info
    erro_nosso = np.abs(pred - v["age"].to_numpy())
    erro_lima = np.abs(v["nn_predicted_age"].to_numpy() - v["age"].to_numpy())
    r_nosso = np.corrcoef(pred, v["age"])[0, 1]
    r_lima = np.corrcoef(v["nn_predicted_age"], v["age"])[0, 1]
    rng = np.random.default_rng(E.SEED)                              # IC da diferença de MAE, bootstrap de pacientes
    dif = [(erro_nosso[i].mean() - erro_lima[i].mean()) for i in (rng.integers(0, len(v), len(v)) for _ in range(1000))]
    res = pd.DataFrame([dict(partes=tag, n_treino=len(tr), n_val=len(v), epocas=len(hist.history["loss"]),
                             melhor_epoca=int(np.argmin(hist.history["val_mae"])) + 1, minutos=(time.time() - t0) / 60,
                             mae_nosso=erro_nosso.mean(), r_nosso=r_nosso, mae_lima=erro_lima.mean(), r_lima=r_lima,
                             dif_mae=erro_nosso.mean() - erro_lima.mean(), dif_ic_inf=np.percentile(dif, 2.5),
                             dif_ic_sup=np.percentile(dif, 97.5))])
    res.to_csv(a.saida / f"parte_b_idade_{tag}.csv", index=False, float_format="%.4f")
    pd.DataFrame({"exam_id": v["exam_id"], "age": v["age"], "idade_ecg_nossa": pred,
                  "nn_predicted_age": v["nn_predicted_age"]}).to_csv(
        a.saida / f"parte_b_idade_{tag}_previsoes_validacao.csv", index=False, float_format="%.3f")
    print(res.round(3).to_string(index=False), flush=True)


def etapa_b5(a):
    tag = nome_partes(a.partes)
    el, rel = carregar(a.partes, a.saida)
    print("dados:", rel, flush=True)
    tr = el[el.grupo == "treino"]
    idade = (tr["age"].mean(), tr["age"].std())                      # igual ao B3z
    g_tr = GeradorObito(tr, idade=idade, embaralhar=True)
    g_va = GeradorObito(el[el.grupo == "validação"], idade=idade)
    m = cnn_do_zero_com_idade()
    pre = keras.models.load_model(a.modelos / f"idade_{tag}_melhor.keras", compile=False)
    conv_pre = [c for c in pre.layers if isinstance(c, layers.Conv1D)]
    conv_b5 = [c for c in m.layers if isinstance(c, layers.Conv1D)]
    assert len(conv_pre) == len(conv_b5) == 4
    for c_b5, c_pre in zip(conv_b5, conv_pre):                       # a única mudança: pesos iniciais das convoluções
        c_b5.set_weights(c_pre.get_weights())
    caminho = a.modelos / f"B5_{tag}_melhor.keras"
    limite = LimiteDeTempo(a.max_min)
    t0 = time.time()
    hist = m.fit(g_tr, validation_data=g_va, epochs=a.max_epocas, verbose=2, callbacks=[
        keras.callbacks.EarlyStopping("val_auroc", mode="max", patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(caminho, monitor="val_auroc", mode="max", save_best_only=True),
        keras.callbacks.CSVLogger(a.saida / f"parte_b_B5_{tag}_log_por_epoca.csv"),
        limite])
    h = hist.history
    melhor = int(np.argmax(h["val_auroc"]))
    print(f"B5: {len(h['loss'])} épocas em {(time.time() - t0) / 60:.1f} min (limite: {limite.estourou}); "
          f"melhor época {melhor + 1} (val_auroc {h['val_auroc'][melhor]:.4f})", flush=True)
    m = keras.models.load_model(caminho, compile=False)
    S = m.predict(g_va, verbose=0)[:, 0]
    np.savez(a.saida / f"parte_b_B5_{tag}_previsoes_validacao.npz", S=S, exam_id=g_va.info["exam_id"].to_numpy(),
             epocas=len(h["loss"]), melhor_epoca=melhor + 1, minutos=(time.time() - t0) / 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=["idade", "b5"])
    ap.add_argument("--partes", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--saida", default=str(RAIZ / "resultados"))
    ap.add_argument("--modelos", default=str(RAIZ / "modelos"))
    ap.add_argument("--max-epocas", type=int, default=15)
    ap.add_argument("--max-min", type=float, default=180)
    a = ap.parse_args()
    a.partes = [int(p) for p in a.partes.split(",")]
    a.saida, a.modelos = Path(a.saida), Path(a.modelos)
    {"idade": etapa_idade, "b5": etapa_b5}[a.etapa](a)
