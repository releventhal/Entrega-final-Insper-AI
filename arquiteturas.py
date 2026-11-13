"""Parte A: arquiteturas maiores que o A3 (FORA DAS AULAS), comparadas só na VALIDAÇÃO. Feito para rodar no Colab (GPU).

Sugestão da intermediária (11/11): "testar modelos sequenciais" e "melhorar o A". Cada experimento muda SÓ a
arquitetura em relação ao A3: mesmos dados (partes 0 a 3, split por paciente, exames todo-zero fora), mesma
normalização por exame e derivação (A2), mesmo otimizador (Adam), lote de 64, early stopping na perda de validação
(paciência 3), limiar = F1 máximo na validação. O treino é o MESMO código do A3 (`a1_cnn.treinar`); aqui só
trocamos a função que constrói o modelo. O grupo teste é cortado na carga (a1_cnn.carregar_indices) e nunca é lido.

  A5  ResNet 1D pequena, inspirada na de Ribeiro et al. 2020 (não é cópia: tamanhos escolhidos por nós):
      Conv(64, k16) -> 4 blocos residuais (64, 128, 192, 256 filtros; kernel 16; cada bloco reduz o tempo por 4;
      atalho = MaxPool + Conv 1x1; BatchNorm, ReLU, Dropout 0,2) -> GlobalAveragePooling -> 6 sigmoid.
  A6  CNN + LSTM: as MESMAS 4 convoluções do A3 (4096 -> 64 passos de tempo x 128 canais); no lugar do
      GlobalAveragePooling, uma LSTM bidirecional (64 unidades por sentido) lê os 64 trechos em ordem -> 6 sigmoid.
      "Modelo sequencial": guarda memória do que já viu, o que pode ajudar no ritmo (AF, SB, ST) e no intervalo PR.

Uso (raiz do repositório):
  python arquiteturas.py bench A5 --partes 0,1,2,3
  python arquiteturas.py treinar A6 --partes 0,1,2,3 --max-min 240 --max-epocas 20
"""
import argparse

import keras
from keras import layers

import a1_cnn
from ecg_dados import ROTULOS, SEED

DESCRICAO = {"A5": "A5 ResNet 1D pequena (partes 0 a 3; normalização A2; fora das aulas)",
             "A6": "A6 CNN do A3 + LSTM bidirecional (partes 0 a 3; normalização A2; fora das aulas)"}


def bloco_residual(x, filtros, kernel=16, reduz=4, dropout=0.2):
    atalho = layers.MaxPooling1D(reduz, padding="same")(x)
    if atalho.shape[-1] != filtros:
        atalho = layers.Conv1D(filtros, 1, padding="same", use_bias=False)(atalho)
    y = layers.Conv1D(filtros, kernel, padding="same", use_bias=False)(x)
    y = layers.Activation("relu")(layers.BatchNormalization()(y))
    y = layers.Dropout(dropout)(y)
    y = layers.Conv1D(filtros, kernel, strides=reduz, padding="same", use_bias=False)(y)
    y = layers.Add()([y, atalho])
    y = layers.Activation("relu")(layers.BatchNormalization()(y))
    return layers.Dropout(dropout)(y)


def resnet_pequena(seed=SEED, pesos=None):
    keras.utils.set_random_seed(seed)
    ent = keras.Input((4096, 12))
    x = layers.Conv1D(64, 16, padding="same", use_bias=False)(ent)
    x = layers.Activation("relu")(layers.BatchNormalization()(x))
    for f in (64, 128, 192, 256):                     # 4096 -> 1024 -> 256 -> 64 -> 16 passos de tempo
        x = bloco_residual(x, f)
    x = layers.GlobalAveragePooling1D()(x)
    m = keras.Model(ent, layers.Dense(len(ROTULOS), activation="sigmoid")(x), name="A5_resnet")
    m.compile(optimizer="adam", loss="binary_crossentropy")
    return m


def cnn_lstm(seed=SEED, pesos=None):
    keras.utils.set_random_seed(seed)
    ent = keras.Input((4096, 12))
    x = ent
    for i, f in enumerate([32, 64, 128, 128]):          # exatamente as convoluções do A3
        x = layers.Conv1D(f, 15, padding="same", activation="relu")(x)
        if i < 3:
            x = layers.MaxPooling1D(4)(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)        # no lugar do GlobalAveragePooling
    m = keras.Model(ent, layers.Dense(len(ROTULOS), activation="sigmoid")(x), name="A6_cnn_lstm")
    m.compile(optimizer="adam", loss="binary_crossentropy")
    return m


CONSTRUTORES = {"A5": resnet_pequena, "A6": cnn_lstm}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=["bench", "treinar", "resumo"])
    ap.add_argument("nome", choices=list(CONSTRUTORES))
    ap.add_argument("--partes", default="0,1,2,3")
    ap.add_argument("--max-min", type=float, default=240)
    ap.add_argument("--max-epocas", type=int, default=20)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    partes = [int(p) for p in a.partes.split(",")]
    # a única troca: o construtor do modelo usado pelo treino do A3
    a1_cnn.construir_modelo = CONSTRUTORES[a.nome]
    a1_cnn.ROTULO_EXPERIMENTO[a.nome] = DESCRICAO[a.nome]
    if a.modo == "resumo":
        CONSTRUTORES[a.nome]().summary()
    elif a.modo == "bench":
        print("parâmetros:", CONSTRUTORES[a.nome]().count_params())
        a1_cnn.bench(normalizar=True, partes=partes)
    else:
        df, _, _ = a1_cnn.treinar(a.nome, normalizar=True, max_min=a.max_min, max_epocas=a.max_epocas, partes=partes, seed=a.seed)
        a1_cnn.gravar_no_csv(df)
