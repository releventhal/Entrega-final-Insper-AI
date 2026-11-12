"""Robustez, tarefa 5: Grad-CAM do A3 (seed 42) em 3 exames POSITIVOS por alteração da VALIDAÇÃO. TÉCNICA FORA DAS AULAS (interpretabilidade).
Só inferência: o modelo e os limiares não mudam. Os 3 exames de cada alteração são sorteados com seed 42 entre TODOS os positivos da validação
(não se escolhe por acerto nem por erro; a nota e se passou do limiar aparecem no título só como informação).

Grad-CAM 1D: A = saída da última Conv1D (64 passos x 128 filtros); alpha_k = média no tempo de d(logit da alteração)/dA_k (a 1ª versão usava a nota sigmoide; o resultado foi IDÊNTICO: os mapas zerados não eram saturação da sigmoide, e sim exames sem NENHUM trecho com
evidência positiva para a alteração, que o ReLU do Grad-CAM zera; são exames de nota ~0, ou seja, falsos negativos entre os positivos sorteados); mapa = ReLU(soma_k alpha_k * A_k),
levado a 4.096 amostras por interpolação linear, restrito à região com sinal e dividido pelo máximo do exame (escala 0 a 1, só dentro do exame).
O mapa é UM por exame e alteração (as convoluções misturam as 12 derivações): ele diz EM QUE MOMENTO o modelo olhou, não em qual derivação.
O sinal desenhado é o que o modelo recebe (depois da normalização do A3). Sem teste de sanidade do Grad-CAM (pesos aleatórios): não verificado.
Saídas novas: figuras/robustez_gradcam_<alteração>.png e resultados/robustez_05_gradcam_exames.csv.
"""
import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from a1_cnn import carregar_indices
from ecg_dados import ROTULOS, GeradorECG
from robustez_comum import FIG, MODELOS, RES, conferir_modelo, limiares_json, salvar_novo

DERIV = ["DI", "DII", "DIII", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def gradcam(m, X, k):
    """(n, 4096) mapas para a alteração k, e as notas (n,)"""
    ultima = [l for l in m.layers if isinstance(l, keras.layers.Conv1D)][-1]
    x = tf.convert_to_tensor(X)
    with tf.GradientTape() as tape:
        h = x
        for l in m.layers[:-1]:
            h = l(h)
            if l is ultima:
                A = h
                tape.watch(A)  # gradiente em relação à ativação da última convolução
        z = tf.matmul(h, m.layers[-1].kernel) + m.layers[-1].bias  # logit (antes da sigmoide): a sigmoide saturada (nota ~0 ou ~1) zeraria o gradiente
        nota = tf.sigmoid(z)[:, k]
        alvo = z[:, k]
    g = tape.gradient(alvo, A)  # (n, 64, 128)
    alfa = tf.reduce_mean(g, axis=1, keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(alfa * A, axis=-1)).numpy()  # (n, 64)
    T = X.shape[1]
    passo = (np.arange(cam.shape[1]) + 0.5) * T / cam.shape[1]  # centro de cada passo, em amostras
    mapas = np.stack([np.interp(np.arange(T), passo, c) for c in cam])
    return mapas, nota.numpy()


def main():
    sha = conferir_modelo()
    print("SHA256 do A3 conferido:", sha[:8])
    lim = limiares_json()
    _, va, _ = carregar_indices(partes=[0, 1, 2, 3])  # só treino e validação
    rng = np.random.default_rng(42)
    escolha = {r: rng.choice(np.where(va[r].to_numpy() == 1)[0], 3, replace=False) for r in ROTULOS}  # posições em `va` (ordem de carregar_indices)
    todas = sorted({p for v in escolha.values() for p in v})
    g = GeradorECG(va.iloc[todas], lote=len(todas), normalizar=True)
    X, _ = g[0]
    pos_de = {e: i for i, e in enumerate(g.info.exam_id)}  # g.info reordena por (parte, linha)
    m = keras.models.load_model(MODELOS / "A3_melhor.keras", compile=False)
    linhas = []
    for k, r in enumerate(ROTULOS):
        ids = va.exam_id.to_numpy()[escolha[r]]
        ii = [pos_de[e] for e in ids]
        mapas, nota = gradcam(m, X[ii], k)
        assert np.abs(nota - m.predict(X[ii], verbose=0)[:, k]).max() < 1e-4, "nota do passo manual diferente da do modelo"
        fig, ax = plt.subplots(3, 1, figsize=(14, 11))
        for a, e, x, cam, s in zip(ax, ids, X[ii], mapas, nota):
            ativo = np.where(np.abs(x).max(axis=1) > 0)[0]
            lo, hi = ativo[0], ativo[-1] + 1
            cam = cam.copy()
            cam[:lo], cam[hi:] = 0, 0
            cam = cam / cam.max() if cam.max() > 0 else cam
            t = np.arange(lo, hi) / 400
            a.imshow(cam[None, lo:hi], aspect="auto", cmap="Reds", alpha=.6, extent=[t[0], t[-1], -1, 12 * 5 + 1], vmin=0, vmax=1)
            for d in range(12):
                a.plot(t, np.clip(x[lo:hi, d], -4, 4) * 0.8 + (11 - d) * 5, c="k", lw=.5)
            a.set_yticks([(11 - d) * 5 for d in range(12)], DERIV, fontsize=7)
            a.set_xlim(t[0], t[-1])
            a.set_ylim(-4, 12 * 5)
            info = va[va.exam_id == e].iloc[0]
            a.set_title(f"exame {e} | {r} verdadeiro | nota {s:.3f} (limiar {lim[r]:.3f}: {'acima' if s >= lim[r] else 'abaixo'}) | idade {info.age:.0f}, "
                        f"{'homem' if info.is_male else 'mulher'} | outras alterações verdadeiras: {[q for q in ROTULOS if q != r and info[q] == 1] or 'nenhuma'}", fontsize=8)
            linhas.append(dict(rotulo=r, exam_id=int(e), nota=float(s), limiar=lim[r], acima_do_limiar=bool(s >= lim[r]),
                               instante_do_pico_s=float(np.argmax(cam) / 400), fracao_da_massa_na_metade_central=float(cam[lo + (hi - lo) // 4: lo + 3 * (hi - lo) // 4].sum() / max(cam.sum(), 1e-9))))
        ax[-1].set_xlabel("tempo (s); vermelho = Grad-CAM (quanto mais escuro, mais o modelo usou aquele trecho para a nota desta alteração)")
        fig.suptitle(f"Grad-CAM do A3 — {r}: 3 positivos sorteados na validação (seed 42, sem escolher por acerto)", fontsize=11)
        fig.tight_layout()
        salvar_novo(fig, FIG / f"robustez_gradcam_logit_{r}.png")  # a 1ª versão (gradiente da nota sigmoide) ficou em robustez_gradcam_<alteração>.png
        plt.close(fig)
    t = pd.DataFrame(linhas)
    salvar_novo(t, RES / "robustez_05_gradcam_logit_exames.csv")
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
