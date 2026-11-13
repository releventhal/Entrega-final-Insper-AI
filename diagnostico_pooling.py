"""Diagnóstico (13/11): palpite do pooling. A3 congelado (só inferência) na VALIDAÇÃO com o recorte de 25% de robustez_02_estresse.py (o resto vira zero, mesmo rng por exame).
Compara ANTES/DEPOIS: nota média de positivos e negativos, fração de positivos acima do limiar do JSON, com o pooling normal (GAP) e com a média só sobre os passos com sinal
(recalculada fora do modelo com a camada Dense do A3). Nada é treinado nem alterado. Saídas: resultados/diagnostico_pooling.csv e diagnostico_pooling_notas.npz (abortam se já existem).
Uso (na raiz, com o .venv): python diagnostico_pooling.py
"""
import sys, time
from pathlib import Path
RES = Path(__file__).resolve().parent / "resultados"
assert not (RES / "diagnostico_pooling.csv").exists(), "já existe, não sobrescrevo"
import numpy as np, pandas as pd, keras, tensorflow as tf
from a1_cnn import carregar_indices
from ecg_dados import GeradorECG, ROTULOS, normalizar_exame
from robustez_comum import RES as RES_ROBUSTEZ, MODELOS, carregar_val, conferir_modelo, limiares_json
import robustez_02_estresse as r2

print("SHA256:", conferir_modelo()[:8])
S0, Y, *_ = carregar_val()
lim = np.array([limiares_json()[r] for r in ROTULOS])
nomes = list(r2.PERTURBACOES)
j = nomes.index("recorte de 25% do sinal")
salvo = np.load(RES_ROBUSTEZ / "robustez_02_previsoes.npz")
assert str(salvo["nomes"][j]) == "recorte de 25% do sinal"
S25_salvo = salvo[f"S{j}"]

_, va, _ = carregar_indices(partes=[0, 1, 2, 3])
g = GeradorECG(va, normalizar=False)
m = keras.models.load_model(MODELOS / "A3_melhor.keras", compile=False)
print([type(l).__name__ for l in m.layers])
gap = [i for i, l in enumerate(m.layers) if isinstance(l, keras.layers.GlobalAveragePooling1D)][0]
dense = m.layers[-1]
W, b = dense.kernel.numpy(), dense.bias.numpy()
f = r2.PERTURBACOES["recorte de 25% do sinal"][1]

def passos(raw, T=64):
    ix = np.where(np.abs(raw).max(axis=1) > 0)[0]
    mk = np.zeros(T, bool)
    if len(ix) == 0: mk[:] = True
    else: mk[ix[0] * T // 4096: ix[-1] * T // 4096 + 1] = True
    return mk

def feats(Xn):
    h = tf.convert_to_tensor(Xn)
    for l in m.layers[:gap]: h = l(h)
    return h.numpy()  # (n, 64, 128)

def sig(z): return 1 / (1 + np.exp(-z))
res = {k: [] for k in ["orig_gap", "orig_mask", "rec_gap", "rec_mask"]}
msk_frac = []
t0 = time.perf_counter()
for i in range(len(g)):
    X, _ = g[i]; base = i * g.lote
    Xo, Xr, mo, mr = [], [], [], []
    for k, x in enumerate(X):
        ix, ok, sd = r2.ctx(x)
        rng = np.random.default_rng([42, base + k, j])
        xr = f(x, ix, ok, sd, rng) if len(ix) else x
        Xo.append(normalizar_exame(x) if len(ix) else x); Xr.append(normalizar_exame(xr) if len(ix) else x)
        mo.append(passos(x)); mr.append(passos(xr))
    Fo, Fr = feats(np.stack(Xo)), feats(np.stack(Xr))
    mo, mr = np.stack(mo), np.stack(mr); msk_frac.append(mr.mean(1))
    for nome, F, mk in (("orig", Fo, mo), ("rec", Fr, mr)):
        res[nome + "_gap"].append(sig(F.mean(1) @ W + b))
        pm = (F * mk[:, :, None]).sum(1) / mk.sum(1, keepdims=True)
        res[nome + "_mask"].append(sig(pm @ W + b))
    if i % 20 == 0: print(f"lote {i+1}/{len(g)} {time.perf_counter()-t0:.0f}s", flush=True)
R = {k: np.concatenate(v) for k, v in res.items()}
print("fração de passos com sinal no recorte (média):", np.concatenate(msk_frac).mean().round(3))
print("conferência: orig_gap vs salvo", np.abs(R["orig_gap"] - S0).max(), "| rec_gap vs salvo do estresse", np.abs(R["rec_gap"] - S25_salvo).max())
assert not (RES / "diagnostico_pooling_notas.npz").exists()
np.savez(RES / "diagnostico_pooling_notas.npz", **R)
linhas = []
for k, r in enumerate(ROTULOS):
    y = Y[:, k] == 1
    row = dict(alteracao=r, n_pos=int(y.sum()), limiar=lim[k])
    for nome in R:
        s = R[nome][:, k]
        row[f"{nome}_pos"] = s[y].mean(); row[f"{nome}_neg"] = s[~y].mean(); row[f"{nome}_pos>=lim"] = (s[y] >= lim[k]).mean()
    row["pos_que_caem_gap"] = (R["rec_gap"][y, k] < R["orig_gap"][y, k]).mean()
    row["neg>=lim_orig"] = (R["orig_gap"][~y, k] >= lim[k]).mean(); row["neg>=lim_rec"] = (R["rec_gap"][~y, k] >= lim[k]).mean(); row["neg>=lim_rec_mask"] = (R["rec_mask"][~y, k] >= lim[k]).mean()
    linhas.append(row)
T = pd.DataFrame(linhas)
T.to_csv(RES / "diagnostico_pooling.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(T.round(3).T.to_string())
