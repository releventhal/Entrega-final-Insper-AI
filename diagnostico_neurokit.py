"""Diagnóstico (13/11): piloto de `ecg_medidas.medidas_exame` (neurokit2) em 40 exames de VALIDAÇÃO (seed 42: 15 de 1dAVb, 15 normais, 10 aleatórios). Mede tempo, falhas e PR (n=40: SÓ UM SINAL, não conclusão).
Só validação; nenhum modelo; nenhum teste. Saída: resultados/diagnostico_neurokit.csv (aborta se já existe). Uso (na raiz, com o .venv): python diagnostico_neurokit.py
"""
import sys, time
from pathlib import Path
RES = Path(__file__).resolve().parent / "resultados"
assert not (RES / "diagnostico_neurokit.csv").exists(), "já existe, não sobrescrevo"
import numpy as np, pandas as pd
from a1_cnn import carregar_indices
from ecg_dados import GeradorECG, ROTULOS
from ecg_medidas import medidas_exame

_, va, _ = carregar_indices(partes=[0, 1, 2, 3])
rng = np.random.default_rng(42)
seis = va[ROTULOS].to_numpy().sum(1)
pos = np.where(va["1dAVb"].to_numpy() == 1)[0]
nor = np.where((va["normal_ecg"].to_numpy() == 1) & (seis == 0))[0]
p = rng.choice(pos, 15, replace=False)
n = rng.choice(nor, 15, replace=False)
resto = np.setdiff1d(np.arange(len(va)), np.concatenate([p, n]))
o = rng.choice(resto, 10, replace=False)
grupo = {int(i): "1dAVb" for i in p} | {int(i): "normal" for i in n} | {int(i): "aleatorio" for i in o}
idx = sorted(grupo)
sel = va.iloc[idx]
g = GeradorECG(sel, lote=len(idx), normalizar=False)
X, _ = g[0]
ids = list(g.info.exam_id)
tab = sel.set_index("exam_id")
print("composição:", pd.Series([grupo[i] for i in idx]).value_counts().to_dict())
print("dos 10 aleatórios, 1dAVb positivos:", int(sum(tab.loc[ids[k], '1dAVb'] for k in range(len(ids)) if grupo[idx[list(sel.exam_id).index(ids[k])]] == 'aleatorio')))

t0 = time.perf_counter(); medidas_exame(X[0]); print(f"primeira chamada (import+JIT): {time.perf_counter()-t0:.2f} s")
rows, tempos = [], []
for k, x in enumerate(X):
    t0 = time.perf_counter()
    m = medidas_exame(x)
    tempos.append(time.perf_counter() - t0)
    e = ids[k]
    rows.append(dict(exam_id=e, grupo=grupo[idx[list(sel.exam_id).index(e)]], d1=int(tab.loc[e, "1dAVb"]), normal=int(tab.loc[e, "normal_ecg"]), idade=tab.loc[e, "age"], **m))
d = pd.DataFrame(rows)
tempos = np.array(tempos)
print(f"\n(a) tempo por exame (40 exames, 1 processo, sem leitura do HDF5): média {tempos.mean()*1000:.0f} ms, mediana {np.median(tempos)*1000:.0f}, min {tempos.min()*1000:.0f}, máx {tempos.max()*1000:.0f}")
for nome, np_ in (("1 processo", 1), ("8 processos (ideal, /8, não medido)", 8)):
    print(f"   63.719 exames, {nome}: {tempos.mean()*63719/np_/60:.1f} min")
print("\n(b) falhas por medida (ausente = chave não existe; NaN = existe mas é NaN):")
for c in ["fc_media", "p_r_peak", "pr_dwt", "qrs_dwt", "qrs_largura_mediano", "qt_dwt", "frac_p_peak"]:
    aus = d[c].isna().sum() if c in d else len(d)
    print(f"   {c}: sem valor {aus}/{len(d)}" + ("" if c in d else " (coluna inexistente)"))
print("   exame sem nenhuma medida (dict vazio):", int(d.drop(columns=['exam_id','grupo','d1','normal','idade']).isna().all(axis=1).sum()))
print("\n(c) PR por grupo:")
for c in ["pr_dwt", "p_r_peak"]:
    for gname, sub in (("1dAVb", d[d.d1 == 1]), ("normal", d[(d.normal == 1) & (d.d1 == 0)])):
        v = sub[c].dropna()
        print(f"   {c} {gname}: n total {len(sub)}, com valor {len(v)}, mediana {v.median():.0f} ms, faixa {v.min():.0f}-{v.max():.0f}, PR>200: {(v>200).sum()}")
print("   idade mediana 1dAVb / normal:", d[d.d1==1].idade.median(), d[(d.normal==1)&(d.d1==0)].idade.median())
print("   FC mediana 1dAVb / normal:", d[d.d1==1].fc_media.median().round(1), d[(d.normal==1)&(d.d1==0)].fc_media.median().round(1))
d.to_csv(RES / "diagnostico_neurokit.csv", index=False)
