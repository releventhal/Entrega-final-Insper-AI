"""Bloco 2 (08/11): análise de erros do A3 (seed 42) na VALIDAÇÃO.

DESCRITIVO, POSTERIOR AO TREINO, NÃO MUDA O MODELO. Provisório, a confirmar com a Renata.
- Modelo: modelos/A3_melhor.keras; limiares: congelado_candidato_A3.json (só leitura; o JSON não é alterado).
- Conjunto: a mesma validação do A3 (partes 0 a 3, sem os 22 exames todo-zero). O teste NUNCA é lido:
  o índice é filtrado para treino/validação em a1_cnn.carregar_indices, e o hash dos sinais lê só linhas de treino/validação.
Uso: .venv\\Scripts\\python analise_erros_A3.py
"""
import hashlib
import json

import h5py
import keras
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from a0_baselines import PASTA_RESULTADOS, metricas
from a1_cnn import PASTA_MODELOS, carregar_indices
from ecg_dados import DADOS, ROTULOS, GeradorECG, normalizar_exame

FAIXAS = [(0, 40, "<40"), (40, 60, "40-59"), (60, 70, "60-69"), (70, 200, "70+")]
ESCALAS = [0.45, 2.0]  # teste de estresse: multiplica o sinal BRUTO antes da normalização
SAIDA = PASTA_RESULTADOS / "erros_A3"


def prever_com_escalas(m, g):
    """Uma passada pela validação: previsões no original e com o sinal multiplicado por cada escala,
    e o MD5 do sinal bruto (float32, como gravado no HDF5) de cada exame."""
    S = {1.0: [], **{e: [] for e in ESCALAS}}
    hashes = []
    for i in range(len(g)):
        X, _ = g[i]  # g com normalizar=False: sinal bruto
        hashes += [hashlib.md5(x.tobytes()).hexdigest() for x in X]
        for e in S:
            S[e].append(m.predict_on_batch(np.stack([normalizar_exame(x * np.float32(e)) for x in X])))
    return {e: np.concatenate(v) for e, v in S.items()}, np.array(hashes)


def hashes_treino(tr):
    """MD5 dos sinais de treino, lendo só as linhas de treino (em trechos de linhas consecutivas)."""
    out = []
    for p, d in tr.sort_values(["parte", "linha"]).groupby("parte"):
        linhas = d.linha.to_numpy()
        cortes = np.where(np.diff(linhas) != 1)[0] + 1
        with h5py.File(DADOS / f"exams_part{p}.hdf5", "r") as f:
            ds = f["tracings"]
            for trecho in np.split(linhas, cortes):
                for ini in range(trecho[0], trecho[-1] + 1, 2000):
                    fim = min(ini + 2000, trecho[-1] + 1)
                    out += [hashlib.md5(x.tobytes()).hexdigest() for x in ds[ini:fim]]
    return np.array(out)


def metricas_conjunto(Y, S, lim, nome):
    linhas = []
    for k, r in enumerate(ROTULOS):
        y = Y[:, k].astype(int)
        linhas.append(dict(conjunto=nome, rotulo=r, n=len(y), positivos=int(y.sum()), **metricas(y, S[:, k], lim[r])))
    df = pd.DataFrame(linhas)
    macro = df[["auroc", "auprc", "sensibilidade", "precisao", "f1"]].mean().to_dict() | dict(conjunto=nome, rotulo="MACRO", n=len(Y))
    return pd.concat([df, pd.DataFrame([macro])], ignore_index=True)


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(DADOS.parent / "congelado_candidato_A3.json", encoding="utf-8"))
    lim = cfg["limiares"]["valores"]
    tr, va, rel = carregar_indices(partes=cfg["dados_de_treino"]["partes"])
    assert len(va) == cfg["dados_de_treino"]["n_validacao"], (len(va), rel)
    g = GeradorECG(va, normalizar=False)  # bruto; a normalização do A2 é aplicada em prever_com_escalas
    info, Y = g.info, g.Y
    m = keras.models.load_model(PASTA_MODELOS / "A3_melhor.keras")
    S_esc, h_va = prever_com_escalas(m, g)
    S = S_esc[1.0]
    salvo = np.load(PASTA_RESULTADOS / "A3_previsoes_validacao.npy")
    print(f"previsões recalculadas vs salvas: diferença máxima {np.abs(S - salvo).max():.2e}", flush=True)
    P = np.column_stack([S[:, k] >= lim[r] for k, r in enumerate(ROTULOS)])
    np.save(SAIDA / "previsoes_e_rotulos.npy", np.column_stack([S, Y, info.age, info.is_male, info.patient_id]).astype(np.float64))

    # 1) contagens por alteração
    cont = pd.DataFrame([dict(rotulo=r, limiar=lim[r], VP=int((P[:, k] & (Y[:, k] == 1)).sum()), FP=int((P[:, k] & (Y[:, k] == 0)).sum()),
                              FN=int((~P[:, k] & (Y[:, k] == 1)).sum()), VN=int((~P[:, k] & (Y[:, k] == 0)).sum()))
                         for k, r in enumerate(ROTULOS)])
    print("\n== 1) VP/FP/FN/VN por alteração (limiares do JSON) ==\n", cont.to_string(index=False))

    # 2) por faixa de idade e por sexo
    sub = []
    grupos = [(f"idade {n}", ((info.age >= lo) & (info.age < hi)).to_numpy()) for lo, hi, n in FAIXAS]
    grupos += [("mulheres", ~info.is_male.astype(bool).to_numpy()), ("homens", info.is_male.astype(bool).to_numpy())]
    for nome, k_ in grupos:
        for k, r in enumerate(ROTULOS):
            y, p = Y[k_, k] == 1, P[k_, k]
            vp = (p & y).sum()
            sub.append(dict(subgrupo=nome, rotulo=r, n=int(k_.sum()), positivos=int(y.sum()),
                            sensibilidade=vp / y.sum() if y.sum() else np.nan, precisao=vp / p.sum() if p.sum() else np.nan))
    sub = pd.DataFrame(sub)
    print("\n== 2) sensibilidade e precisão por subgrupo ==\n", sub.round(3).to_string(index=False))

    # 3) concentração de erros por paciente
    conc = []
    for k, r in enumerate(ROTULOS):
        for tipo, mask in (("FP", P[:, k] & (Y[:, k] == 0)), ("FN", ~P[:, k] & (Y[:, k] == 1))):
            por_pac = info.patient_id[mask].value_counts().to_numpy()
            n = mask.sum()
            metade = int(np.searchsorted(np.cumsum(por_pac), n / 2) + 1) if n else 0
            conc.append(dict(rotulo=r, tipo=tipo, erros=int(n), pacientes_com_erro=len(por_pac), max_erros_1_paciente=int(por_pac.max()) if n else 0,
                             pacientes_para_metade=metade, se_cada_erro_fosse_1_paciente=int(np.ceil(n / 2))))
    conc = pd.DataFrame(conc)
    print(f"\n== 3) concentração por paciente (validação: {len(info)} exames de {info.patient_id.nunique()} pacientes) ==\n", conc.to_string(index=False))

    # 4) confusões entre alterações e exames com mais de um rótulo
    conf = pd.DataFrame(index=ROTULOS, columns=ROTULOS, dtype=float)
    for i, ri in enumerate(ROTULOS):
        for j, rj in enumerate(ROTULOS):
            k = (Y[:, i] == 1) & (Y[:, j] == 0)
            conf.loc[ri, rj] = P[k, j].mean() if i != j else np.nan
    sem_nada = Y.sum(1) == 0
    base_fp = pd.Series({r: P[sem_nada, k].mean() for k, r in enumerate(ROTULOS)})
    print("\n== 4a) linha = alteração verdadeira (e a da coluna ausente); valor = fração prevista positiva na coluna ==\n", conf.round(3).to_string())
    print("taxa de 'positivo' da coluna em exames SEM nenhuma das 6:", base_fp.round(4).to_dict())
    nrot = Y.sum(1)
    erro_exame = (P != (Y == 1)).any(1)
    multi = pd.DataFrame([dict(rotulos_no_exame=f"{a}" if a < 2 else "2+", exames=int(k.sum()), exames_com_algum_erro=int(erro_exame[k].sum()),
                               taxa_de_erro=erro_exame[k].mean())
                          for a, k in ((0, nrot == 0), (1, nrot == 1), (2, nrot >= 2))])
    print("\n== 4b) erro por número de rótulos verdadeiros no exame ==\n", multi.round(4).to_string(index=False))

    # 5) 1dAVb: notas e erros por idade
    k = ROTULOS.index("1dAVb")
    q = [0.05, 0.25, 0.5, 0.75, 0.95]
    notas = pd.DataFrame({"positivos": np.quantile(S[Y[:, k] == 1, k], q), "negativos": np.quantile(S[Y[:, k] == 0, k], q)}, index=q)
    print(f"\n== 5) 1dAVb: quantis das notas (limiar {lim['1dAVb']:.3f}) ==\n", notas.round(4).to_string())
    avb = sub[sub.rotulo == "1dAVb"]
    err_idade = []
    for lo, hi, n in FAIXAS:
        kk = ((info.age >= lo) & (info.age < hi)).to_numpy()
        err_idade.append(dict(faixa=n, positivos=int(Y[kk, k].sum()), FN=int((~P[kk, k] & (Y[kk, k] == 1)).sum()),
                              FP=int((P[kk, k] & (Y[kk, k] == 0)).sum()), negativos=int((Y[kk, k] == 0).sum())))
    print(pd.DataFrame(err_idade).to_string(index=False))

    # 6) calibração (faixas fixas de nota) e F1 contra o limiar
    bordas = np.array([0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001])
    cal, f1c = [], []
    for k, r in enumerate(ROTULOS):
        b = np.digitize(S[:, k], bordas) - 1
        for i in np.unique(b):
            kk = b == i
            cal.append(dict(rotulo=r, faixa=f"{bordas[i]:.2f}-{min(bordas[i + 1], 1):.2f}", n=int(kk.sum()),
                            nota_media=S[kk, k].mean(), fracao_positiva=Y[kk, k].mean()))
        for t in np.round(np.arange(0.02, 0.99, 0.02), 2):
            mt = metricas(Y[:, k].astype(int), S[:, k], t)
            f1c.append(dict(rotulo=r, limiar=t, f1=mt["f1"], sensibilidade=mt["sensibilidade"], precisao=mt["precisao"]))
    cal, f1c = pd.DataFrame(cal), pd.DataFrame(f1c)
    print("\n== 6a) calibração ==\n", cal.round(4).to_string(index=False))
    print("\n== 6b) F1 contra o limiar: F1 máximo da grade e F1 no limiar do JSON ==")
    for k, r in enumerate(ROTULOS):
        d = f1c[f1c.rotulo == r]
        print(f"  {r}: F1 máx da grade {d.f1.max():.3f} em {d.limiar[d.f1.idxmax()]:.2f}; limiar JSON {lim[r]:.3f}; "
              f"F1 >= máx-0,05 entre {d.limiar[d.f1 >= d.f1.max() - 0.05].min():.2f} e {d.limiar[d.f1 >= d.f1.max() - 0.05].max():.2f}")

    # 7) com e sem exames de derivação parcialmente zerada e de sinal idêntico a outro
    z = np.load(DADOS / "exames_zerados_p0-3.npy")
    parcial = info.exam_id.isin(z[(z[:, 3] > 0) & (z[:, 3] < 12), 0]).to_numpy()
    h_tr = hashes_treino(tr)
    todos = pd.Series(np.concatenate([h_va, h_tr]))
    contagem = todos.value_counts()
    identico = (contagem.loc[h_va].to_numpy() > 1)
    identico_ao_treino = np.isin(h_va, h_tr)
    print(f"\n== 7) validação: {parcial.sum()} exames com derivação parcialmente zerada; {identico.sum()} com sinal idêntico a outro "
          f"exame de treino/validação ({identico_ao_treino.sum()} idênticos a algum de treino); com alguma das 6: "
          f"parcial {int((Y[parcial].sum(1) > 0).sum())}, idêntico {int((Y[identico].sum(1) > 0).sum())}")
    sens7 = pd.concat([metricas_conjunto(Y, S, lim, "todos"),
                       metricas_conjunto(Y[~parcial], S[~parcial], lim, "sem parciais"),
                       metricas_conjunto(Y[~identico], S[~identico], lim, "sem idênticos"),
                       metricas_conjunto(Y[~parcial & ~identico], S[~parcial & ~identico], lim, "sem parciais e sem idênticos")])
    print(sens7[sens7.rotulo == "MACRO"][["conjunto", "n", "auroc", "auprc", "f1"]].round(4).to_string(index=False))

    # 8) estresse de escala
    est = []
    for e in ESCALAS:
        d = np.abs(S_esc[e] - S)
        Pe = np.column_stack([S_esc[e][:, k] >= lim[r] for k, r in enumerate(ROTULOS)])
        est.append(dict(escala=e, dif_max=d.max(), dif_media=d.mean(), **{f"dif_max_{r}": d[:, k].max() for k, r in enumerate(ROTULOS)},
                        decisoes_trocadas=int((Pe != P).sum())))
    est = pd.DataFrame(est)
    print("\n== 8) estresse de escala (sinal bruto x escala, depois a normalização do A2) ==\n", est.to_string(index=False))

    for nome, df in (("contagens", cont), ("subgrupos", sub), ("concentracao", conc), ("multirrotulo", multi), ("calibracao", cal),
                     ("f1_vs_limiar", f1c), ("parciais_identicos", sens7), ("estresse_escala", est), ("avb_erros_idade", pd.DataFrame(err_idade))):
        df.to_csv(SAIDA / f"{nome}.csv", index=False, float_format="%.6g")
    conf.to_csv(SAIDA / "confusoes.csv", float_format="%.4f")
    notas.to_csv(SAIDA / "avb_quantis_notas.csv", float_format="%.4f")


if __name__ == "__main__":
    main()
