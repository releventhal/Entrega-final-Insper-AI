"""Parte B: avaliação do modelo final (B3z) e das referências (B0, B1, B4, B4p) num conjunto, SEM retreinar o B3z.

TRAVA (como no avaliar_externo.py da Parte A): no conjunto "teste" o script RECUSA rodar, antes de abrir qualquer dado,
a menos que (1) congelado_candidato_parte_b.json tenha "status": "congelado", (2) seja passada a flag
--confirmo-congelado e (3) o SHA256 do modelo e do A3 batam com o JSON. Congelar = Renata E Tom confirmarem (log de
14/11) e uma pessoa mudar o status à mão. O conjunto "validacao" roda livre: serve para conferir que este script
reproduz as notas já salvas da validação (B0, B1, B4, B4p, B3z), sem abrir o teste.

O que é feito (igual à validação, decidido antes; ver o JSON):
  - pacientes: primeiro exame de cada paciente do conjunto (grupo do split por paciente de sempre), sem exames com as
    12 derivações zeradas (pelo sinal), regra do horizonte de 2 anos;
  - B0, B1, B4, B4p: regressões logísticas ajustadas SÓ nos pacientes de treino elegíveis, com as mesmas funções e
    colunas da validação (parte_b_tabular.ajustar/features e parte_b_alteracoes.matriz); B4p usa as notas do A3
    congelado (características do A3 -> camada final do A3); no treino, as características já salvas;
  - B3z: o modelo final aplicado aos pacientes do conjunto; idade padronizada com a média e o desvio do treino elegível;
  - IC 95% por bootstrap pareado de pacientes (1.000, seed 42); risco escondido e triagem (parte_b_analises.com_ic).
A saída bruta (notas) é gravada logo depois das previsões; no "teste", se ela já existir, as previsões NÃO são refeitas.

Uso:
  python parte_b_teste.py --conjunto validacao --partes 0,...,7     # conferência (com o B3z p0-7), roda livre
  python parte_b_teste.py --conjunto teste --confirmo-congelado     # SÓ depois de congelar; partes 0 a 17 (Colab)
Saídas: resultados/parte_b_<conjunto>_bruto_<tag>.csv, parte_b_<conjunto>_<tag>.csv, parte_b_<conjunto>_analises_<tag>.csv
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).parent
JSON = RAIZ / "congelado_candidato_parte_b.json"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def trava(cfg, conjunto, confirmo, partes):
    """Roda ANTES de importar o TensorFlow ou ler qualquer dado."""
    if conjunto != "teste":
        return
    if cfg.get("status") != "congelado":
        sys.exit(f"RECUSADO: o JSON tem status '{cfg.get('status')}' (precisa ser 'congelado'). Nada foi lido.")
    if not confirmo:
        sys.exit("RECUSADO: passe --confirmo-congelado para confirmar que o modelo está congelado. Nada foi lido.")
    if partes != cfg["dados"]["partes"]:
        sys.exit(f"RECUSADO: partes {partes} diferentes das do JSON {cfg['dados']['partes']}. Nada foi lido.")
    for chave in ("modelo_final", "a3_para_b4p"):
        if sha256(RAIZ / cfg[chave]["caminho"]) != cfg[chave]["sha256"]:
            sys.exit(f"RECUSADO: SHA256 de {cfg[chave]['caminho']} diferente do JSON. Nada foi lido.")


def main(a):
    cfg = json.loads(JSON.read_text(encoding="utf-8"))
    trava(cfg, a.conjunto, a.confirmo_congelado, a.partes)       # <- antes de qualquer dado

    import keras
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    import ecg_dados as E
    from parte_b_alteracoes import matriz
    from parte_b_analises import com_ic
    from parte_b_cnn import (HORIZONTE, MODELO_A3, SHA_A3, GeradorObito, bootstrap, carregar, extrator_a3,
                             nome_partes, todo_zero_por_sinal)
    from parte_b_tabular import ajustar, features, rotular

    R = E.ROTULOS
    tag = nome_partes(a.partes)
    saida = RAIZ / "resultados"
    grupo = {"validacao": "validação", "teste": "teste"}[a.conjunto]
    modelo_b3z = RAIZ / "modelos" / f"B3z_{tag}_melhor.keras"
    assert sha256(MODELO_A3) == SHA_A3, "o A3 não é o congelado"
    print(f"conjunto {a.conjunto} | partes {tag} | B3z {modelo_b3z.name} sha256 {sha256(modelo_b3z)[:12]}...", flush=True)

    # ---------------------------------------------------------------- treino elegível (o mesmo da validação)
    el, rel = carregar(a.partes, saida)
    tr = el[el.grupo == "treino"].copy()
    for c in R + ["normal_ecg"]:
        tr[c] = tr[c].astype(bool)
    print("treino:", rel["treino"], flush=True)

    # ---------------------------------------------------------------- conjunto avaliado
    if a.conjunto == "validacao":
        av = el[el.grupo == "validação"].copy()
        rel_av = rel["validação"]
    else:
        info = E.construir_indice(a.partes)
        info = info[info["grupo"] == grupo]
        info = info[info["death"].notna()].copy()
        assert info["patient_id"].is_unique
        info["death"] = info["death"].astype(bool)
        info["delta"] = info["nn_predicted_age"] - info["age"]
        zer = todo_zero_por_sinal(info, saida / f"parte_b_todo_zero_teste_{tag}.npy")
        sai = info["exam_id"].isin(zer)
        av, cens = rotular(info[~sai], HORIZONTE)
        rel_av = dict(primeiros_exames=len(info), todo_zero=int(sai.sum()), censurados=int(cens.sum()),
                      elegiveis=len(av), obitos=int(av["y"].sum()))
        assert not set(av["patient_id"]) & set(el["patient_id"]), "paciente do teste no treino/validação"
    for c in R + ["normal_ecg"]:
        av[c] = av[c].astype(bool)
    av = av.sort_values(["parte", "linha"], ignore_index=True)
    print(f"{a.conjunto}:", rel_av, flush=True)

    bruto = saida / f"parte_b_{a.conjunto}_bruto_{tag}.csv"
    if a.conjunto == "teste" and bruto.exists():
        print(f"{bruto.name} já existe: as previsões NÃO são refeitas", flush=True)
        d = pd.read_csv(bruto)
        assert list(d["exam_id"]) == list(av["exam_id"]), "pacientes diferentes da saída bruta salva"
    else:
        # B0 e B1
        esc, mods = ajustar(tr)
        notas = {m: mods[m].predict_proba(features(av, esc, m))[:, 1] for m in ("B0", "B1")}
        # notas do A3: treino pelas características salvas; conjunto avaliado calculado agora
        W, b = keras.models.load_model(MODELO_A3, compile=False).layers[-1].get_weights()
        c = np.load(saida / f"parte_b_caracteristicas_A3_{tag}.npz")
        F_tr = pd.DataFrame(c["F"], index=c["exam_id"]).loc[tr["exam_id"]].to_numpy()
        g = GeradorObito(av)
        F_av = pd.DataFrame(extrator_a3().predict(g, verbose=0), index=g.info["exam_id"]).loc[av["exam_id"]].to_numpy()
        for d_, F in ((tr, F_tr), (av, F_av)):
            p = np.clip(1 / (1 + np.exp(-(F @ W + b))), 1e-6, 1 - 1e-6)
            for j, r in enumerate(R):
                d_[f"logit_{r}"] = np.log(p[:, j] / (1 - p[:, j]))
        # B4 e B4p
        for m in ("B4", "B4p"):
            e = StandardScaler().fit(matriz(tr, m))
            lr = LogisticRegression(max_iter=2000, random_state=E.SEED).fit(e.transform(matriz(tr, m)), tr["y"])
            notas[m] = lr.predict_proba(e.transform(matriz(av, m)))[:, 1]
        # B3z
        g = GeradorObito(av, idade=(tr["age"].mean(), tr["age"].std()))
        s = keras.models.load_model(modelo_b3z, compile=False).predict(g, verbose=0)[:, 0]
        notas["B3z"] = pd.Series(s, index=g.info["exam_id"]).loc[av["exam_id"]].to_numpy()
        d = pd.DataFrame({"exam_id": av["exam_id"], "patient_id": av["patient_id"], "age": av["age"],
                          "is_male": av["is_male"], "y": av["y"], "normal_ecg": av["normal_ecg"],
                          **{r: av[r] for r in R}, **notas})
        d.to_csv(bruto, index=False, float_format="%.6f")           # saída bruta, logo depois das previsões
        print(f"saída bruta gravada: {bruto.name}", flush=True)

    modelos = ["B0", "B1", "B4", "B4p", "B3z"]
    y = d["y"].to_numpy().astype(int)

    if a.conjunto == "validacao":                                   # conferência com as notas salvas
        s1 = pd.read_csv(saida / f"parte_b_alteracoes_{tag}_previsoes_validacao.csv").set_index("exam_id").loc[d["exam_id"]]
        s2 = pd.read_csv(saida / f"parte_b_cnn_{tag}_previsoes_validacao.csv").set_index("exam_id").loc[d["exam_id"]]
        for m in modelos:
            ref = (s2 if m == "B1" else s1)[m].to_numpy()
            dif = np.abs(d[m].to_numpy() - ref).max()
            print(f"conferência {m}: dif. máx {dif:.2e} | AUROC aqui {roc_auc_score(y, d[m]):.4f} "
                  f"salvo {roc_auc_score(y, ref):.4f}", flush=True)
            assert dif < 1e-3, f"{m} não reproduz as notas salvas"

    notas = {m: d[m].to_numpy() for m in modelos}
    pares = [(p.split(" - ")[0], p.split(" - ")[1]) for p in cfg["comparacoes_no_teste"]["pares"]]
    ic = bootstrap(y, notas, pares)
    linhas = []
    for m in modelos + [f"{x} - {z}" for x, z in pares]:
        lin = dict(conjunto=a.conjunto, modelo=m, n=len(y), obitos=int(y.sum()))
        for met, f in (("auroc", roc_auc_score), ("auprc", average_precision_score)):
            lin[met] = f(y, notas[m]) if m in notas else f(y, notas[m.split(" - ")[0]]) - f(y, notas[m.split(" - ")[1]])
            lin[f"{met}_ic_inf"], lin[f"{met}_ic_sup"] = ic[(met, m)]
        linhas.append(lin)
    res = pd.DataFrame(linhas)
    res["partes"] = tag
    res.to_csv(saida / f"parte_b_{a.conjunto}_{tag}.csv", index=False, float_format="%.4f")

    alguma = d[R].astype(bool).any(axis=1).to_numpy()
    grupos = {"todos": np.ones(len(d), bool), "sem_nenhuma_das_6": ~alguma,
              "laudo_ecg_normal": d["normal_ecg"].astype(bool).to_numpy()}
    an = []
    for gname, msk in grupos.items():
        r_ = com_ic(y[msk], {m: notas[m][msk] for m in ("B0", "B4", "B4p", "B3z")})
        r_.insert(0, "grupo", gname)
        r_["n"], r_["obitos"] = int(msk.sum()), int(y[msk].sum())
        an.append(r_)
    an = pd.concat(an, ignore_index=True)
    an["conjunto"], an["partes"] = a.conjunto, tag
    an.to_csv(saida / f"parte_b_{a.conjunto}_analises_{tag}.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 220)
    print(res.round(4).to_string(index=False))
    print(an.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--conjunto", choices=["validacao", "teste"], required=True)
    ap.add_argument("--partes", default=",".join(map(str, range(18))))
    ap.add_argument("--confirmo-congelado", action="store_true")
    a = ap.parse_args()
    a.partes = [int(p) for p in a.partes.split(",")]
    main(a)
