"""Avaliação de um modelo congelado, SEM retreinar, com os limiares e o pré-processamento do JSON.

TRAVA: em qualquer conjunto de teste (interno, PTB-XL, CODE-TEST) o script RECUSA rodar, antes de abrir qualquer dado, a menos que
  (1) o JSON tenha "status": "congelado"  E  (2) seja passada a flag --confirmo-congelado.
Só o conjunto "validacao_A3" roda livre (serve para conferir que o script reproduz os números já medidos).

Uso:
  python avaliar_externo.py --json congelado_candidato_A3.json --conjunto validacao_A3
  python avaliar_externo.py --json congelado_final.json --conjunto interno --confirmo-congelado   # SÓ depois de congelar

Nenhuma estatística é calculada no conjunto avaliado: normalização é por exame (só o próprio exame) e os limiares vêm do JSON.
IC de 95% por bootstrap (1.000 reamostras, seed 42), reamostrando PACIENTES quando há patient_id.
PROVISÓRIO, a confirmar com a Renata. Os caminhos "interno", "ptbxl" e "code_test" NÃO FORAM EXECUTADOS (código não rodado).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).parent
CONJUNTOS_DE_TESTE = ("interno", "ptbxl", "code_test")
LIVRES = ("validacao_A3",)
N_BOOT = 1000


def trava(cfg, conjunto, confirmo):
    """Roda ANTES de qualquer leitura de dados. Recusa teste sem status 'congelado' + flag."""
    if conjunto in LIVRES:
        return
    if conjunto not in CONJUNTOS_DE_TESTE:
        sys.exit(f"RECUSADO: conjunto desconhecido '{conjunto}'. Permitidos: {LIVRES + CONJUNTOS_DE_TESTE}.")
    if cfg.get("status") != "congelado":
        sys.exit(f"RECUSADO: '{conjunto}' é conjunto de teste e o JSON tem status '{cfg.get('status')}' (precisa ser 'congelado'). Nada foi lido.")
    if not confirmo:
        sys.exit(f"RECUSADO: '{conjunto}' é conjunto de teste; passe --confirmo-congelado para confirmar que o modelo está congelado. Nada foi lido.")


def sha256(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def carregar_conjunto(conjunto, cfg):
    """Devolve (gerador, info) com os exames na ordem fixa do gerador. Só roda depois da trava."""
    import ecg_dados as E
    partes = cfg["dados_de_treino"]["partes"]
    if conjunto == "validacao_A3":
        from a1_cnn import carregar_indices  # esta função NÃO devolve o grupo teste
        _, va, _ = carregar_indices(partes=partes)
        return E.GeradorECG(va, normalizar=True), None
    if conjunto == "interno":  # [não rodado] teste interno: grupo 'teste' do split, SEM filtro de exames zerados (regra vale só para treino/validação)
        info = E.construir_indice(partes)
        return E.GeradorECG(info[info.grupo == "teste"], normalizar=True), None
    # [não implementado] PTB-XL: falta decidir limite de likelihood e integrar a normalização ao GeradorPTBXL.
    # CODE-TEST: falta o carregador (reordenar AVL/AVF/AVR e ler o gold_standard).
    sys.exit(f"'{conjunto}' ainda não está implementado neste script (ver 'a decidir' no JSON). Nada foi lido.")


def metricas_exame(Y, S, limiares):
    """AUROC, AUPRC e, com os limiares dados, sensibilidade, precisão, F1, especificidade. Reaproveita 'metricas' do a0_baselines."""
    from a0_baselines import metricas
    return {r: metricas(Y[:, k].astype(int), S[:, k], limiares[r]) for k, r in enumerate(limiares)}


def com_macro(m):
    cols = ["auroc", "auprc", "sensibilidade", "precisao", "f1", "especificidade"]
    return {**m, "MACRO": {c: float(np.mean([m[r][c] for r in m])) for c in cols}}


def bootstrap(Y, S, limiares, pacientes, n=N_BOOT, seed=42):
    """Percentis 2,5 e 97,5 de cada métrica; reamostra pacientes (com todos os seus exames) quando há patient_id."""
    rng = np.random.default_rng(seed)
    if pacientes is not None:
        up = np.unique(pacientes)
        idx = {p: np.where(pacientes == p)[0] for p in up}
    cols = ["auroc", "auprc", "sensibilidade", "precisao", "f1", "especificidade"]
    amostras = []
    for _ in range(n):
        ii = np.concatenate([idx[p] for p in rng.choice(up, len(up))]) if pacientes is not None else rng.integers(0, len(Y), len(Y))
        m = com_macro(metricas_exame(Y[ii], S[ii], limiares))
        amostras.append({(r, c): m[r][c] for r in m for c in cols})
    df = pd.DataFrame(amostras)
    return {k: (float(np.percentile(df[k], 2.5)), float(np.percentile(df[k], 97.5))) for k in df.columns}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--conjunto", required=True, help="validacao_A3 | interno | ptbxl | code_test")
    ap.add_argument("--confirmo-congelado", action="store_true")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    a = ap.parse_args()

    cfg = json.loads((RAIZ / a.json).read_text(encoding="utf-8"))
    trava(cfg, a.conjunto, a.confirmo_congelado)  # <- antes de abrir qualquer dado

    import keras
    caminho = RAIZ / cfg["modelo"]["caminho"]
    if sha256(caminho) != cfg["modelo"]["sha256"]:
        sys.exit("RECUSADO: o SHA256 do modelo não bate com o do JSON.")
    modelo = keras.models.load_model(caminho)
    rotulos, limiares = cfg["rotulos"], cfg["limiares"]["valores"]

    g, _ = carregar_conjunto(a.conjunto, cfg)
    S = modelo.predict(g, verbose=0)  # sem retreinar; mesma normalização do JSON (normalizar=True no gerador)
    Y, pacientes = g.Y, g.info["patient_id"].to_numpy() if "patient_id" in g.info else None
    assert S.shape == Y.shape and list(limiares) == rotulos
    m = com_macro(metricas_exame(Y, S, limiares))
    ic = bootstrap(Y, S, limiares, pacientes, a.boot)

    print(f"conjunto: {a.conjunto} | exames: {len(Y)} | pacientes: {len(np.unique(pacientes)) if pacientes is not None else 'n/d'} | JSON status: {cfg['status']}")
    linhas = []
    for r in rotulos + ["MACRO"]:
        d = m[r]
        pos = int(Y[:, rotulos.index(r)].sum()) if r != "MACRO" else None
        print(f"{r}" + (f" (positivos {pos})" if pos is not None else "") + ": " + ", ".join(
            f"{c} {d[c]:.3f} [{ic[(r, c)][0]:.3f}-{ic[(r, c)][1]:.3f}]" for c in ("auroc", "auprc", "sensibilidade", "precisao", "f1", "especificidade")))
        linhas.append({"conjunto": a.conjunto, "rotulo": r, "positivos": pos, **{c: d[c] for c in ("auroc", "auprc", "sensibilidade", "precisao", "f1", "especificidade")},
                       **{f"{c}_ic_inf": ic[(r, c)][0] for c in ("auroc", "auprc", "sensibilidade", "precisao", "f1", "especificidade")},
                       **{f"{c}_ic_sup": ic[(r, c)][1] for c in ("auroc", "auprc", "sensibilidade", "precisao", "f1", "especificidade")}})
    pd.DataFrame(linhas).to_csv(RAIZ / "resultados" / f"avaliacao_{a.conjunto}.csv", index=False, float_format="%.4f")

    if a.conjunto == "validacao_A3":  # confere com o que já estava medido (resultados/experimentos.csv)
        ref = pd.read_csv(RAIZ / "resultados" / "experimentos.csv")
        ref = ref[(ref.experimento == "A3 provisório (partes 0 a 3; normalização por exame e derivação)") & (ref.rotulo == "MACRO")].iloc[0]
        for c in ("auroc", "auprc", "f1"):
            ok = abs(m["MACRO"][c] - ref[c]) < 6e-4
            print(f"confere com experimentos.csv: macro {c} {m['MACRO'][c]:.4f} vs {ref[c]:.4f} -> {'OK' if ok else 'DIFERENTE'}")


if __name__ == "__main__":
    main()
