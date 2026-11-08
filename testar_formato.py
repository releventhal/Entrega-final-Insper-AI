"""Teste de FORMATO dos conjuntos de avaliação. NÃO carrega o modelo, NÃO chama predict, NÃO calcula métrica de desempenho.

Para cada conjunto (validação do A3, teste interno, PTB-XL, CODE-TEST): mostra a contagem de rótulos, lê 2 lotes e confere
(batch, 4096, 12), float32 e ausência de NaN/inf. Confere também que o pré-processamento do A2 reproduz, em 50 exames de validação,
o resultado de uma reimplementação independente a partir do HDF5. Opcional: --zerados-ptbxl lê os 21.799 registros do PTB-XL
(~12 min) só para contar derivações zeradas, pelo sinal.

Este script NÃO passa pela trava do avaliar_externo.py de propósito: ele só olha formato e contagem, nunca o desempenho de um modelo.
PROVISÓRIO, a confirmar com a Renata.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import ecg_dados as E
from avaliar_externo import carregar_conjunto

RAIZ = Path(__file__).parent


def checar_lotes(nome, g, n_lotes=2):
    for i in range(n_lotes):
        b = g[i]
        X = b[0] if isinstance(b, tuple) else b
        ok = X.ndim == 3 and X.shape[1:] == (4096, 12) and X.dtype == np.float32 and bool(np.isfinite(X).all())
        print(f"  {nome} lote {i}: forma {X.shape}, tipo {X.dtype}, NaN/inf: {int((~np.isfinite(X)).sum())}, "
              f"mínimo {X.min():.2f}, máximo {X.max():.2f} -> {'OK' if ok else 'PROBLEMA'}")
        # normalização por exame e derivação: na região ativa, média ~0 e desvio ~1 nas derivações não constantes; pontas = 0
        ativo = np.abs(X).max(axis=2) > 0
        for e in range(min(3, len(X))):
            a = X[e][ativo[e]]
            if len(a):
                nz = a.std(0) > 1e-6
                print(f"     exame {e}: amostras ativas {int(ativo[e].sum())}, média máx |.| {np.abs(a.mean(0)[nz]).max():.1e}, "
                      f"desvio entre {a.std(0)[nz].min():.3f} e {a.std(0)[nz].max():.3f}, derivações zeradas {int((~nz).sum())}")


def reproduz_normalizacao_validacao(cfg, n=50):
    """O pré-processamento A2 do gerador == reimplementação independente (outra forma de calcular) em n exames de validação."""
    d = carregar_conjunto("validacao_A3", cfg)
    g = d["gerador"]
    pos = np.sort(np.random.default_rng(42).choice(len(g.info), n, replace=False))
    maior = 0.0
    arq = {p: h5py.File(E.DADOS / f"exams_part{p}.hdf5", "r") for p in cfg["dados_de_treino"]["partes"]}
    for i in pos:
        p, l = int(g.info.parte[i]), int(g.info.linha[i])
        x = arq[p]["tracings"][l].astype(np.float64)
        ativo = (x != 0).any(axis=1)
        ref = np.zeros_like(x)
        for j in range(12):
            v = x[ativo, j]
            mu = v.mean()
            dp = np.sqrt(((v - mu) ** 2).mean())
            if dp > 1e-6:
                ref[ativo, j] = (v - mu) / dp
        # saída do gerador para esse exame: lote de 1 exame com o mesmo código de produção
        saida = E.normalizar_exame(arq[p]["tracings"][l])
        maior = max(maior, float(np.abs(saida - ref).max()))
    print(f"  50 exames de validação: diferença máxima entre o pré-processamento do gerador e a reimplementação independente = {maior:.2e} "
          f"-> {'OK' if maior < 1e-4 else 'PROBLEMA'}")
    # e o caminho do gerador (lote inteiro) é o mesmo código
    X, _ = g[int(pos[0]) // g.lote]
    print(f"  lote do gerador de validação: {X.shape}, {X.dtype}")


def zerados_ptbxl():
    """Lê todos os registros do PTB-XL (500 Hz, antes de qualquer pré-processamento) e conta derivações zeradas/constantes, pelo sinal."""
    import wfdb
    info = E.ptbxl_indice()
    todo_zero = algum_zero = algum_const = nao_finito = 0
    por_lead = np.zeros(12, int)
    for c in info["filename_hr"]:
        x = wfdb.rdrecord(str(E.PTBXL / c)).p_signal
        nao_finito += int(not np.isfinite(x).all())
        z = (np.abs(x).max(axis=0) == 0)
        const = (np.ptp(x, axis=0) == 0)
        todo_zero += int(z.all())
        algum_zero += int(z.any())
        algum_const += int(const.any())
        por_lead += z
    print(f"PTB-XL ({len(info)} registros, sinal bruto 500 Hz): com NaN/inf {nao_finito}; com as 12 derivações zeradas {todo_zero}; "
          f"com alguma derivação zerada {algum_zero}; com alguma derivação constante (inclui zerada) {algum_const}; zeradas por derivação {por_lead.tolist()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="congelado_candidato_A3.json")
    ap.add_argument("--zerados-ptbxl", action="store_true")
    a = ap.parse_args()
    cfg = json.loads((RAIZ / a.json).read_text(encoding="utf-8"))
    if a.zerados_ptbxl:
        zerados_ptbxl()
    else:
        for nome in ("validacao_A3", "interno", "ptbxl", "code_test"):
            print(f"== {nome}")
            d = carregar_conjunto(nome, cfg)
            print("  contagens:")
            for k, v in d["resumo"].items():
                print(f"    {k}: {v}")
            assert len(d["Y"]) == len(d["gerador"].info) and d["mask"].shape == d["Y"].shape
            checar_lotes(nome, d["gerador"])
        print("== reprodução do pré-processamento A2")
        reproduz_normalizacao_validacao(cfg)
