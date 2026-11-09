"""Avaliação de um modelo congelado, SEM retreinar, com os limiares e o pré-processamento do JSON.

TRAVA: em qualquer conjunto de teste (interno, PTB-XL, CODE-TEST) o script RECUSA rodar, antes de abrir qualquer dado, a menos que
  (1) o JSON tenha "status": "congelado"  E  (2) seja passada a flag --confirmo-congelado.
Só o conjunto "validacao_A3" roda livre (serve para conferir que o script reproduz os números já medidos).

Uso:
  python avaliar_externo.py --json congelado_candidato_A3.json --conjunto validacao_A3
  python avaliar_externo.py --json congelado_candidato_A3.json --conjunto interno --confirmo-congelado   # SÓ depois de congelar (congelar.py)

Conjuntos (PROVISÓRIO, a confirmar com a Renata, salvo onde dito):
  interno   : grupo teste das partes 0 a 3, ordem fixa, mesma regra de exclusão do A3 (só exames com as 12 derivações zeradas, pelo sinal).
  ptbxl     : 21.799 registros menos os com as 12 derivações zeradas (mesma regra, lida do sinal cru; 0 na medição anterior); positivo = código SCP presente (limite de likelihood é parâmetro do JSON); registros com AFLT, IRBBB ou ILBBB
              saem só da avaliação do rótulo correspondente (AFLT->AF, IRBBB->RBBB, ILBBB->LBBB), também parâmetro; bootstrap por patient_id.
  code_test : 827 exames menos os com as 12 derivações zeradas (mesma regra, lida do sinal cru), SEM patient_id (bootstrap por exame), rótulo = gold_standard.csv. LIMITAÇÕES: AF tem só 13 positivos (IC largo);
              a sobreposição de pacientes com a CODE-15% é NÃO VERIFICADA (o CODE-TEST não tem identificador).
Nenhuma estatística é calculada no conjunto avaliado: normalização é por exame (só o próprio exame) e os limiares vêm do JSON.
IC de 95% por bootstrap (1.000 reamostras, seed 42).
Os caminhos interno, ptbxl e code_test NÃO FORAM EXECUTADOS COM O MODELO (código não rodado; só o formato dos dados foi testado, em testar_formato.py).
"""
import argparse
import ast
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).parent
CONJUNTOS_DE_TESTE = ("interno", "ptbxl", "code_test")
LIVRES = ("validacao_A3",)
N_BOOT = 1000
METRICAS = ("auroc", "auprc", "sensibilidade", "precisao", "f1", "especificidade")
# [PROV] padrões provisórios, a confirmar com a Renata (usados se o JSON não disser outra coisa)
LIMITE_LIKELIHOOD_PADRAO = "presente"  # código presente com qualquer valor (>= 0)
LIMITES_VALIDOS = ("presente", ">0", ">=50", "=100")
EXCLUSAO_PADRAO = {"AFLT": "AF", "IRBBB": "RBBB", "ILBBB": "LBBB"}  # registro com o código sai SÓ da avaliação do rótulo ao lado


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


# ------------------------------------------------------------------------------------------------ exclusão de exames zerados
def mascara_todo_zero(gerador_cru):
    """True nos exames com as 12 derivações zeradas, lidos do SINAL CRU (gerador com normalizar=False; a normalização zera derivações
    constantes e mudaria a regra). Mesma definição do caminho `interno` (ecg_dados.todo_zero = n_deriv_zeradas == 12). Nunca usa rótulos."""
    from ecg_dados import todo_zero
    return np.concatenate([todo_zero(gerador_cru[i]) for i in range(len(gerador_cru))]) if len(gerador_cru) else np.zeros(0, bool)


# ------------------------------------------------------------------------------------------------ rótulos do PTB-XL
def parametros_ptbxl(cfg):
    """Lê do JSON o limite de likelihood e a exclusão AFLT/IRBBB/ILBBB. 'a decidir' ou ausente -> padrão provisório."""
    p = cfg.get("ptbxl", {})
    limite = p.get("limite_de_likelihood", LIMITE_LIKELIHOOD_PADRAO)
    if limite not in LIMITES_VALIDOS:
        limite = LIMITE_LIKELIHOOD_PADRAO
    excl = p.get("exclusao_por_rotulo", EXCLUSAO_PADRAO)
    if not isinstance(excl, dict):  # False/None = exclusão desligada
        excl = {}
    return limite, excl


def rotulos_ptbxl(info, mapa, limite, exclusao):
    """Y (n,6), máscara de avaliação (n,6) e contagens, a partir de scp_codes. Só usa o campo de rótulos; não olha sinal nem modelo.
    Positivo = o código do mapa aparece com likelihood que passa no `limite`. Negativo = qualquer outro caso.
    Máscara: o registro com o código de `exclusao` (qualquer valor) sai da avaliação do rótulo correspondente."""
    from ecg_dados import ROTULOS
    scp = info["scp_codes"].apply(ast.literal_eval)
    teste = {"presente": lambda v: v >= 0, ">0": lambda v: v > 0, ">=50": lambda v: v >= 50, "=100": lambda v: v == 100}[limite]
    Y = np.zeros((len(info), len(ROTULOS)), np.float32)
    mask = np.ones_like(Y, dtype=bool)
    for k, r in enumerate(ROTULOS):
        Y[:, k] = scp.apply(lambda d: float(mapa[r] in d and teste(d[mapa[r]])))
    for codigo, r in exclusao.items():
        mask[:, ROTULOS.index(r)] &= ~scp.apply(lambda d: codigo in d).to_numpy()
    resumo = {r: {"positivos_sem_exclusao": int(Y[:, k].sum()), "positivos": int((Y[:, k] * mask[:, k]).sum()), "avaliados": int(mask[:, k].sum()),
                  "excluidos": int((~mask[:, k]).sum())} for k, r in enumerate(ROTULOS)}
    resumo["registros_com_AFLT_IRBBB_ILBBB"] = int(scp.apply(lambda d: any(c in d for c in EXCLUSAO_PADRAO)).sum())
    return Y, mask, resumo


# ------------------------------------------------------------------------------------------------ conjuntos
def carregar_conjunto(conjunto, cfg):
    """Devolve dict(gerador, Y, mask, pacientes, resumo). NUNCA é chamada pelo main antes da trava.
    (testar_formato.py também a chama, para checar só formato e contagem de rótulos, sem modelo.)"""
    import ecg_dados as E
    partes = cfg["dados_de_treino"]["partes"]
    if conjunto == "validacao_A3":
        from a1_cnn import carregar_indices  # esta função NÃO devolve o grupo teste
        _, va, _ = carregar_indices(partes=partes)
        g = E.GeradorECG(va, normalizar=True)
        return dict(gerador=g, Y=g.Y, mask=np.ones(g.Y.shape, bool), pacientes=g.info["patient_id"].to_numpy(), resumo={"exames": len(g.info)})
    if conjunto == "interno":
        info = E.construir_indice(partes)
        teste = info[info.grupo == "teste"]
        # MESMA regra de exclusão do A3 (provisória): só exames com as 12 derivações zeradas, decidida pelo sinal (lista), nunca pelos rótulos
        sai = teste.exam_id.isin(E.exames_todo_zero(partes))
        teste = teste[~sai].reset_index(drop=True)
        g = E.GeradorECG(teste, normalizar=True)  # ordem fixa (parte, linha)
        print(f"interno: excluídos por terem as 12 derivações zeradas: {int(sai.sum())} de {len(sai)}")
        resumo = {"exames_antes": int(len(sai)), "exames_excluidos_todo_zero": int(sai.sum()), "exames": len(teste),
                  "positivos": {r: int(teste[r].sum()) for r in E.ROTULOS}, "pacientes": int(teste.patient_id.nunique())}
        return dict(gerador=g, Y=g.Y, mask=np.ones(g.Y.shape, bool), pacientes=g.info["patient_id"].to_numpy(), resumo=resumo)
    if conjunto == "ptbxl":
        info = E.ptbxl_indice()  # 21.799 registros, ordem de ecg_id
        limite, excl = parametros_ptbxl(cfg)
        mapa = cfg["ptbxl"]["mapeamento_de_rotulos"]
        sai = mascara_todo_zero(E.GeradorPTBXL(info, normalizar=False))  # passada extra de leitura (~12 min), só pelo sinal
        print(f"ptbxl: excluídos por terem as 12 derivações zeradas: {int(sai.sum())} de {len(info)}")
        info = info[~sai]
        Y, mask, resumo = rotulos_ptbxl(info, mapa, limite, excl)
        resumo.update(limite_de_likelihood=limite, exclusao=excl, registros=len(info), registros_excluidos_todo_zero=int(sai.sum()),
                      pacientes=int(info.patient_id.nunique()))
        g = E.GeradorPTBXL(info, normalizar=True)  # 500 -> 400 Hz, 48 zeros de cada lado, depois normalizar_exame (A2)
        return dict(gerador=g, Y=Y, mask=mask, pacientes=info["patient_id"].to_numpy(), resumo=resumo)
    if conjunto == "code_test":
        sai = mascara_todo_zero(E.GeradorCodeTest(normalizar=False))  # só pelo sinal
        print(f"code_test: excluídos por terem as 12 derivações zeradas: {int(sai.sum())} de {len(sai)}")
        g = E.GeradorCodeTest(normalizar=True, linhas=np.where(~sai)[0])  # reordena [0,1,2,5,3,4,...], float64 -> float32
        resumo = {"exames_antes": len(sai), "exames_excluidos_todo_zero": int(sai.sum()), "exames": g.n, "positivos": {r: int(g.Y[:, k].sum()) for k, r in enumerate(E.ROTULOS)},
                  "aviso": "AF tem so 13 positivos; sobreposicao de pacientes com a CODE-15% e NAO VERIFICADA (sem identificador)"}
        return dict(gerador=g, Y=g.Y, mask=np.ones(g.Y.shape, bool), pacientes=None, resumo=resumo)  # sem patient_id -> bootstrap por exame
    sys.exit(f"conjunto desconhecido: {conjunto}")


# ------------------------------------------------------------------------------------------------ métricas
def metricas_exame(Y, S, limiares, mask):
    """AUROC, AUPRC e, com os limiares dados, sensibilidade, precisão, F1, especificidade; cada rótulo só nos exames com mask[:, k]."""
    from a0_baselines import metricas
    out = {}
    for k, r in enumerate(limiares):
        m = mask[:, k]
        out[r] = metricas(Y[m, k].astype(int), S[m, k], limiares[r])
    return out


def com_macro(m):
    return {**m, "MACRO": {c: float(np.mean([m[r][c] for r in m])) for c in METRICAS}}


def bootstrap(Y, S, limiares, mask, pacientes, n=N_BOOT, seed=42):
    """Percentis 2,5 e 97,5 de cada métrica. Reamostra PACIENTES (com todos os seus exames) quando há patient_id; senão, exames."""
    rng = np.random.default_rng(seed)
    if pacientes is not None:
        up = np.unique(pacientes)
        idx = {p: np.where(pacientes == p)[0] for p in up}
    amostras = []
    for _ in range(n):
        ii = np.concatenate([idx[p] for p in rng.choice(up, len(up))]) if pacientes is not None else rng.integers(0, len(Y), len(Y))
        try:
            m = com_macro(metricas_exame(Y[ii], S[ii], limiares, mask[ii]))
        except ValueError:  # reamostra sem positivo (ou sem negativo) num rótulo: AUROC indefinido; descarta a reamostra
            continue
        amostras.append({(r, c): m[r][c] for r in m for c in METRICAS})
    df = pd.DataFrame(amostras)
    return {k: (float(np.percentile(df[k], 2.5)), float(np.percentile(df[k], 97.5))) for k in df.columns}, len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--conjunto", required=True, help="validacao_A3 | interno | ptbxl | code_test")
    ap.add_argument("--confirmo-congelado", action="store_true")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    a = ap.parse_args()

    cfg = json.loads((RAIZ / a.json).read_text(encoding="utf-8"))
    trava(cfg, a.conjunto, a.confirmo_congelado)  # <- antes de abrir qualquer dado
    pendentes = [k for k, v in cfg.get("ptbxl", {}).items() if isinstance(v, str) and v.lower().startswith("a decidir")]
    if a.conjunto not in LIVRES and pendentes:
        sys.exit(f"RECUSADO: o JSON ainda tem itens 'a decidir' em ptbxl: {pendentes}. Decida e congele com congelar.py antes de avaliar um conjunto de teste.")

    import keras
    caminho = RAIZ / cfg["modelo"]["caminho"]
    if sha256(caminho) != cfg["modelo"]["sha256"]:
        sys.exit("RECUSADO: o SHA256 do modelo não bate com o do JSON.")
    modelo = keras.models.load_model(caminho)
    rotulos, limiares = cfg["rotulos"], cfg["limiares"]["valores"]

    # Saída bruta (notas do modelo) gravada logo depois do predict, antes do bootstrap. Se já existir e abrir íntegra, o predict NÃO é refeito.
    brutos = sorted((RAIZ / "resultados").glob(f"bruto_{a.conjunto}_*.npz")) if a.conjunto not in LIVRES else []
    if brutos:
        b = np.load(brutos[-1], allow_pickle=False)
        S, Y, mask, pacientes = b["S"], b["Y"], b["mask"], (b["pacientes"] if b["pacientes"].size else None)
        print(f"saída bruta já existe ({brutos[-1].name}): predict NÃO refeito; só as métricas são recalculadas. resumo: {b['resumo']}")
    else:
        d = carregar_conjunto(a.conjunto, cfg)
        print("resumo do conjunto:", d["resumo"])
        Y, mask, pacientes = d["Y"], d["mask"], d["pacientes"]
        S = modelo.predict(d["gerador"], verbose=0)  # sem retreinar; a normalização do JSON já está no gerador
        if a.conjunto not in LIVRES:
            gi = d["gerador"].info
            ids = gi["exam_id"] if "exam_id" in gi else (gi["linha"] if "linha" in gi else gi.index)  # interno: exam_id; ptbxl: ecg_id; code_test: linha
            commit = subprocess.run(["git", "-C", str(RAIZ), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
            sujo = subprocess.run(["git", "-C", str(RAIZ), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
            nome = RAIZ / "resultados" / f"bruto_{a.conjunto}_{datetime.date.today():%Y%m%d}_{commit}{'-sujo' if sujo else ''}.npz"
            tmp = nome.with_suffix(".tmp")
            with open(tmp, "wb") as f:  # escreve em arquivo temporário e renomeia: uma queda no meio não deixa um .npz pela metade
                np.savez(f, S=S, Y=Y, mask=mask, ids=np.asarray(ids), pacientes=np.array([]) if pacientes is None else pacientes,
                         resumo=json.dumps(d["resumo"], default=str), json_sha256=sha256(RAIZ / a.json), modelo_sha256=cfg["modelo"]["sha256"])
            os.replace(tmp, nome)
            print(f"saída bruta salva: {nome.name}")
    assert S.shape == Y.shape and list(limiares) == rotulos
    m = com_macro(metricas_exame(Y, S, limiares, mask))
    ic, n_validas = bootstrap(Y, S, limiares, mask, pacientes, a.boot)

    print(f"conjunto: {a.conjunto} | exames: {len(Y)} | pacientes: {len(np.unique(pacientes)) if pacientes is not None else 'n/d (bootstrap por exame)'} "
          f"| JSON status: {cfg['status']} | reamostras válidas: {n_validas}")
    linhas = []
    for r in rotulos + ["MACRO"]:
        x = m[r]
        k = rotulos.index(r) if r != "MACRO" else None
        pos = int((Y[:, k] * mask[:, k]).sum()) if k is not None else None
        print(f"{r}" + (f" (positivos {pos} de {int(mask[:, k].sum())})" if pos is not None else "") + ": " + ", ".join(
            f"{c} {x[c]:.3f} [{ic[(r, c)][0]:.3f}-{ic[(r, c)][1]:.3f}]" for c in METRICAS))
        linhas.append({"conjunto": a.conjunto, "rotulo": r, "positivos": pos, **{c: x[c] for c in METRICAS},
                       **{f"{c}_ic_inf": ic[(r, c)][0] for c in METRICAS}, **{f"{c}_ic_sup": ic[(r, c)][1] for c in METRICAS}})
    pd.DataFrame(linhas).to_csv(RAIZ / "resultados" / f"avaliacao_{a.conjunto}.csv", index=False, float_format="%.4f")

    if a.conjunto == "validacao_A3":  # confere com o que já estava medido (resultados/experimentos.csv)
        ref = pd.read_csv(RAIZ / "resultados" / "experimentos.csv")
        ref = ref[(ref.experimento == "A3 provisório (partes 0 a 3; normalização por exame e derivação)") & (ref.rotulo == "MACRO")].iloc[0]
        for c in ("auroc", "auprc", "f1"):
            ok = abs(m["MACRO"][c] - ref[c]) < 6e-4
            print(f"confere com experimentos.csv: macro {c} {m['MACRO'][c]:.4f} vs {ref[c]:.4f} -> {'OK' if ok else 'DIFERENTE'}")


if __name__ == "__main__":
    main()
