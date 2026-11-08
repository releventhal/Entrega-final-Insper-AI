"""Gera `congelado_candidato_A3.json`: tudo o que define o A3 antes de qualquer teste externo.
Status "candidato": NÃO é congelado. Quem congela é a dupla (Tom e Renata), mudando o status à mão para "congelado".
Usa só treino e validação (os limiares vêm da validação completa do A3). Não abre teste, PTB-XL nem CODE-TEST.
Uso (raiz do repositório, Python do .venv): python gerar_congelado_candidato.py
"""
import hashlib
import json
import subprocess
from datetime import date
from importlib.metadata import version
from pathlib import Path

import numpy as np

from a0_baselines import melhor_limiar_f1
from a1_cnn import carregar_indices
from ecg_dados import DADOS, ROTULOS, SEED

RAIZ = Path(__file__).parent
PARTES = [0, 1, 2, 3]
MODELO = RAIZ / "modelos" / "A3_melhor.keras"


def sha256(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def main():
    tr, va, rel = carregar_indices(partes=PARTES)
    va = va.sort_values(["parte", "linha"], ignore_index=True)  # mesma ordem do GeradorECG (previsões salvas)
    S = np.load(RAIZ / "resultados" / "A3_previsoes_validacao.npy")
    assert S.shape == (len(va), len(ROTULOS))
    # limiar por rótulo = o que maximiza o F1 na validação COMPLETA do A3, com precisão total (o CSV guarda só 4 casas)
    limiares = {r: melhor_limiar_f1(va[r].to_numpy().astype(int), S[:, k]) for k, r in enumerate(ROTULOS)}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=RAIZ).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=RAIZ).stdout.strip())
    j = {
        "status": "candidato",
        "aviso": "NAO congelado. Enquanto o status for 'candidato', avaliar_externo.py recusa rodar em qualquer conjunto de teste. "
                 "Congelar = decisao da dupla (Tom e Renata): mudar o status para 'congelado' SO depois de decidir os itens 'a decidir'.",
        "provisorio": "todo o conteudo e provisorio, a confirmar com a Renata",
        "gerado_em": str(date.today()),
        "modelo": {
            "caminho": "modelos/A3_melhor.keras",
            "sha256": sha256(MODELO),
            "origem": "melhor epoca pela perda de validacao (A3, epoca 4 de 7)",
            "arquitetura": "A1 (secao 5.3 do CLAUDE.md): 4 x Conv1D(32,64,128,128; kernel 15; same; ReLU), MaxPool 4 nas tres primeiras, GlobalAveragePooling1D, Dense 6 sigmoid",
            "entrada": "(N, 4096, 12) float32, 400 Hz, derivacoes DI, DII, DIII, AVR, AVL, AVF, V1-V6",
            "saida": "(N, 6) sigmoid, na ordem dos rotulos abaixo",
        },
        "rotulos": ROTULOS,
        "preprocessamento": {
            "nome": "A2: normalizacao por exame e por derivacao (ecg_dados.normalizar_exame)",
            "descricao": "Para cada exame (4096 x 12): a regiao ativa e o conjunto de amostras em que ALGUMA derivacao e diferente de zero "
                         "(isto exclui o preenchimento de zeros das pontas). Em cada derivacao, media e desvio padrao populacional (np.std, ddof=0) "
                         "sao calculados SO na regiao ativa, usando apenas o proprio exame. Na regiao ativa: x = (x - media) / desvio. "
                         "O preenchimento continua zero. Derivacao com desvio <= 1e-6 (zerada ou constante) fica zero. Exame sem nenhuma amostra ativa e devolvido sem mudanca. "
                         "Nenhuma estatistica e calculada em treino, validacao ou teste inteiros; nenhum rotulo e usado.",
            "para_ptbxl": "ecg_dados.preprocessar_ptbxl (resample_poly up=4 down=5 de 500 para 400 Hz = 4000 amostras uteis; 48 zeros de cada lado ate 4096; "
                          "ordem das colunas ja igual a da CODE-15%), DEPOIS normalizar_exame (GeradorPTBXL(normalizar=True)); a regiao ativa exclui os 48 zeros de cada lado.",
            "para_code_test": "ecg_dados.GeradorCodeTest: reordenar colunas [0,1,2,5,3,4,6..11] (AVL, AVF, AVR -> AVR, AVL, AVF), float64 -> float32, depois normalizar_exame.",
            "para_teste_interno": "GeradorECG(normalizar=True) em ordem fixa sobre o grupo teste das partes 0 a 3, sem os exames com as 12 derivacoes zeradas.",
        },
        "dados_de_treino": {
            "partes": PARTES,
            "split": "por patient_id, np.random.default_rng(42).random sobre np.sort(pacientes unicos da tabela inteira): <0,7 treino, <0,8 validacao, resto teste",
            "regra_de_exclusao": "descartar so os exames com as 12 derivacoes zeradas (lista code15/exames_zerados_p0-3.npy, n_deriv_zeradas == 12): "
                                 "no treino e na validacao (como no A3) e, na avaliacao, tambem no teste interno (mesma regra, provisoria); "
                                 "os parciais (1 a 11 derivacoes zeradas) ficam; regra so pelo sinal, nunca pelos rotulos",
            "descartados": rel,
            "n_treino": int(len(tr)),
            "n_validacao": int(len(va)),
        },
        "limiares": {
            "regra": "por alteracao, o limiar que maximiza o F1 na validacao completa do A3 (a0_baselines.melhor_limiar_f1)",
            "valores": limiares,
        },
        "seed": SEED,
        "treino": {"epocas_rodadas": 7, "melhor_epoca": 4, "otimizador": "adam", "loss": "binary_crossentropy", "batch": 64, "early_stopping": "val_loss, paciencia 3"},
        "ambiente": {"python_keras": version("keras"), "tensorflow": version("tensorflow"), "numpy": version("numpy"), "scipy": version("scipy")},
        "git": {"commit_base": commit, "arvore_com_alteracoes_nao_commitadas": dirty,
                "nota": "commit antes de criar este arquivo; o proprio JSON entra no commit seguinte"},
        "ptbxl": {
            "mapeamento_de_rotulos": {"1dAVb": "1AVB", "RBBB": "CRBBB", "LBBB": "CLBBB", "SB": "SBRAD", "ST": "STACH", "AF": "AFIB"},
            "limite_de_likelihood": "presente",
            "limite_de_likelihood_estado": "provisorio, a confirmar com a Renata (criterio proposto so pela documentacao: codigo presente com qualquer valor; opcoes: presente, >0, >=50, =100)",
            "exclusao_por_rotulo": {"AFLT": "AF", "IRBBB": "RBBB", "ILBBB": "LBBB"},
            "exclusao_por_rotulo_estado": "provisorio, a confirmar com a Renata (padrao ligado: o registro com o codigo sai so da avaliacao do rotulo ao lado; false desliga)",
            "folds": "todos os 21.799 registros (provisorio, a confirmar com a Renata)",
        },
        "code_test": {
            "arquivo": "code_test/extraido/data/ecg_tracings.hdf5 (827 x 4096 x 12, float64, 400 Hz)",
            "rotulos": "annotations/gold_standard.csv, so como rotulo",
            "bootstrap": "por exame (sem patient_id)",
            "limitacoes": "AF tem so 13 positivos; sobreposicao de pacientes com a CODE-15% NAO VERIFICADA (o CODE-TEST nao tem identificador)",
        },
    }
    saida = RAIZ / "congelado_candidato_A3.json"
    saida.write_text(json.dumps(j, ensure_ascii=False, indent=2), encoding="utf-8")
    print("escrito:", saida.name, "| status:", j["status"], "| sha256 do modelo:", j["modelo"]["sha256"][:16], "...")
    print("limiares:", {r: round(v, 4) for r, v in limiares.items()})


if __name__ == "__main__":
    main()
