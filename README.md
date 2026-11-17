# O que o ECG diz sobre o paciente, avaliado do jeito honesto

Projeto final de deep learning (Insper AI, Trilha de Trainees) de Tomaz Almeida e Renata Leventhal.
Base: CODE-15% (ECG de 12 derivações, 233.770 pacientes de Minas Gerais).

- **Parte A:** CNN 1D que detecta 6 alterações do ECG (multi-rótulo). Avaliada em pacientes novos (teste interno), no PTB-XL (Alemanha) e no CODE-TEST (gabarito de especialistas).
- **Parte B:** o ECG prevê morte em até 2 anos além do que idade e sexo já dizem?

O documento único de contexto, com decisões, resultados e log, é o [CLAUDE.md](CLAUDE.md). Leia-o antes de mexer em qualquer coisa.

## Apresentação e paper

Pasta `Apresentação + Paper/` (slides `.pptx` e paper `.pdf`).

## Mapa do repositório

Os scripts `.py`, os `.sh`, os notebooks e os JSONs de congelamento ficam **na raiz de propósito**: os scripts se importam pelo nome (`from ecg_dados import ...`) e derivam `code15/`, `resultados/`, `figuras/` e `modelos/` da própria localização. Não mova nenhum.

| Onde | O que tem |
|---|---|
| `projeto_ecg.ipynb` | notebook principal (entregável) |
| `projeto_arritmia.ipynb`, `arquivo/` | fase 1 (MIT-BIH), arquivada; plano antigo em `arquivo/CLAUDE_fase1_arritmia.md` |
| `ecg_dados.py`, `a0_baselines.py`, `a1_cnn.py`, `arquiteturas.py`, `ecg_medidas.py`, `robustez_comum.py` | biblioteca compartilhada: dados, split por paciente, gerador, métricas, modelos |
| `a1_cnn.py`, `arquiteturas.py`, `atributos_*.py`, `arvore_rf*.py`, `robustez_*.py`, `diagnostico_*.py`, `noite_*.py`, `sessao*.py`, `analise_erros_A3.py`, `seeds_A3.py` | Parte A: treino, baselines, robustez, diagnósticos |
| `parte_b_*.py` | Parte B (óbito em 2 anos) |
| `congelar.py`, `avaliar_externo.py`, `gerar_congelado_candidato.py`, `parte_b_teste.py`, `analise_testes.py`, `analise_posterior_testes.py`, `testar_*.py` | congelamento e testes |
| `figuras.py` | gera as figuras de `figuras/` a partir de `resultados/` |
| `colab_*.py`, `colab_*.sh`, `colab_treino.ipynb` | execução no Colab (GPU) |
| `docs/` | documentos de apoio (rascunhos de paper, relatórios, decisões pendentes, fundamentação médica, literatura, roteiro de slides) |
| `resultados/` | CSVs e saídas brutas; `resultados/experimentos.csv` é a tabela de experimentos gerada pelo código |
| `figuras/` | figuras (PNG) |
| `modelos/` | só os dois modelos congelados ficam no git; os demais são locais (ver `.gitignore`) |
| `logs/` | logs de execução |
| `revisao/` | revisões e conferências de números (registro histórico) |
| `code15/`, `ptbxl/`, `code_test/`, `.venv/` | dados e ambiente; **fora do git** |

## Modelos congelados e testes de uso único

- Modelos no git: `modelos/A3_melhor.keras` (Parte A) e `modelos/B3z_p0-17_melhor.keras` (Parte B).
- JSONs de congelamento (na raiz): `congelado_candidato_A3.json` e `congelado_candidato_parte_b.json` têm `status` = "congelado"; `congelado_candidato_arvore_rf.json` está como "candidato". O nome "candidato" no arquivo ficou do início do trabalho.
- **Os testes (interno, PTB-XL, CODE-TEST e o teste da Parte B) foram rodados uma vez cada, com o modelo congelado, e não devem ser rodados de novo nem usados para escolher nada.** `avaliar_externo.py` e `parte_b_teste.py` têm trava (recusam o teste sem status "congelado" e sem `--confirmo-congelado`). Saídas: `resultados/avaliacao_*_20261108_466e7a3.csv`, `resultados/bruto_*_20261108_466e7a3.*` (Parte A) e `resultados/parte_b_teste_*_p0-17.*` (Parte B). Detalhe em `docs/RELATORIO_TESTES.md`.

## Como reproduzir

Python 3.12 (3.11 falha), ambiente **só dentro do `.venv`**:

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt     # Mac/Linux: .venv/bin/python
```

Execute os scripts a partir da raiz, sem a opção `-I` do Python. Ordem:

1. `projeto_ecg.ipynb` (kernel `.venv`): seção 2 baixa `exams.csv` e a parte 0 (~2,7 GB) em `code15/`. Partes extras: `colab_baixar.py --partes ...`.
2. `a0_baselines.py`: baselines A0.
3. `a1_cnn.py treinar A1`, `treinar A2 --normalizar`, `treinar A3 --normalizar --partes 0,1,2,3 ...`: CNN da Parte A (opções completas no cabeçalho do script).
4. Parte A, extras de validação: `atributos_sinal.py`, `arvore_rf.py`, `atributos_pr.py`, `arvore_rf_pr.py`, `robustez_01..05_*.py`, `diagnostico_*.py`.
5. Parte B: `parte_b_tabular.py` (B0/B1) e, com as 18 partes, o roteiro de `colab_parte_b.sh` (`parte_b_cnn.py preparar`, `treinar_zero_idade`, `caracteristicas`, `avaliar`, depois `parte_b_alteracoes.py`, `parte_b_analises.py`, `parte_b_km.py`).
6. Testes (**uso único, já feitos**): `avaliar_externo.py --json congelado_candidato_A3.json --conjunto ... --confirmo-congelado` e `colab_teste_b.sh` / `parte_b_teste.py --conjunto teste --confirmo-congelado`.
7. `figuras.py`: regenera as figuras.

Passos 3 em diante assumem os dados baixados e levam de minutos a horas (CPU). Os números esperados estão em `CLAUDE.md` (seções 0 e 5.3).
