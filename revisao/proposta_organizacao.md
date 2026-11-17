# Proposta de organização do repositório (Fase 1, só análise; 17/11)

Nada foi movido. Este arquivo é novo e sem commit.

## 1. Estado atual (medido)

- Repositório rastreado: ~102 MB. Dados grandes (`code15/`, `ptbxl/`, `code_test/` 519 MB, `.venv/`) já ficam fora do git.
- Raiz: 77 itens soltos.
- Pastas: `resultados/` (147 itens, 74 MB, subpastas `erros_A3/`, `posterior_testes/`), `figuras/` (16 MB, 44 arquivos), `logs/` (11 arquivos), `modelos/` (2 `.keras` rastreados), `revisao/` (4), `arquivo/` (1), `Aprensentação + Paper/` (não rastreada), `code_test/` (ignorada).

### Agrupamento por função (raiz)

| Grupo | Arquivos |
|---|---|
| Biblioteca compartilhada (importada por quase tudo) | `ecg_dados.py`, `a0_baselines.py`, `a1_cnn.py`, `arquiteturas.py`, `ecg_medidas.py`, `robustez_comum.py` |
| Parte A, treino e baselines | `a0_baselines.py`, `a1_cnn.py`, `arquiteturas.py`, `atributos_sinal.py`, `atributos_pr.py`, `arvore_rf.py`, `arvore_rf_pr.py`, `arvore_rf_pr_experimentos.py`, `seeds_A3.py` |
| Parte A, robustez e diagnósticos | `robustez_01..06_*.py`, `diagnostico_{neurokit,pooling,preenchimento}.py`, `analise_erros_A3.py`, `sessao_sem_gpu.py`, `sessao2_a5.py`, `sessao2_conferencia.py`, `sessao_a5_parcial.py`, `noite_fila.py`, `noite_parcial.py`, `noite_passo0.py` |
| Congelamento e testes de uso único | `congelar.py`, `avaliar_externo.py`, `gerar_congelado_candidato.py`, `testar_formato.py`, `testar_todo_zero.py`, `congelado_candidato_{A3,arvore_rf,parte_b}.json`, `analise_testes.py`, `analise_posterior_testes.py`, `parte_b_teste.py` |
| Parte B | `parte_b_{tabular,cnn,alteracoes,analises,b4p_checagem,comparar,idade,km,seeds,teste}.py` |
| Figuras | `figuras.py` (script; importado por `analise_testes.py` e `analise_posterior_testes.py`) |
| Colab | `colab_baixar.py`, `colab_parte_b.sh`, `colab_seeds.sh`, `colab_teste_b.sh`, `colab_treino.ipynb` |
| Notebooks | `projeto_ecg.ipynb` (principal, 1,8 MB), `projeto_arritmia.ipynb` (fase 1), `teste_viabilidade_code15_colab.ipynb` (descartável) |
| Documentos (.md) | `CLAUDE.md`, `README.md` (25 bytes, quase vazio), `decisoes_pendentes.md`, `fundamentacao_medica.md`, `literatura_modelos_abertos.md`, `parte_a_rascunho.md`, `parte_b_rascunho.md`, `RELATORIO_NOITE.md`, `RELATORIO_TESTES.md`, `roteiro_slides.md` |
| Outros | `requirements.txt`, `entrega_final.pdf` (enunciado), `.gitignore` |

## 2. Dependências de caminho (grep)

**Imports por nome na raiz (confirmado):** os scripts usam `from ecg_dados import ...`, `from a1_cnn import ...`, `from a0_baselines import ...`, `import a1_cnn`, `import arquiteturas`, `import avaliar_externo as ae`, `from robustez_comum import ...`, `from parte_b_cnn import ...`, `from figuras import ...` etc. O notebook principal faz `import ecg_dados`, `from a1_cnn import ...`, `from parte_b_tabular import ...` e chama os scripts pelo nome. Mover qualquer `.py` quebra a cadeia.

**Caminhos relativos à localização do script:** ~15 scripts usam `Path(__file__).parent` (ou `.resolve().parent`) como raiz e dali derivam `code15/`, `ptbxl/`, `code_test/`, `resultados/`, `figuras/`, `modelos/`. Se o script sair da raiz, todos esses caminhos apontam para o lugar errado.

**JSONs de congelamento:** lidos da raiz por `avaliar_externo.py`, `congelar.py`, `parte_b_teste.py`, `parte_b_cnn.py`, `robustez_comum.py`, `analise_*`, `arvore_rf.py`, `gerar_congelado_candidato.py`, `testar_formato.py`, `colab_teste_b.sh`, `colab_treino.ipynb` e `projeto_ecg.ipynb`. Também gravam SHA256 e commit base.

**Shell:** `colab_*.sh` chamam `colab_baixar.py`, `parte_b_*.py` e `ecg_dados.py` pelo nome, na raiz.

**.md lidos por código:** nenhum. `sessao2_conferencia.py` cita `roteiro_slides.md` e `CLAUDE.md` só como rótulos de texto, não abre os arquivos. `sessao_sem_gpu.py` apenas grava um `.md` em `resultados/`.

**Referências cruzadas entre os .md** (nomes simples, sem pasta):

| Documento | Citado por |
|---|---|
| `decisoes_pendentes.md` | CLAUDE.md (9x), RELATORIO_NOITE.md, roteiro_slides.md, `revisao/status_renata.txt` |
| `fundamentacao_medica.md` | CLAUDE.md (4x), parte_a_rascunho.md, parte_b_rascunho.md, `revisao/*` |
| `literatura_modelos_abertos.md` | CLAUDE.md (2x), decisoes_pendentes.md, parte_a_rascunho.md |
| `parte_a_rascunho.md` | CLAUDE.md (2x), parte_b_rascunho.md |
| `parte_b_rascunho.md` | CLAUDE.md (4x), `revisao/*` (36 menções, registro histórico da revisão) |
| `RELATORIO_NOITE.md` | CLAUDE.md (2x), decisoes_pendentes.md (3x) |
| `RELATORIO_TESTES.md` | CLAUDE.md (6x), parte_a_rascunho.md, `projeto_ecg.ipynb` (1 menção em texto de célula markdown) |
| `roteiro_slides.md` | CLAUDE.md (2x), RELATORIO_NOITE.md, `revisao/*` |

Como os 8 .md vão juntos para `docs/`, as menções de um para o outro (nome simples) continuam coerentes. Só CLAUDE.md e o notebook (1 texto) precisam ganhar o prefixo `docs/`. Os arquivos em `revisao/` são registro histórico; não proponho editá-los.

**NÃO PODEM mudar de lugar sem quebrar algo:** todos os `.py` e `.sh` da raiz, os 3 `congelado_*.json`, `projeto_ecg.ipynb`, `colab_treino.ipynb`, `requirements.txt` (instruções de instalação), `CLAUDE.md` (lido por IAs na raiz), `figuras/`, `resultados/`, `modelos/` (caminhos fixos em dezenas de scripts e no notebook).

## 3. Duplicatas, temporários, .gitignore

- **Duplicatas exatas rastreadas (não mexer):** `resultados/avaliacao_{interno,ptbxl}.csv` = `..._20261108_466e7a3.csv`; `resultados/bruto_interno_20261108_466e7a3.log` = `logs/teste_interno_...log`; `figuras/robustez_gradcam_X.png` = `..._logit_X.png` (5 pares); 2 pares de `A5_completo_*`. Estão em áreas protegidas (testes) ou ligadas a scripts que os leem pelo nome. Deixar.
- **`logs/` já tem 11 arquivos rastreados.** Colocar `logs/` no `.gitignore` só impede novos; não desrastreia os antigos (desrastrear = `git rm --cached`, mudança maior; não proponho).
- **`revisao/status.txt` não existe;** o que existe e é rastreado é `revisao/status_renata.txt`. Ignorar `revisao/status.txt` é inofensivo.
- Sem `__pycache__`, `.pyc` ou checkpoints rastreados. `.venv/`, `__pycache__/`, `.ipynb_checkpoints/` já estão no `.gitignore`.
- `README.md` existe com 1 linha (`# Entrega-final-Insper-AI`); será sobrescrito.
- `Aprensentação + Paper/` não rastreada (o nome tem erro de digitação, "Aprensentação"; o texto do pedido diz "Apresentação"). Não mexo.
- Sem arquivo grande novo a evitar além dos já rastreados (maior: `parte_b_caracteristicas_A3_p0-7.npz`, 39 MB, já no git; não mexer).

**Acrescentar ao `.gitignore`:** `logs/`, `revisao/status.txt`, `*.pyc`, `.DS_Store`, `Thumbs.db`. (`.venv/`, caches e `.ipynb_checkpoints/` já existem.)

## 4. Estrutura-alvo (conforme a decisão de escopo)

```
README.md              (novo conteúdo: mapa, reprodução)
CLAUDE.md              (fica; ganha seção de estrutura + linha de log)
*.py *.sh *.ipynb      (ficam na raiz)
congelado_*.json       (ficam)
docs/                  (novo: 8 .md de apoio)
arquivo/ resultados/ figuras/ modelos/ logs/ revisao/ code_test/
Aprensentação + Paper/ (intocada)
```

## 5. Tabela de movimentos

| Arquivo | Destino | O que atualizar | Classe |
|---|---|---|---|
| `README.md` | (fica) reescrito | — | SEGURO |
| `.gitignore` | (fica) 5 linhas acrescentadas | — | SEGURO |
| `RELATORIO_NOITE.md` | `docs/` | nomes no CLAUDE.md; menções entre .md ficam válidas | RISCO (só texto, baixo) |
| `RELATORIO_TESTES.md` | `docs/` | CLAUDE.md; `projeto_ecg.ipynb` (1 menção em célula markdown, troca de string, sem tocar em saídas) | RISCO (só texto; notebook) |
| `decisoes_pendentes.md` | `docs/` | CLAUDE.md | RISCO (só texto, baixo) |
| `fundamentacao_medica.md` | `docs/` | CLAUDE.md | RISCO (só texto, baixo) |
| `literatura_modelos_abertos.md` | `docs/` | CLAUDE.md | RISCO (só texto, baixo) |
| `parte_a_rascunho.md` | `docs/` | CLAUDE.md | RISCO (só texto, baixo) |
| `parte_b_rascunho.md` | `docs/` | CLAUDE.md | RISCO (só texto, baixo) |
| `roteiro_slides.md` | `docs/` | CLAUDE.md | RISCO (só texto, baixo) |
| `revisao/*` | fica | cita nomes antigos; é registro histórico, não edito | — |
| `noite_parcial.py` (único arquivo sem nenhuma referência no repo) | fica (regra: não mover `.py`) | — | NÃO MOVER |
| `projeto_arritmia.ipynb`, `entrega_final.pdf`, `teste_viabilidade_code15_colab.ipynb` | ficam | têm menções em CLAUDE.md/outros | NÃO MOVER |
| todos os `.py`, `.sh`, `.ipynb`, `congelado_*.json`, modelos, CSVs de teste, `congelar.py`, `avaliar_externo.py`, `Aprensentação + Paper/` | ficam | — | PROTEGIDO / fora de escopo |

Nada em "arquivo/" por ora: não achei item claramente obsoleto e sem nenhuma referência que não seja `.py`/notebook. Se quiser mover `noite_parcial.py` (zero referências, mas é script), é decisão sua.

Nenhum movimento é 100% "SEGURO" no sentido de zero referência, exceto README/.gitignore. Os 8 `.md` são risco baixo: referências só em texto, sem código lendo.

## 6. Plano da Fase 2 (resumo)

1. `git pull`; branch `organizacao-repo` a partir da main.
2. Lote 1: `.gitignore` e `README.md`. Verificar `git status`.
3. Lote 2: `git mv` dos 8 .md para `docs/`; sed do prefixo `docs/` em CLAUDE.md e 1 string no notebook; checar nenhum link relativo quebrado nos .md movidos. Verificar imports (`python -c "import ecg_dados, a1_cnn, ..."`), JSON do notebook válido, caminhos lidos existem.
4. Lote 3: CLAUDE.md (seção de estrutura + linha de log).
5. Commits com caminhos explícitos, só na branch. Sem push.

Desfazer tudo: `git checkout main` (a branch pode ser apagada depois pelo Tom).
