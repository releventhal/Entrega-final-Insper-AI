# PROJETO: O que o ECG diz sobre o paciente, avaliado do jeito honesto

> **Para qualquer IA (ou pessoa) que abrir este repositório:** este é o documento único de contexto do projeto. **Leia inteiro antes de responder ou escrever código.** Ele existe para que duas duplas de pessoa+IA (Tom e Renata, cada um com a sua IA) trabalhem no MESMO plano, sem uma depender da conversa da outra. Tudo o que foi decidido, verificado ou aprendido está aqui. Se algo aqui conflitar com o que o usuário pedir na hora, **avise o conflito e pergunte**; não mude o plano em silêncio.

---

## 0. Comece aqui

### Primeira vez neste computador? Faça isto ANTES de qualquer tarefa
A IA pode rodar estes comandos ela mesma (pedindo permissão), ou passar para a pessoa rodar. Conferir cada item; se já estiver feito, pular.
1. **Código atualizado:** `git pull` (ou `git clone` se o repositório ainda não existir aqui). **Clonar em um caminho curto e fora de pastas sincronizadas (OneDrive etc.)**: os dados passam de 46 GB e o sincronizador pode tentar enviá-los para a nuvem (se há problema de caminho longo: **não verificado**).
2. **Ambiente:** se não existir a pasta `.venv/` na raiz, criar com Python 3.12 (`python -m venv .venv`) e instalar `requirements.txt` **com o Python do `.venv`** (Windows: `.venv\Scripts\python -m pip install -r requirements.txt`; Mac/Linux: `.venv/bin/python -m pip install -r requirements.txt`). **Nunca** instalar no Python global ou no Anaconda base (regra 11). Outras versões de Python: não testadas. **O Python 3.11 falha** (`numpy==2.5.4` exige 3.12 ou mais; erro visto em 07/11). No Windows o launcher `py` pode não existir; uma forma de instalar o 3.12 é `winget install Python.Python.3.12`. Depois de instalar, criar o `.venv` com o caminho completo do `python.exe` 3.12, porque o `python` do PATH pode continuar sendo o 3.11 até abrir um terminal novo.
3. **Dados:** a pasta `code15/` **não vem pelo git**. Rodar o `projeto_ecg.ipynb` com o kernel `.venv`: a seção 2 baixa sozinha o `exams.csv` e a parte 0 (~2,7 GB, ~20 min no computador da Renata e ~6 min no do Tom, ~7 GB livres durante o download).
4. **Conferir:** depois de rodar, as saídas das seções 3-5 devem bater com as que já estão salvas no notebook (ex.: "exames: 345,779 | pacientes: 233,770" e "pacientes em mais de um grupo: 0").
4b. **Conferir o A0 (opcional, com as partes 0 a 3 baixadas):** rodar `a0_baselines.py` com o Python do `.venv`. Na validação, o A0b deve dar média macro de AUROC 0,705, AUPRC 0,046 e F1 0,095, e o `resultados/experimentos.csv` deve ficar igual ao do repositório. Com menos partes, os números mudam.
5. **Registrar** no log (seção 10) que o ambiente foi montado nesse computador e, na seção 6, as características dele (sistema, RAM, GPU).

### Antes de cada sessão de trabalho
`git pull` → ver a tabela de tarefas abaixo → marcar a tarefa como "em andamento (nome)" → commit + push dessa marcação → só então mexer no notebook (seção 7, coordenação).

### Ordem de leitura
1. Esta seção (estado atual e próximas tarefas).
2. Seção 2 (a ideia) e seção 3 (**por que** cada escolha foi feita). Sem a seção 3, as decisões parecem arbitrárias.
3. Seção 7 (como trabalhar) e seção 8 (regras de ouro) **antes de mexer em qualquer arquivo**.
4. O notebook `projeto_ecg.ipynb`, que já tem as seções 1-5 rodadas e comentadas.

### Estado em 07/11/2026 (parte do fim do dia 06/11, mais a atualização de 07/11 abaixo)
- **Projeto atual:** CODE-15% (ECG de 12 derivações, 233 mil pacientes brasileiros). Escolhido em 06/11 por Renata e Tom.
- **Pronto e rodado** em `projeto_ecg.ipynb` (só com a parte 0 dos sinais): configuração, download, exploração da tabela inteira, leitura dos sinais, split por paciente.
- **Rodado fora do notebook (07/11, Tom):** baselines A0 (`a0_baselines.py` → `resultados/experimentos.csv`), índice dos exames, lista de exames zerados e gerador em lotes (`ecg_dados.py`); ver seção 6, "Código fora do notebook". Ainda não estão na seção 6 do notebook.
- **CNN A1 e A2 rodados na validação (07/11, Tom; **provisório, a confirmar com a Renata**):** `a1_cnn.py`, só parte 0, resultados em `resultados/experimentos.csv`. Detalhes na tarefa T4 e na seção 5.3. **A3 (07/11, **provisório, a confirmar com a Renata**):** o A2 com as partes 0 a 3 (55.561 exames de treino, 8.158 de validação). O teste interno, o PTB-XL e o CODE-TEST **não foram abertos**.
- **Ainda não feito:** modelo da Parte B; teste interno, PTB-XL e CODE-TEST (só no fim, uma vez); testes externos; slides. Sem decisão da dupla sobre os 411 exames zerados (T1): nada foi descartado.
- **Dados:** baixados **só no computador da Renata** (pasta `code15/`, fora do git) (ver atualização de 07/11: o Tom também tem as partes 0 a 3). Quem clonar o repositório precisa baixar os seus (o notebook faz isso sozinho; ver seção 6).
- **Fase 1 (MIT-BIH, arritmia por batimento):** pausada e arquivada. Notebook `projeto_arritmia.ipynb` e plano antigo `arquivo/CLAUDE_fase1_arritmia.md`. **Não apagar.** Os resultados dela são a motivação da apresentação (seção 3.1).

### Atualização de 07/11 (Tom e Claude do Tom)
- **Decisão (Tom e Renata):** o **PTB-XL é o teste externo PRINCIPAL** e o **CODE-TEST é o SECUNDÁRIO**. Limitação declarada: o CODE-TEST não tem `patient_id` nem `exam_id`, então não dá para verificar se há sobreposição com a CODE-15% (**não verificado**; seção 4).
- **Computador do Tom:** ambiente montado (Python 3.12.10, `.venv`, clone em caminho curto fora do OneDrive), `exams.csv` e **partes 0 a 3** baixados e conferidos (tamanho e md5 dos zips contra o Zenodo; HDF5 com formato N × 4.096 × 12), CODE-TEST baixado e aberto. Hardware na seção 6. Treino: nada treinado ainda.
- **Ordem das derivações, medida:** CODE-15% = DI, DII, DIII, AVR, AVL, AVF; CODE-TEST = DI, DII, DIII, **AVL, AVF, AVR** (precisa reordenar). PTB-XL: **não verificado** (seção 4).
- **Pendências novas:** T8 (escala dos sinais entre as bases) e T9 (RAM: 4 partes em float32 ≈ 15,7 GB não cabem na memória do Computador do Tom).
- **T3 e T9 avançaram (07/11):** A0 rodado (macro na validação: AUROC 0,705, AUPRC 0,046, F1 0,095 para idade + sexo). Decidido pelo Tom: gerador em lotes lendo o HDF5 em float32, sem float16; leitura de ~62 ms por lote de 64 (~1 min por época só de leitura); tempo de treino real **não medido**. Detalhes nas tarefas T3 e T9 e na seção 6.

### Atualização de 08/11 (sessão autônoma do Claude do Tom; **tudo provisório, a confirmar com a Renata**)
- Detalhe completo em `RELATORIO_NOITE.md`; decisões abertas organizadas em `decisoes_pendentes.md`; roteiro da intermediária em `roteiro_slides.md`; figuras em `figuras/` (geradas por `figuras.py`). Só treino e validação foram usados: **o teste interno, o PTB-XL e o CODE-TEST não foram abertos**; `congelar.py` não foi rodado; o JSON candidato não mudou (SHA256 conferido antes e depois).
- **Parte B só com a tabela (`parte_b_tabular.py`, 08/11):** B0 (idade + sexo) e B1 (B0 + `nn_predicted_age` + diferença), horizonte principal 2 anos declarado antes de rodar. Validação: 18.711 elegíveis, 450 óbitos (4.858 saíram por censura). AUROC B0 0,794 (IC 0,775-0,814), B1 0,804 (0,785-0,822); diferença B1 − B0 com IC 0,003 a 0,016. AUPRC 0,082 vs 0,086 (diferença com IC que inclui zero). 1 ano e 3 anos (sensibilidade): AUROC 0,810 → 0,814 e 0,800 → 0,811. Por faixa de idade, só em 40-59 anos a diferença tem IC fora de zero. Unidade de `timey`: a página do Zenodo **não diz** (valores 0 a 7,7, compatíveis com anos; **não verificado**). Paradoxo de Simpson reproduzido só com treino + validação: 3,0% (ECG 8+ mais velho) vs 4,7% (8+ mais jovem) sem controle; 60-69 anos: 3,2% / 3,7% / 6,4%; bate com a seção 3.5.
- **Análise de erros do A3 na validação (`analise_erros_A3.py`, descritiva, não muda o modelo):** os erros não se concentram em poucos pacientes (ex.: 76 FN de 1dAVb em 72 pacientes); exames com 2+ alterações têm algum erro em 66% (85 exames) contra 4% nos sem alteração; as alterações quase não se confundem entre si (≤ 2,3%); 1dAVb perde metade dos positivos (sensibilidade 0,46; homens 0,38, mulheres 0,54); tirar os 21 exames de derivação parcialmente zerada ou os 56 de sinal idêntico a outro muda o macro em ≤ 0,002; multiplicar o sinal por 0,45 ou 2 muda as notas em no máximo 3,3e-5 (a normalização por exame anula fatores de escala).
- **Robustez do A3 entre seeds (08/11):** seed 43 (só a seed muda; modelo em `modelos/A3_seed43/`): macro AUROC 0,9899, AUPRC 0,704, F1 0,713 na validação (seed 42: 0,9892 / 0,691 / 0,709). **A seed 43 parou pelo limite de tempo** depois que o computador ficou parado ~3h45 durante a época 8 (causa **não verificada**); melhor época 7. O A2 tem 1 seed só, então a comparação A2 → A3 com o desvio entre seeds não é um teste. Seed 44 (early stopping normal, melhor época 5): 0,9899 / 0,696 / 0,717. **Média de 3 seeds (desvio):** AUROC 0,9897 (0,0004), AUPRC 0,697 (0,007), F1 0,713 (0,004). Nos 2.069 exames, o A2 fica abaixo das 3 seeds do A3 em AUROC e AUPRC e empata com a pior em F1. Seed 45 e **A4 (perda ponderada, implementado e conferido, fora das aulas) não foram treinados**.
- **Git:** nesta sessão o Tom pediu por escrito que a IA fizesse commit ao fim de cada bloco e push (só sem commits novos no `origin`, nunca `--force`). A regra geral da seção 7 continua valendo fora desse pedido.

### Decisões da dupla de 08/11 (Tom e Renata; registradas pelo Claude do Tom, que só registra)
- **Modelo:** o **A3 com seed 42** (`modelos/A3_melhor.keras`), sem trocar por nenhuma extensão. A4 e as outras seeds ficam de fora do teste.
- **PTB-XL, limite de `likelihood`:** `presente` (código em `scp_codes` com qualquer valor).
- **PTB-XL, AFLT/IRBBB/ILBBB:** exclusão **ligada**, só da avaliação do rótulo correspondente (AFLT→AF, IRBBB→RBBB, ILBBB→LBBB).
- **PTB-XL, registros:** **todos os 21.799**; bootstrap por `patient_id`.
- **Exames zerados:** excluir **só os com as 12 derivações zeradas**, nos **três testes** (interno, PTB-XL, CODE-TEST), decidido **pelo sinal**, nunca pelos rótulos. Os parciais ficam.
- Tudo isso está em `congelado_candidato_A3.json` (seção `decisoes` e campos `*_estado` = "decidido"). **Status continua "candidato"**; `congelar.py` não foi rodado e nenhuma avaliação foi feita. **Lacuna de código:** `avaliar_externo.py` só aplica a exclusão dos exames todo-zero no caminho `interno`; em `ptbxl` e `code_test` ela **ainda não está implementada** (conferido no código; no PTB-XL medido são 0 registros com as 12 derivações zeradas; no CODE-TEST a contagem de todo-zero **não foi medida**, só "14 com alguma derivação zerada"). Implementar antes de congelar.

### Próximas tarefas (atualizar o status ao começar e ao terminar cada uma)
| # | Tarefa | Depende de | Status | Quem |
|---|---|---|---|---|
| T1 | Decidir as questões em aberto da seção 5.6 | — | **Pendente (dupla)** | Tom + Renata |
| T2 | Baixar mais partes dos sinais (proposta: 1-3) | T1 (quantas partes) | **Tom: feito em 07/11 (partes 1-3, md5 e HDF5 conferidos).** Renata: pendente | cada um no seu computador |
| T3 | Seção 6 do notebook: baselines A0 (sempre negativo; idade + sexo) | — | **A0 rodado em 07/11 fora do notebook** (`a0_baselines.py` → `resultados/experimentos.csv`; macro na validação: AUROC 0,705, AUPRC 0,046, F1 0,095). Falta passar para a seção 6 do notebook | a definir |
| T4 | Seção 6: CNN A1, função de métricas multi-rótulo, limiar na validação | T1 (pré-processamento) | **A1 rodado em 07/11 fora do notebook (**provisório, a confirmar com a Renata**):** `a1_cnn.py`; só parte 0; descarta só os exames todo-zero de treino e validação (27 + 4); sem normalização; limiar = F1 máximo na validação. Macro na validação: AUROC 0,976, AUPRC 0,656, F1 0,696 (20,3 min de treino). **A2 provisório** (normalização por exame e derivação): AUROC 0,985, AUPRC 0,682, F1 0,717 (17,6 min). **A3 provisório** (A2 com as partes 0 a 3): macro AUROC 0,989, AUPRC 0,691, F1 0,709 na validação de 8.158 exames (37,5 min de treino). Falta: decisões de T1 com a Renata, passar para a seção 6 do notebook, várias seeds | a definir |
| T5 | Seção 7: carregar PTB-XL (**principal**; reamostrar, mapear rótulos) e CODE-TEST (**secundário**; reordenar AVL/AVF/AVR, seção 4); testar o modelo escolhido | T4, T8 | **Preparo do PTB-XL feito em 07/11:** baixado e conferido; `preprocessar_ptbxl` e `GeradorPTBXL` em `ecg_dados.py` (500 → 400 Hz, 48 zeros de cada lado, ordem já igual à CODE, sem normalização). **Em 08/11 (**provisório, a confirmar com a Renata**):** carregadores de `interno`, `ptbxl` e `code_test` implementados em `avaliar_externo.py` e testados só por formato e contagem; nenhuma avaliação feita. **Decisões de 08/11 tomadas (seção 0):** limite `presente`, exclusão ligada, 21.799 registros, bootstrap por `patient_id`, exames todo-zero nos três testes. **Falta:** implementar a exclusão todo-zero em `ptbxl` e `code_test` no `avaliar_externo.py`, congelar (`congelar.py`) e fazer o teste em si. `avaliar_externo.py` e `congelado_candidato_A3.json` existem, mas o status é "candidato" e a trava impede o teste | a definir |
| T6 | Slides da intermediária (11/11), com a fase 1 como motivação | T4 (T5 se der) | Pendente | a definir |
| T7 | Parte B (seção 8 do notebook): B0-B3 | T4 | **B0 e B1 (só tabela) rodados em 08/11 fora do notebook** (`parte_b_tabular.py`, **provisório, a confirmar com a Renata**; horizonte 2 anos; validação: AUROC 0,794 vs 0,804). B2/B3 (CNN com sinal): pendentes | a definir |
| T8 | Comparar a escala (amplitude) dos sinais da CODE-15%, do CODE-TEST e do PTB-XL antes de qualquer teste externo (seção 4) | — | **Medido em 07/11 (seção 4):** PTB-XL com amplitudes menores (~0,4-0,5× por exame) e CODE com cauda pesada/offsets; unidade da CODE **não verificada**. **Falta decidir** se e como normalizar (dupla) | a definir |
| T9 | Definir como alimentar a CNN sem carregar tudo na RAM (ex.: gerador lendo o HDF5 em lotes) | T1, T4 | **Decidido (Tom, 07/11): gerador em lotes lendo o HDF5 em float32, sem converter para float16.** Pronto em `ecg_dados.py` (índice, lista de zerados, gerador); falta usá-lo no treino (T4). Leitura medida: ~62 ms por lote de 64 (~1 min por época só de leitura); tempo de treino real não medido | a definir |

---

## 1. Contexto

- **Quem:** Tom (Tomaz Santos Almeida) e Renata, estudantes do Insper, Trilha de Trainees da liga **Insper AI**.
- **O quê:** projeto final de **deep learning** em uma área (saúde, segurança ou mobilidade; escolhemos **saúde**), conforme `entrega_final.pdf`. Objetivo: achar uma boa ideia, planejar, **justificar as escolhas** e fazer até o fim. Entrega final = **notebook + paper**; **a nota vem da apresentação**. Os "cases de sucesso" do PDF são só inspiração e não podem ser copiados.
- **Prazos:** intermediária **11/11** (o PDF diz 12/11, mas a nossa é dia 11), entrega **17/11**, apresentação **18/11** (membros, diretores e trainees).
- **Feedback dos diretores (05/11):** a fase 1 tinha **dados demais de poucos pacientes** (44) para um modelo robusto. Querem um projeto **robusto e mais complexo**. É por isso que a base mudou e que técnicas fora das aulas estão liberadas (regra 5).
- **Aulas vistas:** regressão linear (A1), escalando o modelo (A2), classificação (A3), neurônio → rede (A4), MLP + backprop (A5), treino na prática: época, batch, otimizadores, curvas de loss (A6), métricas, matriz de confusão, erros, regularização (A7), Claude Code (A8), CNNs (A9, A10), transfer learning (A10). **Não vistos:** RNN/LSTM, linguagem, aprendizado por reforço, Grad-CAM, análise de sobrevivência.
- **Experiência prévia da dupla:** MLP em Keras no MNIST; projeto de carros usados (EDA, regressão, classificação).
- **Regra da trilha (Aula 8):** pode usar IA para escrever código; **não dá para delegar a decisão**. Arquitetura, métricas, divisão dos dados e interpretação são da dupla, que precisa entender e explicar tudo.

---

## 2. A ideia

Usar ECGs de 12 derivações de **233.770 pacientes brasileiros** (CODE-15%, Rede de Telessaúde de Minas Gerais) em duas partes:

- **Parte A (núcleo, até a intermediária):** uma CNN 1D detecta **6 alterações** no ECG (multi-rótulo). Avaliada (1) em pacientes que nunca viu (split por paciente), (2) **em outro país** (PTB-XL, Alemanha) e (3) contra um **gabarito de especialistas** (CODE-TEST).
- **Parte B (pergunta principal do paper final):** **o ECG prevê morte em até X anos além do que a idade e o sexo já dizem?** Comparar a CNN com um modelo só de idade + sexo e com a "idade do ECG" do artigo de Lima et al. (2021).

### Perguntas de pesquisa
1. Um modelo treinado na atenção primária de Minas Gerais funciona em pacientes novos, em outro país e contra o gabarito de especialistas? Onde ele falha?
2. O ECG traz informação sobre risco de morte que a idade e o sexo não trazem? Quanto?

### Narrativa da apresentação
"Começamos com uma base pequena (MIT-BIH) e mostramos que, com poucos pacientes, o modelo decora as pessoas: F1 macro de 0,89 com split aleatório e de 0,39 em pacientes novos. Por isso migramos para 233 mil pacientes, mantendo a mesma exigência: avaliar sempre em pessoas, lugares e perguntas que o modelo nunca viu."

### Relação com a literatura (não copiar; diferenciar)
- **Ribeiro et al., Nat. Commun. 2020** ("Automatic diagnosis of the 12-lead ECG using a deep neural network"): mesmo grupo de Minas, mesmas 6 alterações, CODE-TEST como teste. A Parte A **reproduz em menor escala**. O diferencial é o teste externo no PTB-XL e a análise honesta dos erros.
- **Lima et al., Nat. Commun. 2021** ("Deep neural network-estimated electrocardiographic age as a mortality predictor", DOI 10.1038/s41467-021-25351-7): previu a **idade** pelo ECG com uma ResNet 1D treinada nos outros 85% da CODE (~1,3 milhão de pacientes, 70 épocas, GPU NVIDIA Titan V). **A CODE-15% foi o conjunto de teste deles.** MAE de 8,38 anos na CODE-15%. "Idade do ECG" mais de 8 anos acima da real → HR ~1,79 de morte (ajustado por idade e sexo). **Eles não treinaram um modelo para prever morte direto do ECG**: a morte só foi usada para validar a idade. A Parte B entra aí. Código deles: github.com/antonior92/ecg-age-prediction.
- Pelo que o Claude da Renata lembra (**não verificado**), já existe trabalho americano prevendo morte em 1 ano direto do ECG, com outra base. A Parte B não é inédita no mundo, mas é nova nesta base e na comparação com a idade do ECG.

---

## 3. Por que cada escolha foi feita (linhas de raciocínio)

### 3.1 Por que saímos do MIT-BIH (fase 1)
- O MIT-BIH tem 48 gravações de 47 pacientes. Na fase 1, a mesma MLP teve F1 macro de 0,893 com split aleatório e de 0,385 com split por paciente. O recall da classe S caiu de 0,752 para 0,046, e a perda de validação piorava já depois da 1ª época, porque o modelo decorava os pacientes do treino.
- A análise por paciente mostrou que 75% dos batimentos S do teste vinham de **uma** gravação (232). Ou seja, o resultado dependia de uma pessoa.
- Os diretores concluíram que eram poucos pacientes para um modelo robusto. Concordamos. A lição ("avaliar por paciente, desconfiar de número bonito") foi mantida.

### 3.2 Bases consideradas e descartadas
- **Competição Kaggle "RSNA Knee Abnormality Detection":** já encerrada (prazo final 22/10/2026); ressonância 3D (~800 mil arquivos DICOM); os rótulos precisariam ser extraídos de laudos em 9-12 idiomas (processamento de linguagem, não visto). Inviável em 12 dias.
- **MIMIC (UTI):** exige curso e credenciamento que podem levar dias ou semanas.
- **Raio-X de tórax (NIH/CheXpert) e pele (ISIC):** viáveis, mas são temas muito explorados e com rótulos ruidosos (raio-X). Ficaram como reserva.
- **MIT-BIH como teste externo do modelo da CODE-15%:** **não serve.** Ele tem só 2 derivações (o modelo usa 12), gravações de 30 min com rótulo por batimento (a CODE tem exames de 7-10 s com rótulo por exame), e as 6 alterações não existem lá do mesmo jeito. O teste externo certo é o PTB-XL.

### 3.3 Por que a CODE-15%
- Muitos pacientes (233.770), acesso aberto e imediato (CC BY 4.0), dados brasileiros, rótulos prontos, `patient_id` (permite split por paciente) e dados de óbito com tempo de acompanhamento.
- Aproveita o que a dupla aprendeu sobre ECG na fase 1 (inclusive bloqueios de ramo, que causaram erros lá e aqui são classes).
- Um modelo pequeno treina no computador da Renata (~1 min por época com 16 mil exames), então não dependemos de GPU.

### 3.4 Por que a Parte A vem antes da Parte B
- A Parte A tem resultado quase garantido: no teste de viabilidade, 2 épocas já deram AUROC ~0,98-0,99 em 4 das 6 alterações. Ela monta todo o "encanamento" (dados, split, treino, métricas) que a Parte B reaproveita.
- A Parte B é a pergunta mais interessante, mas o resultado é incerto: óbitos são raros e boa parte do sinal pode ser só a idade. **"O ECG acrescenta pouco além da idade" também é um resultado válido**, se bem medido.

### 3.5 Por que a Parte B compara com idade + sexo (e a lição do paradoxo de Simpson)
- O óbito depende muito da idade: 0,5% antes dos 40 e 13,9% aos 80+. Um modelo que só "adivinha a idade" pelo ECG já pareceria bom. Por isso a pergunta certa é **quanto o ECG acrescenta ao B0 (idade + sexo)**.
- Na exploração apareceu um paradoxo de Simpson. **Comparando direto**, quem tem o ECG "8+ anos mais velho" morre **menos** (3,0%) do que quem tem o ECG "8+ anos mais jovem" (4,6%), o contrário do artigo. Isso acontece porque os grupos têm idades reais muito diferentes (média de 42 vs 63 anos): o modelo de idade provavelmente "puxa para o meio". **Dentro de cada faixa de idade**, o resultado do artigo aparece em todas as faixas (60-69 anos: 3,0% / 3,6% / 6,5%). Lição: **toda comparação de risco tem que levar a idade em conta**. É um ótimo exemplo para a apresentação.

### 3.6 Outras escolhas
- **Split sorteado sobre a lista completa de pacientes do `exams.csv`, e não sobre os exames carregados:** assim, adicionar partes depois não muda o grupo de ninguém, e resultados com 1 ou 4 partes ficam comparáveis.
- **Só o primeiro exame de cada paciente na Parte B:** é o único que tem `death`/`timey` (seção 4).
- **Regra do horizonte:** para "morreu em até X anos", só entra quem morreu dentro do prazo ou foi acompanhado o prazo inteiro. Alguém vivo e acompanhado por só 1 ano não diz nada sobre 3 anos.
- **`nn_predicted_age` como baseline B1:** é (provavelmente) a saída do modelo de Lima et al., treinado nos outros 85% da CODE. Na CODE-15% essas previsões são "fora da amostra", então servem como comparação justa com o "estado da arte".
- **CODE-TEST como gabarito:** os rótulos da CODE-15% vêm da rotina de laudos (origem exata **não verificada**) e podem ter erros. O CODE-TEST foi revisado por vários médicos.
- **float16 para os sinais:** metade da memória; perda de precisão irrelevante para ECG; mesmo em float16, 4 partes ≈ 7,9 GB; ver T9. **Vale só para o `carregar_partes` do notebook (parte 0).** Para o treino com 4 partes, a decisão de 07/11 é o `GeradorECG` (`ecg_dados.py`), que lê o HDF5 em float32 em lotes, sem converter para float16.
- **CNN pequena primeiro:** roda no computador da Renata. Modelos maiores (ResNet como nos artigos) ficam como extensão, com GPU.

---

## 4. Dados (tudo conferido nos arquivos oficiais em 05-06/11)

### CODE-15% (treino e teste interno)
- Zenodo **10.5281/zenodo.4916206**, **CC BY 4.0**. 345.779 exames, 233.770 pacientes, 18 partes `exams_partN.zip` (N = 0-17; ~2,7 GB cada zip, ~3,9 GB cada `.hdf5`; 46,3 GB no total) + `exams.csv` (35 MB).
- **`exams.csv`**: `exam_id, age, is_male, nn_predicted_age, 1dAVb, RBBB, LBBB, SB, ST, AF, patient_id, death, timey, normal_ecg, trace_file` (`trace_file` diz em que parte está o sinal).
- **Rótulos** (exames positivos): 1dAVb 5.716 · RBBB 9.672 · LBBB 6.026 · SB 5.605 · ST 7.584 · AF 7.033 · `normal_ecg` 134.657 (38,9%). **Multi-rótulo:** 89% dos exames não têm nenhuma das 6; ~1% tem duas ou mais.
- **HDF5:** `tracings` (N × 4.096 × 12, float32, 400 Hz, valores em mV aparentemente) e `exam_id`. Derivações nesta ordem: **DI, DII, DIII, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6**. **Medido** em 07/11 (Computador do Tom): em 2.997 exames sorteados da parte 0 (seed 42), as colunas 3, 4 e 5 têm correlação 1,000 com aVR = −(DI+DII)/2, aVL = (DI−DIII)/2 e aVF = (DII+DIII)/2, nessa ordem (2.997 de 2.997 exames; erro mediano 0,001-0,002). Em 361-538 exames há alguma amostra isolada com erro > 0,05 (até 18,7); causa **não investigada**.
  - Durações: 7,3 s (2.934 amostras; 58% dos exames da parte 0), 10,2 s (4.096; 38%) e 9,7 s (raros). **Os exames curtos são centralizados, com zeros dos dois lados** (581 amostras no começo e 581 no fim).
  - Cada parte tem 1 linha extra com `exam_id` = 0 (preenchimento). Na parte 0, 38 de 20.000 exames são todos zero e 61 têm alguma derivação zerada (estes 61 ainda **não** são descartados; decidir). **Medido em 07/11 nas partes 0 a 3** (passada única de 44 s; lista em `code15/exames_zerados_p0-3.npy`, fora do git): 411 exames com alguma derivação zerada, sendo 172 com as 12 derivações zeradas e 239 parciais (196 com 1 derivação, 15 com 2, 4 com 3, 18 com 5, 5 com 6, 1 com 9). Por parte: 99, 108, 124, 80. Por grupo: treino 294, validação 43, teste 74. **Listados e NÃO descartados** (decisão T1 da dupla). Causa das derivações zeradas parciais: **não investigada**.
  - **Memória (cálculo, 07/11):** a parte 0 tem 20.001 × 4.096 × 12 valores = ~3,93 GB em float32 (arquivo `.hdf5` de 3.932.518.664 bytes) e ~2 GB em float16 (medido no notebook). 4 partes = ~15,7 GB em float32 e ~7,9 GB em float16 (cálculo, não medido). Os 15,7 GB em float32 **não cabem na RAM do Computador do Tom** (15,7 GB no total); ver T9.
- **As partes NÃO são separadas por paciente.** 29% dos pacientes têm mais de um exame (até 38) e somam 52% dos exames; metade dos pacientes da parte 0 aparece em outras partes. **O split é feito por `patient_id`.**
- **Óbito:** `death` e `timey` (anos de acompanhamento a partir do exame) só estão preenchidos em **um exame por paciente** (233.647), que é o **primeiro exame** (em 99,3% dos casos, o de menor idade). 8.341 óbitos (3,6%). `timey`: mediana 3,5 anos (p5 1,0; p95 6,9).
- **Horizontes possíveis:** 1 ano → 224.869 pacientes, 2.812 óbitos (1,3%); 2 anos → 185.106, 4.716 (2,5%); 3 anos → 143.779, 6.173 (4,3%).
- **`nn_predicted_age`:** MAE de 8,53 anos (r = 0,83) contra a idade real, quase o número do artigo (8,38; r = 0,84). **Muito provavelmente** é a saída do modelo de Lima et al. (não documentado).
- **Idade e sexo:** 17 a 100 anos (média 53), 40% homens. Bloqueios de ramo, bloqueio AV e FA crescem muito com a idade (FA: ~0,1% antes dos 30, ~7% aos 80+); SB e ST quase não mudam.
- **Download:** o Zenodo limita ~0,3 MB/s por conexão. Com 16 conexões em paralelo (função `baixar` no notebook), ~2 MB/s, ~20 min por parte no computador da Renata e ~6-7 min no do Tom (partes 0 a 3, 07/11). **Atenção:** uma versão antiga do downloader gerou um arquivo truncado silenciosamente; a versão atual confere o tamanho de cada pedaço e o total.

### PTB-XL (teste externo, Alemanha)
- PhysioNet `ptb-xl/1.0.3`. 21.799 ECGs, 18.869 pacientes, 10 folds (`strat_fold`), 500 Hz (pasta `records500`) ou 100 Hz, **10 s**, **mesmas 12 derivações na mesma ordem** (I, II, III, AVR, AVL, AVF, V1-V6, ordem registrada em 05/11), em mV. **Ordem medida em 07/11** (1.000 registros sorteados, seed 42, `records500`): `sig_name` do cabeçalho é I, II, III, AVR, AVL, AVF, V1-V6, e as fórmulas de Einthoven confirmam: as colunas 3, 4 e 5 têm correlação > 0,99 com −(DI+DII)/2, (DI−DIII)/2 e (DII+DIII)/2, nessa ordem, em 1.000 de 1.000 registros (erro mediano 0,0003 mV); DIII = DII − DI com erro mediano 0,0001. **A página do PhysioNet, que diz "AVL, AVR", está trocada em relação aos arquivos.** A permutação para chegar a DI, DII, DIII, AVR, AVL, AVF, V1-V6 é a **identidade** (nenhuma reordenação). Cabeçalho: unidade mV, 500 Hz, (5000, 12), sem NaN.
- Metadados: `ptbxl_database.csv` (coluna `scp_codes`, um dicionário em texto; usar `ast.literal_eval`) e `scp_statements.csv`.
- **CODE-TEST, contagem de rótulos do `gold_standard.csv` (08/11):** 827 exames; 1dAVb 28, RBBB 34, LBBB 30, SB 16, ST 37, **AF 13**.
- **Idade e sexo (medido em 07/11, nível registro, para comparar com a CODE-15%):** 21.799 registros, 18.869 pacientes distintos (2.111 com mais de um registro; máximo 10). Idade: média 59,5 e mediana 61 anos sem os 293 registros com idade 300 (código de privacidade para > 89 anos); percentis 5 e 95: 27 e 84. Sexo: 52,1% homens (0 = homem: 11.354; 1 = mulher: 10.445). CODE-15% (exames): 345.779, 233.770 pacientes, média 53,2 e mediana 54 anos, 40,3% homens. Faixa etária, % dos registros (PTB-XL / CODE-15%): < 30: 6,3 / 15,7; 30-39: 6,7 / 13,0; 40-49: 11,3 / 14,0; 50-59: 20,4 / 14,9; 60-69: 24,1 / 15,9; 70-79: 18,8 / 16,6; 80-89: 11,1 / 9,8; 90 ou mais (PTB-XL inclui o 300): 1,3 / 0,02. O PTB-XL é mais velho e tem mais homens que a CODE-15%.
- Leitura testada: `wfdb.rdrecord('00001_hr', pn_dir='ptb-xl/1.0.3/records500/00000')` → (5000, 12). **O `pn_dir` precisa incluir a subpasta**; passar o caminho inteiro no nome do registro dá erro 404.
- **Equivalência de rótulos** (códigos SCP): 1dAVb ↔ `1AVB` (793), RBBB ↔ `CRBBB` (541), LBBB ↔ `CLBBB` (536), SB ↔ `SBRAD` (637), ST ↔ `STACH` (826), AF ↔ `AFIB` (1.514); NORM 9.514. **A decidir:** se `AFLT` (flutter) e `IRBBB`/`ILBBB` (bloqueios incompletos) contam como positivos, negativos ou ficam fora; e se usamos a confiança (`likelihood`) dos códigos.
  - **Contagem de positivos por limite de `likelihood` (medida em 07/11, só contagem; o limite NÃO foi escolhido):** código presente com qualquer valor / `> 0` / `>= 50` / `= 100`: 1dAVb 793 / 793 / 790 / 779; RBBB 541 / 541 / 540 / 536; LBBB 536 / 536 / 536 / 535; **SB 637 / 0 / 0 / 0; ST 826 / 4 / 4 / 1; AF 1.514 / 48 / 48 / 48.** Os códigos de ritmo (SB, ST, AF) vêm quase todos com `likelihood` 0 (SB: 637 de 637; ST: 822 de 826; AF: 1.466 de 1.514), então um limite `> 0`, `>= 50` ou `= 100` **apaga quase todos os positivos desses três rótulos**. Isso precisa entrar na decisão do limite.
  - **O que a documentação diz sobre `likelihood` (lido em 07/11):** página do PhysioNet (`ptb-xl/1.0.3`): "scp_codes (SCP-ECG statements as a dictionary with entries of the form statement: likelihood, where likelihood is set to 0 if unknown)". Artigo dos dados (Wagner et al., Scientific Data 2020, seção de métodos, nature.com/articles/s41597-020-0495-6): "The likelihood ranges from 0 to 100 conveying the certainty the cardiologist (...) is about a statement. For form and rhythm statements or in cases where no likelihood information was available, the corresponding likelihood was set to zero." Em `scp_statements.csv`, SBRAD, STACH, AFIB e AFLT são `rhythm = 1` (sem `diagnostic` nem `form`); 1AVB, CRBBB, CLBBB, IRBBB e ILBBB são `diagnostic = 1`. **Medido no `ptbxl_database.csv` inteiro:** entradas com `likelihood` 0 são 97,9% das de ritmo (20.731 de 21.174), 88,3% das de forma (8.048 de 9.112) e 0,7% das diagnósticas (213 de 30.721). Ou seja, para ritmo, `likelihood` 0 significa "sem informação de verossimilhança" (não "improvável"). **Não explicado pela documentação:** alguns statements de ritmo têm `likelihood` > 0 (AFIB 48, AFLT 56, STACH 4, PACE 287, BIGU 24, PSVT 24); o motivo é **não verificado**. Contagem de positivos de SB, ST e AF por limite (registros), `>= 0` (código presente) / `> 0` / `>= 50` / `= 100`: SB 637 / 0 / 0 / 0; ST 826 / 4 / 4 / 1; AF 1.514 / 48 / 48 / 48. **Critério proposto só pela documentação, NÃO escolhido (decisão da dupla):** positivo = o código aparece em `scp_codes` com qualquer valor (`>= 0`) para os 6 rótulos, porque nos rótulos de ritmo (SB, ST, AF) o 0 não carrega informação; para 1dAVb, RBBB e LBBB (diagnósticos) a documentação não prescreve limite e as contagens quase não mudam (793 / 790 / 779 em 1dAVb). Ressalva: código ausente significa "não declarado", e se isso equivale a "negativo confirmado" nos rótulos de ritmo é **não verificado**.
  - **Contagens com os padrões provisórios (08/11, **provisório, a confirmar com a Renata**; limite `presente`, exclusão AFLT/IRBBB/ILBBB ligada; só contagem de rótulos, medida pelo código de `avaliar_externo.py`):** positivos (avaliados, excluídos): 1dAVb 793 (21.799, 0); RBBB 540 (20.681, 1.118; sem a exclusão seriam 541); LBBB 536 (21.722, 77); SB 637 (21.799, 0); ST 826 (21.799, 0); AF 1.497 (21.726, 73; sem a exclusão seriam 1.514). Registros com algum dos três códigos: 1.264. **Sinal bruto de 500 Hz (os 21.799 registros lidos, 08/11):** 0 com NaN/inf, 0 com as 12 derivações zeradas, 1 com alguma derivação zerada (a V5; também é o único com derivação constante).
  - **`AFLT`, `IRBBB`, `ILBBB` (proposta: fora da avaliação):** AFLT 73 registros (56 com `likelihood` > 0), IRBBB 1.118 (1.118), ILBBB 77 (77). Registros com pelo menos um dos três: 1.264 (1.250 com `likelihood` > 0); desses, 51 também têm um dos 6 rótulos com `likelihood` > 0.
  - **Quem não tem nenhum dos 6 (`likelihood` > 0):** 20.042 de 21.799 registros; com algum dos 6: 1.757.
- **Para usar no modelo:** reamostrar de 500 para 400 Hz (ex.: `scipy.signal.resample_poly(x, 4, 5)` → 4.000 amostras) e **centralizar com zeros** até 4.096 (48 de cada lado), igual à CODE. A documentação do PTB-XL diz 1 µV por LSB e o cabeçalho dos arquivos diz mV. **Escala medida em 07/11 (T8), sem nenhuma normalização:** 1.000 registros do PTB-XL, 998 exames da parte 0 da CODE-15% (sorteio com seed 42; 2 exames todo-zero ignorados só nesta estatística) e os 827 do CODE-TEST, calculado só na janela com sinal (sem o preenchimento de zeros). Empilhando todas as amostras, a razão PTB-XL / CODE-15% é 0,23 (mediana de |x|), 0,14 (desvio padrão) e 0,145 (percentil 99 de |x|), medianas entre as 12 derivações; CODE-TEST / CODE-15% é 0,62, 0,73 e 0,77. Esses números são puxados por exames com valores muito grandes. **Medindo exame a exame** (mediana entre os exames, derivação II): desvio padrão 0,132 (PTB-XL), 0,304 (CODE-15%), 0,268 (CODE-TEST); percentil 99 de |x| 0,598, 1,563 e 1,379; máximo de |x| por exame 1,88, 4,26 e 3,51. Exames com |x| > 5 em algum ponto: 2,1%, 39,0% e 21,5%; com |x| > 10: 0,4%, 12,4% e 6,7%. A mediana de |x| entre as derivações de um exame passa de 2 em 5% dos exames da CODE-15% (1,45 no CODE-TEST, 0,10 no PTB-XL), o que sugere **deslocamento de linha de base (offset)** grande na CODE. **Conclusão: não há fator de potência de 10 evidente; o PTB-XL tem amplitudes menores (cerca de 0,4 a 0,5 vezes a CODE-15% por exame) e a CODE tem cauda muito mais pesada (offsets e picos).** Se a CODE-15% está de fato em mV: **não verificado** (o README do CODE-TEST diz "scale 1e-4V" e, na frase seguinte, "multiplicar por 1000 para obter V", o que é contraditório). Diferença de população (idade, aparelho) como causa: **não verificado**. Nenhuma normalização foi aplicada.
- **Licença e tamanho (página do PhysioNet, 07/11):** Creative Commons Attribution 4.0 (CC BY 4.0). 3,0 GB descompactado (zip de 1,7 GB), somando `records100` e `records500`. **Baixado em 07/11 para `ptbxl/` (fora do git):** `records500` + `ptbxl_database.csv` + `scp_statements.csv` = 43.600 arquivos, 2.635.735.664 bytes (2,45 GiB); todos conferidos com o `SHA256SUMS.txt` oficial do PhysioNet (0 falhas depois de refazer 7 downloads que falharam).
- **Outros pontos da documentação (07/11):** `patient_id` existe e a coluna `strat_fold` já respeita pacientes (todos os registros de um paciente caem no mesmo fold); os autores sugerem folds 1-8 treino, 9 validação e 10 teste, e os folds 9 e 10 passaram por avaliação humana (rótulos de melhor qualidade). Idades acima de 89 aparecem como ~300 (HIPAA). As versões 1.0.2 e 1.0.3 removeram registros duplicados.

### CODE-TEST (gabarito)
- Zenodo **10.5281/zenodo.3765780**, **CC BY 4.0**, aberto, 218 MB (`data.zip`). 827 ECGs de pacientes diferentes, anotados por vários cardiologistas, residentes e estudantes, nas mesmas 6 classes. Companion code: github.com/antonior92/automatic-ecg-diagnosis.
- **Papel (decisão de 07/11): teste SECUNDÁRIO** (o principal é o PTB-XL).
- **Conteúdo do zip (aberto em 07/11 no Computador do Tom; tamanho de 218.189.452 bytes e md5 conferidos com o Zenodo):** `data/ecg_tracings.hdf5` (dataset `tracings`, 827 × 4.096 × 12, **float64**, 400 Hz); `data/attributes.csv` (só `age` e `sex`); `data/annotations/` com 7 CSVs de 6 colunas de rótulo `1dAVb, RBBB, LBBB, SB, AF, ST` (`gold_standard`, `cardiologist1`, `cardiologist2`, `cardiology_residents`, `emergency_residents`, `medical_students` e `dnn`, a previsão da rede de Ribeiro et al.); e o `README.md`. A linha *i* de cada CSV corresponde ao sinal *i*.
- **Ordem das derivações do CODE-TEST (medida): DI, DII, DIII, AVL, AVF, AVR, V1-V6**, como diz o README (correlação de 0,99 das colunas 3, 4 e 5 com aVL, aVF e aVR por Einthoven). É **diferente** da CODE-15% (AVR, AVL, AVF). Antes de testar, reordenar as colunas com o índice `[0,1,2,5,3,4,6,7,8,9,10,11]`.
- **Gabarito (`gold_standard`), exames positivos:** 1dAVb 28, RBBB 34, LBBB 30, SB 16, AF 13, ST 37; 681 exames sem nenhuma alteração e 12 com mais de uma. **AF tem só 13 positivos**, então as métricas de AF serão muito incertas. Idade de 17 a 97 (média 54,9); 321 homens e 506 mulheres.
- **Sinais:** 538 exames com 2.934 amostras com sinal (7,3 s), 253 com 4.096, 35 com 4.000 e 1 com 3.134, com zeros dos dois lados (o README fala em 2.800 amostras, mas o dado mede 2.934). 14 exames têm alguma derivação zerada. Sem NaN/inf e sem sinais duplicados. O README diz "escala 1e-4 V"; a escala real contra a CODE-15% é **não verificada** (T8).
- **Risco não verificado (mantido):** os pacientes do CODE-TEST podem estar também na CODE-15%. Se estiverem no nosso treino, o teste fica "contaminado". Em 07/11 se viu que o CODE-TEST **não tem identificador nenhum** (nem `patient_id` nem `exam_id`), então não dá para cruzar por id: **limitação declarada**. Comparar os sinais (hash ou correlação) só acharia o mesmo *exame*, não o mesmo *paciente*, e exigiria baixar as 18 partes (~46 GB); **não executado**.

---

## 5. Plano técnico

### 5.1 Cronograma
| Data | O quê |
|---|---|
| 06/11 | ✅ Plano novo, notebook novo, dados + exploração, split por paciente |
| 07/11 | T2 (mais partes) **feito no computador do Tom (partes 0 a 3)**; T1 (decisões) segue pendente; **T3 (baselines A0) rodado fora do notebook** (`a0_baselines.py`); T9 (gerador em lotes) pronto em `ecg_dados.py` |
| 08/11 | T4: CNN A1, métricas por rótulo, limiar escolhido na validação |
| 09/11 | T5: PTB-XL e CODE-TEST |
| 10/11 | Análise de erros + T6 (slides, com a fase 1 como motivação) |
| **11/11** | **Apresentação intermediária** |
| 12-14/11 | T7: Parte B (horizonte, B0-B3) |
| 15/11 | Robustez (várias seeds, subgrupos de idade e sexo) e extensões |
| 16/11 | Notebook e paper finais |
| **17/11 / 18/11** | **Entrega / apresentação** |

### 5.2 Divisão dos dados
- Cada `patient_id` da tabela inteira é sorteado uma vez: `np.random.default_rng(42).random(n)` sobre `np.sort(tabela["patient_id"].unique())`; < 0,7 → treino, < 0,8 → validação, resto → teste. **Proporção 70/10/20 provisória.**
- Na parte 0: treino 13.866 exames / validação 2.069 / teste 4.027; 0 pacientes em mais de um grupo (o notebook confere isso).
- **Validação** = escolher épocas, limiares e hiperparâmetros. **Teste interno, PTB-XL e CODE-TEST** = só no fim, nunca para escolher nada.

### 5.3 Parte A: 6 alterações (multi-rótulo)
- **Saída:** 6 neurônios **sigmoid** (um "sim/não" por alteração), perda `binary_crossentropy`. Não é softmax, porque um exame pode ter várias alterações.
- **A0:** (a) "sempre negativo" (mostra que acurácia engana) e (b) regressão logística só com idade + sexo (mostra quanto a idade sozinha já explica). A CNN precisa ganhar do A0b.
- **A1:** CNN 1D base, a mesma do teste de viabilidade: entrada 4.096 × 12 → 4 × [Conv1D (32, 64, 128, 128 filtros; kernel 15; `padding="same"`; ReLU) + MaxPool 4 nas três primeiras] → GlobalAveragePooling → Dense 6 sigmoid. ~406 mil parâmetros. Adam, batch 64. Usar early stopping na perda de validação (como na fase 1). Os dados entram pelo `GeradorECG` de `ecg_dados.py` (lotes de 64 lidos do HDF5 em float32; treino embaralhado com seed 42; validação em ordem fixa), não por um array carregado inteiro na RAM.
- **Resultados de 07/11 (**provisório, a confirmar com a Renata**; só validação da parte 0: 13.866 exames de treino e 2.069 de validação; seed 42, uma execução cada):** A1 AUROC por rótulo: 1dAVb 0,919, RBBB 0,997, LBBB 0,994, SB 0,992, ST 0,996, AF 0,959; macro AUROC 0,976, AUPRC 0,656, F1 0,696, sensibilidade 0,749, precisão 0,658, especificidade 0,993. O A0b no **mesmo** conjunto (idade + sexo): macro AUROC 0,697, AUPRC 0,046, F1 0,103 (no conjunto das partes 0 a 3 da seção 5.3, A0b: 0,705, 0,046, 0,095). **A2 provisório** (só a normalização muda): 1dAVb 0,952, RBBB 0,993, LBBB 0,994, SB 0,994, ST 0,996, AF 0,983; macro AUROC 0,985, AUPRC 0,682, F1 0,717. Intervalos de 95% do AUROC (bootstrap por paciente, 1.000 reamostragens, seed 42) com 28 a 48 positivos por rótulo: A1 1dAVb 0,884-0,951 e AF 0,920-0,982; A2 1dAVb 0,913-0,981 e AF 0,965-0,993. **Com uma seed e esses intervalos, não dá para afirmar que o A2 é melhor que o A1** (os intervalos se sobrepõem); não fiz um teste da diferença entre os dois. Tudo isto é da validação, que também escolheu épocas e limiares, então é otimista; o número honesto vem do teste (ainda fechado).
- **A3 (07/11, **provisório, a confirmar com a Renata**; uma mudança em relação ao A2: partes 0 a 3 no treino e na validação; uma seed, uma execução):** 55.561 exames de treino e 8.158 de validação (descartados só os todo-zero: 118 de 55.679 no treino e 22 de 8.180 na validação). **Validação completa (8.158 exames, NÃO comparável diretamente com A1/A2, que usaram só 2.069):** AUROC / AUPRC / F1 por alteração: 1dAVb 0,972 / 0,436 / 0,485; RBBB 0,995 / 0,851 / 0,826; LBBB 0,994 / 0,844 / 0,798; SB 0,993 / 0,618 / 0,658; ST 0,995 / 0,695 / 0,783; AF 0,986 / 0,700 / 0,701. Macro: AUROC 0,989, AUPRC 0,691, F1 0,709, sensibilidade 0,746, precisão 0,681, especificidade 0,993. A0b no mesmo conjunto de 8.158: macro 0,705 / 0,046 / 0,095. **Nos mesmos 2.069 exames de validação da parte 0 do A1/A2** (limiar escolhido na validação completa): macro AUROC 0,990, AUPRC 0,721, F1 0,717; com o limiar reescolhido nesses 2.069, como no A1/A2: F1 0,739 (sensibilidade 0,817, precisão 0,680). Por alteração nos 2.069 (AUROC): 1dAVb 0,969, RBBB 0,995, LBBB 0,996, SB 0,996, ST 0,996, AF 0,988. Comparação nos 2.069 (AUROC / AUPRC / F1 macro): A1 0,976 / 0,656 / 0,696; A2 0,985 / 0,682 / 0,717; A3 0,990 / 0,721 / 0,717 (F1 0,739 com o limiar reescolhido). **O A3 ficou um pouco acima do A2 nos 2.069, mas com uma seed e poucos positivos (28 a 48 por rótulo) isso não está demonstrado.** IC de 95% do AUROC na validação completa (bootstrap por paciente, 1.000 reamostragens, seed 42): 1dAVb 0,961-0,981; RBBB 0,993-0,997; LBBB 0,990-0,997; SB 0,990-0,995; ST 0,994-0,996; AF 0,978-0,992. O early stopping parou na época 7 (melhor: época 4, perda de validação 0,0298); a perda de treino continuou caindo (0,0275 na época 7) enquanto a de validação subiu (0,0315), sinal de começo de sobreajuste. Tudo isto é da validação, que escolheu a época e os limiares: otimista.
- **A2+:** uma mudança por vez (ex.: peso por classe, mais partes, normalização, arquitetura maior), sempre na tabela de experimentos.
- **Limiar:** a sigmoid dá uma nota de 0 a 1. Para virar "sim/não", escolher um limiar **por rótulo na validação** (proposta: o que maximiza o F1).
- **Testes externos:** o modelo escolhido na validação é aplicado, **sem retreinar e com os mesmos limiares**, no teste interno, no PTB-XL e no CODE-TEST. **PTB-XL é o teste externo principal e o CODE-TEST o secundário** (decisão de 07/11, com a limitação descrita na seção 4).

### 5.4 Parte B: óbito em até X anos (binário)
- Só o primeiro exame de cada paciente (o que tem `death`/`timey`), aplicando a regra do horizonte (seção 3.6). Horizonte X a decidir (1, 2 ou 3 anos).
- **B0:** regressão logística com idade + sexo.
- **B1:** B0 + `nn_predicted_age` e a diferença `nn_predicted_age − age`.
- **B2:** CNN só com o ECG.
- **B3:** CNN com ECG + idade + sexo (ex.: concatenar idade e sexo depois do GlobalAveragePooling).
- A pergunta é **quanto B2/B3 ganham de B0**. Também vale olhar o resultado por faixa de idade (por causa do paradoxo de Simpson).

### 5.5 Métricas
- **Parte A:** por rótulo: AUROC e AUPRC (não dependem de limiar) e, com o limiar da validação, sensibilidade (recall), precisão, F1 e especificidade. Médias macro. Acurácia só como referência.
- **Parte B:** AUROC e AUPRC, sempre lado a lado com o B0. Curva de calibração se der tempo.
- Seeds fixas (42). Nos resultados finais, idealmente várias seeds com média ± desvio.

### 5.6 Decisões em aberto (T1: decidir com a dupla e registrar na seção 10)
| Decisão | Proposta do Claude da Renata | Por quê |
|---|---|---|
| Quantas partes usar | (07/11, **provisório, a confirmar com a Renata**: o A3 usou as partes 0 a 3; A1 e A2 só a parte 0) Desenvolver com a parte 0; resultados com 3-4 partes (~60-80 mil exames) | Cabe na RAM (16 GB) e em ~4 min por época. (Tom, 07/11: **como array inteiro em float32 não cabe** no Computador do Tom: 4 partes ≈ 15,7 GB, e em float16 ≈ 7,9 GB; ver T9) |
| Proporção do split | 70/10/20 | Teste grande o bastante para alterações raras |
| Normalização do sinal | Nenhuma no A1 (os valores já estão em mV, numa faixa razoável); testar depois como A2 | Uma coisa por vez |
| Exames com alguma derivação zerada (411 nas partes 0 a 3: 172 todo zero e 239 parciais; na parte 0 são 99: 38 todo zero + 61 parciais) | Descartar | Poucos (0,5% dos 80.000 exames), e podem confundir o modelo. **Decidido em 08/11 (Tom e Renata) para a avaliação: excluir só os com as 12 derivações zeradas, nos três testes, pelo sinal** (o treino e a validação do A3 já seguem isso). Antes, provisório no desenvolvimento do A1/A2 (07/11, **provisório, a confirmar com a Renata**): descartar os exames com as 12 derivações zeradas, só do treino e da validação da parte 0 (27 de 13.893 no treino, 4 de 2.073 na validação; os outros 7 dos 38 da parte 0 estão no teste e ficaram intocados); os 239 parciais ficam; a lista em `code15/exames_zerados_p0-3.npy` não foi alterada |
| Limiar por rótulo | Maximizar o F1 na validação | Simples de explicar |
| Horizonte da Parte B | 2 anos (4.716 óbitos, 2,5%) | Equilíbrio entre número de óbitos e pacientes elegíveis |
| `AFLT`, `IRBBB`, `ILBBB` no PTB-XL | **Decidido em 08/11 (Tom e Renata): ligada.** Ficam fora da avaliação (nem positivo nem negativo) só do rótulo ao lado. Implementado como parâmetro do JSON | Não existem como classe na CODE |
| Extensões (fora das aulas) | Saliency/Grad-CAM, peso por classe/focal loss, ResNet 1D maior (GPU), análise de sobrevivência (Cox) | Depois do núcleo |
| Escala dos sinais entre CODE-15%, CODE-TEST e PTB-XL (T8) | Proposta do Claude do Tom: medir as amplitudes antes de qualquer teste externo (**medido em 07/11**, seção 4). **Decisão em aberto:** normalizar ou não, e como | Por exame, o PTB-XL tem ~0,4-0,5× a amplitude da CODE-15%; a CODE tem offsets e picos grandes (|x| > 5 em 39% dos exames vs 2,1% no PTB-XL); causa **não verificada** |
| Limite de `likelihood` do PTB-XL e mapeamento dos rótulos (T5) | **Decidido em 08/11 (Tom e Renata): `presente`** (código em `scp_codes` com qualquer valor; seção 4). Registros: todos os 21.799, bootstrap por `patient_id` | SB, ST e AF vêm com `likelihood` 0 em quase todos os registros: um limite `> 0` ou `>= 50` deixa SB com 0 positivos (seção 4) |
| Como alimentar a CNN sem carregar tudo na RAM (T9) | **Decidido (Tom, 07/11):** gerador em lotes lendo o HDF5 em float32, sem converter para float16 (`ecg_dados.py`) | 4 partes em float32 ≈ 15,7 GB; em float16 ≈ 7,9 GB (cálculo). Leitura medida: ~62 ms por lote de 64, pico de RAM ~0,5 GB |

---

## 6. Ambiente e como rodar

### Montar o ambiente (quem acabou de clonar)
1. Python 3.12 (o 3.11 falha; ver seção 0). Criar o ambiente virtual **na raiz do repositório**: `python -m venv .venv`. Clonar o repositório em um caminho curto, fora de pastas sincronizadas.
2. Instalar: `.venv\Scripts\python -m pip install -r requirements.txt` (Windows) ou `.venv/bin/python -m pip install -r requirements.txt` (Mac/Linux).
3. No VS Code, abrir `projeto_ecg.ipynb` e escolher o kernel `.venv`.
4. Rodar tudo. A seção 2 baixa sozinha o `exams.csv` e as partes listadas em `PARTES` para a pasta `code15/` (~20 min por parte no computador da Renata, ~6 min no do Tom; precisa de ~7 GB livres por parte durante o download, porque o zip só é apagado depois de extraído).

### Rodar o notebook pela linha de comando (como o Claude da Renata faz)
`.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 projeto_ecg.ipynb` (com a parte 0 já baixada, leva poucos minutos). As saídas ficam salvas no notebook e vão para o git, para a outra pessoa ver os resultados sem rodar.

### Recursos
- **Computador da Renata:** Windows, i5-1235U, 16 GB de RAM, sem GPU utilizável (a Intel Iris Xe integrada não serve para o TensorFlow, e o TensorFlow ≥ 2.11 nem usa GPU no Windows nativo). Uma CNN de ~406 mil parâmetros leva ~0,2 s por batch de 64 (~1 min por época com 16 mil exames). Cada parte ocupa ~2 GB de RAM em float16 se carregada inteira pelo notebook; o gerador (`ecg_dados.py`) não carrega parte nenhuma inteira.
- **Colab/Kaggle (GPU):** opção para modelos maiores. `teste_viabilidade_code15_colab.ipynb` (descartável) mede download e treino com GPU; **ainda não rodado no Colab**. No Colab, os dados somem quando a sessão fecha (precisa baixar de novo ou usar o Google Drive).
- **Computador do Tom** (medido em 07/11): Windows 11 Home, Intel Core i7-12700H (14 núcleos, 20 threads), **15,7 GB de RAM** (3,8 GB livres na hora da medida), placas de vídeo Intel UHD Graphics e **NVIDIA GeForce RTX 3060 Laptop**, disco C: de 953 GB com ~253 GB livres antes dos downloads. Python 3.12.10 (instalado com o winget). Pela limitação já registrada acima (TensorFlow ≥ 2.11 não usa GPU no Windows nativo), a RTX 3060 não deve ser usada; **não testado aqui**, e usar a GPU via WSL2 é **não verificado**. Tempo de treino neste computador: **medido em 07/11** (A1: 20,3 min para 14 épocas; seção 9). A RAM é o limite principal: ver T9. **Leitura pelo gerador (medido em 07/11, 5 lotes de treino):** ~62 ms por lote de 64 exames, ou seja ~1 min por época de 55.679 exames só de leitura; pico de RAM ~0,5 GB. A estimativa preliminar de 9 a 27 s por época (leitura de 1.000 exames aleatórios) era **otimista**. O cache do disco pode ter ajudado (leitura com cache frio: **não verificado**), e o tempo de treino real **foi medido depois, em 07/11, com o A1** (tabela da seção 9).

---

### Código fora do notebook (07/11)
- **`ecg_dados.py`:** `tabela_com_grupo()` (mesmo split do notebook), `construir_indice()` (cada exame → parte e linha no HDF5; lê só `exam_id`), `listar_zerados()` (só lista) e `GeradorECG` (`keras.utils.Sequence`, lote de 64, float32; treino embaralhado com seed 42 + época, validação/teste em ordem fixa, previsões alinham com `g.info` e `g.Y`).
- **`a1_cnn.py`:** CNN A1 da seção 5.3 (`python a1_cnn.py bench` mede o tempo; `python a1_cnn.py treinar A1`; `treinar A2 --normalizar`; `treinar A3 --normalizar --partes 0,1,2,3 --max-min 120 --max-epocas 12`). A avaliação usa o melhor modelo salvo (`modelos/<nome>_melhor.keras`), porque o EarlyStopping só restaura pesos se disparar. Tudo marcado `[PROV]` no código é **provisório, a confirmar com a Renata**. Só carrega treino e validação (o grupo teste é cortado em `carregar_indices`). Grava `resultados/experimentos.csv`, `resultados/<nome>_log_por_epoca.csv`, `resultados/<nome>_previsoes_validacao.npy` e o melhor modelo em `modelos/` (fora do git).
- **`gerar_congelado_candidato.py`, `congelado_candidato_A3.json`, `avaliar_externo.py` (07/11, **provisório, a confirmar com a Renata**):** o JSON reúne o modelo A3 (caminho e SHA256), o pré-processamento do A2, as partes, a regra de exclusão, os limiares por alteração (F1 máximo na validação completa do A3), a seed, o commit base e o mapeamento do PTB-XL, com o limite de `likelihood` "a decidir". **Status "candidato": NÃO está congelado**; congelar é decisão da dupla (mudar o status à mão). O `avaliar_externo.py` aplica o modelo sem retreinar, com os limiares e o pré-processamento do JSON (nenhuma estatística calculada no conjunto avaliado), e dá AUROC, AUPRC, sensibilidade, precisão, F1 e especificidade por alteração e macro, com IC de 95% por bootstrap (1.000 reamostras, seed 42, por paciente). **Trava:** em `interno`, `ptbxl` e `code_test` ele recusa rodar, antes de abrir qualquer dado, se o JSON não tiver status "congelado" E faltar a flag `--confirmo-congelado`. Hoje só `validacao_A3` roda livre. **Implementados em 08/11 (**provisório, a confirmar com a Renata**), mas NUNCA executados com o modelo:** `interno`, `ptbxl` e `code_test`. Além da trava, o script recusa um teste se o JSON ainda tiver itens "a decidir" em `ptbxl`.
- **Caminhos de teste do `avaliar_externo.py` (implementados em 08/11, **provisório, a confirmar com a Renata**, sem nenhuma avaliação feita):** `interno` = GeradorECG em ordem fixa sobre o grupo teste das partes 0 a 3, sem os exames com as 12 derivações zeradas (mesma regra do A3, decidida pelo sinal); `ptbxl` = `preprocessar_ptbxl` + `normalizar_exame` (`GeradorPTBXL(normalizar=True)`), rótulos de `scp_codes` com o mapeamento do JSON, **limite de likelihood como parâmetro do JSON** (padrão provisório `presente` = código presente com qualquer valor; opções `presente`, `>0`, `>=50`, `=100`) e **exclusão AFLT->AF, IRBBB->RBBB, ILBBB->LBBB como parâmetro** (padrão provisório ligado: o registro sai só da avaliação do rótulo correspondente), bootstrap por `patient_id`; `code_test` = `GeradorCodeTest` (reordena `[0,1,2,5,3,4,6..11]`, float64 para float32, normaliza), rótulos do `gold_standard.csv`, bootstrap por exame. Limitações do CODE-TEST: AF tem só 13 positivos e a sobreposição de pacientes com a CODE-15% é **não verificada**. `testar_formato.py` confere só o formato e a contagem de rótulos (sem modelo e sem `predict`); `congelar.py` muda o status do JSON para "congelado" e faz um commit local, **só quando alguém o rodar manualmente (pede para digitar CONGELAR); nunca foi executado**.
- **`a0_baselines.py`:** baselines A0 (treino → validação; o teste não é aberto). Escreve `resultados/experimentos.csv`, que é a tabela de experimentos gerada pelo código.
- **Como rodar** (na raiz do repositório): `.venv\Scripts\python a0_baselines.py`. **Sem a opção `-I`** do Python, porque ela tira a pasta do script do caminho de imports e o `import ecg_dados` falha.
- **PTB-XL em `ecg_dados.py`:** `ptbxl_indice()`, `preprocessar_ptbxl()` (500 → 400 Hz com `resample_poly(4, 5)`, 48 zeros de cada lado, float32, sem normalização) e `GeradorPTBXL` (lotes de 64, ordem fixa, só X). Os dados ficam em `ptbxl/` (fora do git).
- O `.npy` de exames zerados fica em `code15/` (fora do git) e se refaz com `listar_zerados()` (~44 s para as partes 0 a 3).
- **Scripts de 08/11 (**provisório, a confirmar com a Renata**; todos só treino/validação):** `parte_b_tabular.py` (B0/B1, regra do horizonte, bootstrap, calibração, faixas de idade, Simpson; ~4 min); `analise_erros_A3.py` (análise de erros e estresse de escala do A3; ~3 min); `seeds_A3.py` (média e desvio entre seeds a partir de `experimentos.csv`); `figuras.py` (só lê `resultados/`). Em `a1_cnn.py`: `--seed N` (outra seed salva em `modelos/<nome>/`, nunca sobrescreve o modelo da seed 42) e `--peso-por-classe` (A4, perda ponderada, fora das aulas; `bce_ponderada` conferida: com peso 1 dá a mesma perda da `binary_crossentropy`). A avaliação dentro de `treinar` passou a carregar o modelo com `compile=False` (só previsão).

## 7. Como trabalhar (processo que as IAs seguem)

### A cada etapa
1. **Explicar antes** à pessoa, em linguagem simples, o que vai ser feito e por quê. Se envolver uma decisão da seção 5.6, **perguntar** em vez de decidir sozinha.
2. Escrever o código no `projeto_ecg.ipynb`, na seção certa, seguindo as convenções abaixo.
3. **Rodar** e **olhar as saídas e os gráficos** (não só conferir se não deu erro).
4. **Conferir com código todo número antes de escrevê-lo no texto.** Na fase 1, um número lido "a olho" num gráfico estava errado e só foi pego porque foi conferido.
5. Escrever as conclusões na célula de texto da subseção, separando o que foi **verificado** do que é **hipótese**.
6. Atualizar a **tabela de status** (seção 0 do notebook), a **tabela de tarefas** (seção 0 deste arquivo) e o **log** (seção 10).
7. Avisar a pessoa que **pode commitar** e sugerir uma mensagem curta.

### Convenções do notebook
- Cada título de seção numa **célula markdown só com o título** (`##` seção, `###` subseção) e o texto na célula de baixo, para recolher no VS Code e ver só os títulos.
- Cada seção abre com 1-3 frases sobre o que faz e por quê. Direto, completo, sem prolixidade.
- A tabela de resultados é **gerada pelo código**, nunca digitada à mão.
- Código não rodado é marcado como tal. Nenhum número entra no texto antes de rodar.
- Cores fixas por rótulo (definidas na seção 1 do notebook), iguais em todos os gráficos.

### Coordenação entre Tom e Renata (git)
- **A IA nunca faz commit nem push.** As pessoas commitam (combinado da Renata em 05/11; vale para as duas IAs, a menos que o Tom decida outra coisa e registre aqui).
- **Antes de começar:** `git pull`. **Ao terminar uma etapa:** commit + push logo, com mensagem curta.
- **Nunca editar o mesmo notebook ao mesmo tempo.** Conflito de git em `.ipynb` é difícil de resolver. Antes de mexer no `projeto_ecg.ipynb`, marcar a tarefa como "em andamento (nome)" na tabela da seção 0, commitar e avisar o outro. Para trabalhar em paralelo, usar um notebook de rascunho próprio (ex.: `rascunho_tom.ipynb`) e passar as células prontas para o principal depois.
- Os dados (`code15/`, `mitdb/`) e o `.venv/` ficam **fora do git**. Cada um baixa ou cria os seus.

---

## 8. Regras de ouro (qualquer IA deve obedecer)

1. **Split por paciente sempre.** Nenhum paciente pode ter exames em mais de um grupo. Isso vale também para normalização, oversampling e escolha de hiperparâmetros.
2. **Nunca só acurácia.** Métricas por rótulo; na Parte B, sempre comparadas com o baseline de idade + sexo.
3. **Nada é escolhido olhando os testes** (teste interno, PTB-XL, CODE-TEST). Se algo for descoberto olhando o teste, registrar, mas não usar para mudar o modelo sem uma justificativa que use só treino e validação. Isso aconteceu na fase 1 (validação que tirou do treino um tipo de batimento).
4. **Não inventar dataset, número, licença ou resultado.** Se não verificou, escreva "não verificado".
5. **Técnica fora das aulas pode entrar** (decisão da Renata em 05/11, confirmada pelo Tom em 06/11), mas **sempre marcada como "fora das aulas"** no notebook e explicada em linguagem simples. O núcleo vem primeiro.
6. **Explicar cada escolha em linguagem simples.** A dupla precisa defender tudo na apresentação.
7. **Não mudar o plano em silêncio.** Toda mudança vira uma linha no log (seção 10).
8. **O texto do paper é da dupla.** A IA ajuda a estruturar e revisar; os resultados precisam bater com o que foi rodado.
9. **Uma coisa por vez** nos experimentos, com tabela de experimentos (configuração, épocas, métricas).
10. Responder em **português**, tom natural e direto. O Tom gosta de ser chamado de "mestre". A Renata é chamada de Renata.
11. **Instalar pacotes só dentro do `.venv/`.** Em 05/11, um `pip install` no Anaconda base da Renata atualizou numpy/pandas e quebrou o ambiente dela (foi restaurado).

---

## 9. Estado de verificação e armadilhas conhecidas

### Verificação
| Item | Status |
|---|---|
| CODE-15%: contagens, colunas, licença, 18 partes | **Conferido** (Zenodo + `exams.csv`, 05/11) |
| CODE-15%: formato do HDF5, durações, zeros centralizados, exames zerados | **Conferido** na parte 0 (05-06/11) |
| Óbito só no primeiro exame de cada paciente | **Conferido** (06/11) |
| CODE-15%: ordem das derivações (DI, DII, DIII, AVR, AVL, AVF) | **Medido** em 07/11 (Einthoven, 2.997 exames da parte 0) |
| `nn_predicted_age` = modelo de Lima et al. | **Provável** (MAE e r batem com o artigo); não documentado |
| Origem dos rótulos da CODE-15% (laudo médico, sistema automático?) | **Não verificado** |
| PTB-XL: tamanho, derivações, códigos das 6 classes, leitura com `wfdb` | **Conferido** (05/11) |
| PTB-XL: licença e tamanho do download | **Conferido** na página do PhysioNet (07/11): CC BY 4.0; 3,0 GB descompactado (zip de 1,7 GB) com `records100` + `records500`; `records500` sozinho: **não verificado** |
| PTB-XL: ordem real das 12 derivações | **Medido** em 07/11 (Einthoven, 1.000 registros): I, II, III, AVR, AVL, AVF; a página do PhysioNet (AVL, AVR) está trocada |
| PTB-XL: escala das amplitudes vs CODE-15% e CODE-TEST | **Medido** em 07/11 (seção 4): PTB-XL ~0,4-0,5× por exame; a CODE tem cauda pesada e offsets; unidade da CODE **não verificada** |
| PTB-XL: download e integridade | **Conferido** em 07/11: 43.600 arquivos, 2.635.735.664 bytes, SHA256 igual ao `SHA256SUMS.txt` do PhysioNet |
| PTB-XL: contagens de rótulos por `likelihood`, AFLT/IRBBB/ILBBB, idade e sexo | **Medido** em 07/11 (seção 4); limite **não escolhido** |
| `GeradorPTBXL`: formato e tempo | **Medido** em 07/11: (64, 4096, 12) float32; ~2,2 s por lote de 64 (leitura com `wfdb` ~32 ms por registro), ~12 min para os 21.799 registros; pico de RAM ~0,45 GB; 48 zeros de cada lado conferidos; registro 0 igual à leitura direta + `resample_poly` |
| CODE-TEST: existe, aberto, CC BY 4.0, 827 ECGs, 6 classes | **Conferido** pela ficha do Zenodo e, em 07/11, abrindo o zip (tamanho e md5 conferidos) |
| CODE-TEST: ordem das derivações (DI, DII, DIII, AVL, AVF, AVR) | **Medido** em 07/11 (Einthoven) e igual ao README |
| CODE-TEST vs CODE-15%: escala das amplitudes | **Não verificado** (desvio padrão por derivação difere; T8) |
| CODE-TEST × CODE-15%: pacientes em comum? | **Não verificado**, e não dá para verificar por identificador: o CODE-TEST não tem `patient_id` nem `exam_id` (07/11). Limitação declarada |
| Treino local (CPU) | **Medido**: ~1 min por época com 16 mil exames |
| Treino no Colab (GPU) | **Não testado** |
| Python da Renata | Anaconda em `C:\Users\rehle\anaconda3` (fora do PATH, **não usar para o projeto**). O projeto usa o `.venv/` (versões no `requirements.txt`) |
| Python do Tom | 3.12.10 (winget), `.venv` criado com o comando do `requirements.txt` (07/11). O 3.11.9 que veio no Windows **falha** com o `requirements.txt` |
| CODE-15%: exames com derivação zerada nas partes 0 a 3 | **Medido** (07/11): 411 (172 todo zero, 239 parciais); causa dos parciais **não investigada** |
| Baselines A0 (validação, partes 0 a 3) | **Rodado** (07/11): macro AUROC 0,705, AUPRC 0,046, F1 0,095 (A0b); A0a AUROC 0,500, F1 0, acurácia 0,980 |
| Gerador: sinal, rótulo e `exam_id` alinhados; pacientes em dois grupos | **Conferido** (07/11) em 640 exames contra leitura independente; 0 pacientes em mais de um grupo (índice e tabela inteira) |
| Tempo de treino real com o gerador (CPU, i7-12700H) | **Medido** em 07/11: ~370-430 ms por lote de 64 (leitura ~100 ms + cálculo ~390 ms), ~80-90 s por época de 13.866 exames (217 lotes) mais a validação; A1: 14 épocas em 20,3 min (early stopping, melhor época 11); A2: 9 épocas em 17,6 min (melhor época 6). Em 1 época de cada (época 5 do A1: 141 s; época 4 do A2: 140 s) o tempo subiu para ~140 s; causa **não verificada**. A estimativa de `bench` (1,9 min/época) foi um pouco alta |
| A3: tempo real de treino (partes 0 a 3, CPU, i7-12700H) | **Medido** em 07/11: 869 lotes de treino por época; 220-259 s por época nas épocas 1 a 4 (~253-298 ms por lote) e ~420-440 s nas épocas 5 a 7 (~490-506 ms por lote); 7 épocas em 37,5 min (melhor: época 4). A estimativa do `bench` (10,1 min por época, leitura ~260 ms + cálculo ~385 ms por lote) foi alta nas primeiras épocas e próxima nas últimas. Causa da lentidão a partir da época 5: **não verificada** |
| `avaliar_externo.py` reproduz o A3 na validação | **Conferido** em 07/11: macro AUROC 0,9892, AUPRC 0,6908, F1 0,7087 (iguais aos de `experimentos.csv`); IC 95% macro: AUROC 0,987-0,991, AUPRC 0,664-0,724, F1 0,685-0,733 (8.158 exames, 7.206 pacientes) |
| Teste interno: exames e positivos (só contagem) | **Medido** em 08/11 (**provisório, a confirmar com a Renata**): 16.141 exames no grupo teste das partes 0 a 3, 32 com as 12 derivações zeradas excluídos pela mesma regra do A3, ficam 16.109 de 14.360 pacientes; positivos: 1dAVb 281, RBBB 442, LBBB 290, SB 308, ST 353, AF 310. Nenhum modelo foi aplicado |
| Formato dos conjuntos de avaliação | **Conferido** em 08/11 com 2 lotes de cada (validação do A3, teste interno, PTB-XL, CODE-TEST): (64, 4096, 12), float32, 0 NaN/inf; nas 3 primeiras amostras de cada lote, média ~0 e desvio 1,000 por derivação na região ativa (4.000 amostras úteis no PTB-XL, com as pontas em zero). Sem modelo e sem `predict` |
| Pré-processamento A2 em 50 exames de validação | **Conferido** em 08/11: diferença máxima de 4,3e-05 entre `normalizar_exame` e uma reimplementação independente (float64) a partir do HDF5; `avaliar_externo.py --conjunto validacao_A3` continua dando macro AUROC 0,9892, AUPRC 0,6908, F1 0,7087 depois da refatoração (nenhuma das previsões salvas foi usada) |
| Trava do `avaliar_externo.py` | **Testada** em 07/11 só nas recusas: `interno`, `ptbxl` e `code_test`, com status "candidato", com e sem a flag, recusaram antes de ler dados; com status "congelado" e sem a flag, recusou; conjunto desconhecido, recusou. O caso "congelado + flag" **não foi executado** (abriria o teste). Em 08/11 repetido para os 3 conjuntos de teste (6 recusas) e testado que um JSON "congelado" com item "a decidir" em `ptbxl` também recusa, antes de carregar modelo ou dados |
| Sinais idênticos em treino e validação: quem são | **Medido** em 07/11 (só treino e validação; o teste NÃO foi lido): 388 exames em 180 grupos (166 de um mesmo paciente, 14 com pacientes diferentes), 273 pares, 173 do mesmo paciente e 100 de pacientes diferentes. **Mesmo paciente:** sexo igual em 173 de 173, idade igual em 169 de 173 (provável repetição do mesmo traçado do mesmo paciente), nunca em grupos diferentes. **Pacientes diferentes:** idade igual em 3 de 100 (3,0%) e sexo igual em 50 de 100 (50,0%), contra 1,4% e 51,9% esperados ao acaso, ou seja, parecem pessoas diferentes; esses 100 pares estão em treino (68) ou em treino + validação (32), nenhum só na validação. 30 dos 388 têm alguma das 6 alterações positiva (os 8 exames de validação que coincidem com o treino não têm nenhuma); 12 têm derivação zerada. Causa: **não verificada**. Nada foi descartado |
| A3: sem vazamento aparente na validação | **Conferido** em 07/11: 0 pacientes e 0 `exam_id` em comum entre treino e validação; 0 pacientes em mais de um grupo; 634 pacientes de validação têm exames em mais de uma parte e ficam todos na validação; previsões salvas = previsões recalculadas lendo o HDF5 sem o gerador (dif. máx 2,1e-7, 300 exames das 4 partes); rótulos independentes = `g.Y`; AUROC com rótulos embaralhados 0,503 (1dAVb) e 0,531 (AF). **Sinais idênticos (MD5 de todos os 63.719 exames de treino e validação):** 388 exames em 180 grupos de sinais idênticos (0,6%); 8 desses grupos misturam treino e validação (8 exames de validação, 0,1%), de pacientes diferentes, e **todos os exames desses grupos têm as 6 alterações negativas**, então não inflam as métricas por positivos; 12 dos 388 têm derivações zeradas. Causa dessas duplicatas: **não verificada** |
| A1/A2: sem vazamento aparente na validação | **Conferido** em 07/11: 0 pacientes e 0 `exam_id` em comum entre treino e validação; 0 sinais idênticos (hash MD5, amostra de 2.000 de cada); previsões salvas = previsões recalculadas lendo o HDF5 sem o gerador (diferença máx 1,5e-7, 200 exames; A2 também); rótulos independentes = `g.Y`; AUROC cai para 0,47 com rótulos embaralhados. Não conferido: a origem dos rótulos da CODE-15% (seção 9, "não verificado") |

### Armadilhas já encontradas (não repetir)
- **`pip install` no Python errado:** quebrou o Anaconda base da Renata. Usar só o `.venv`.
- **Download truncado em silêncio:** conexões do Zenodo caem sem erro. A função `baixar` atual retoma e confere o tamanho; o zip também é testado ao extrair.
- **Soma em float16 estoura** (`np.abs(X).sum()` com milhões de valores vira infinito). Para checar exames zerados, usar `.max()`, ou converter para float32 antes de somar.
- **`wfdb` com o PhysioNet:** o `pn_dir` tem que incluir a subpasta do registro.
- **TensorFlow no Windows não usa GPU** (≥ 2.11). O aviso no treino é esperado.
- **Terminal do Windows mostra acentos embaralhados** na saída de scripts. Os arquivos estão em UTF-8 e estão certos; é só a exibição.
- **Conclusões "a olho":** conferir todo número com código antes de escrever.
- **Comparações de risco sem controlar a idade enganam** (paradoxo de Simpson, seção 3.5).
- **Python 3.11 não instala o `requirements.txt`** (`numpy==2.5.4` exige 3.12 ou mais). Conferir `python --version` antes de criar o `.venv`.
- **Projeto em pasta sincronizada (OneDrive):** os dados (até 46 GB) e o `.venv` seriam candidatos a subir para a nuvem. Clonar em caminho curto, fora dela.
- **A ordem das derivações difere entre as bases:** CODE-15% (AVR, AVL, AVF) e CODE-TEST (AVL, AVF, AVR). Conferir com as fórmulas de Einthoven antes de testar. PTB-XL (medido em 07/11): I, II, III, AVR, AVL, AVF, igual à CODE-15%.

---

## 10. Log de decisões (acrescentar ao final, com data e autor)

- 2026-11-04 a 2026-11-05: **fase 1 (MIT-BIH, arritmia por batimento)**. Log completo em `arquivo/CLAUDE_fase1_arritmia.md`. Resultados principais (seed 42): MLP com split aleatório F1 macro 0,893; mesma MLP com split por paciente 0,385 (recall de S 0,752 → 0,046). Pausada em 05/11 após o feedback dos diretores.
- 2026-11-05 (Claude da Renata): avaliadas novas bases (seção 3.2). Viabilidade da CODE-15% testada: dados, PTB-XL, CODE-TEST e tempo de treino local (seções 4 e 6).
- 2026-11-05 (Claude da Renata): criado `teste_viabilidade_code15_colab.ipynb` (descartável) para medir tempos no Colab.
- 2026-11-06 (Renata e Tom): **novo projeto escolhido**: CODE-15% com Parte A (6 alterações + testes externos) e Parte B (óbito em X anos vs idade + sexo vs idade do ECG). MIT-BIH fora deste projeto.
- 2026-11-06 (Claude da Renata): plano antigo copiado para `arquivo/CLAUDE_fase1_arritmia.md`; este `CLAUDE.md` reescrito. `code15/exams.csv` movido para a pasta de dados (fora do git).
- 2026-11-06 (Claude da Renata): **criado `projeto_ecg.ipynb`**, com as seções 1-5 rodadas (parte 0) e as seções 6-9 como estrutura. Achados da exploração na seção 4 deste arquivo e no notebook (prevalências, efeito da idade, horizontes, paradoxo de Simpson com a idade do ECG).
- 2026-11-06 (Claude da Renata): **split por paciente implementado** (provisório 70/10/20; seção 5.2).
- 2026-11-06 (Claude da Renata): `CLAUDE.md` reorganizado para a outra dupla conseguir continuar sozinha: seção 0 (comece aqui + tarefas), seção 3 (linhas de raciocínio), seção 6 (ambiente), seção 7 (processo de trabalho e coordenação no git), armadilhas conhecidas, tabela de decisões em aberto com propostas. Conferido que os exames curtos têm zeros dos dois lados (581 + 581). `h5py==3.14.0` adicionado ao `requirements.txt`.
- 2026-11-06 (Claude da Renata): seção 0 ganhou o checklist "Primeira vez neste computador?" (pull, `.venv`, dados, conferência, registro) e a rotina "Antes de cada sessão", para a IA do Tom montar o ambiente antes de começar.
- 2026-11-07 (Tom e Renata): **decisão: PTB-XL = teste externo principal; CODE-TEST = teste secundário**, com a limitação de que o CODE-TEST não tem `patient_id` nem `exam_id` e a sobreposição com a CODE-15% é **não verificada**.
- 2026-11-07 (Claude do Tom): **CODE-TEST baixado e inspecionado** (zip de 218.189.452 bytes, md5 conferido). Conteúdo, ordem das derivações (AVL, AVF, AVR) e números do gabarito na seção 4. O `.gitignore` ganhou `code_test/`.
- 2026-11-07 (Claude do Tom): **PTB-XL consultado no PhysioNet (só leitura, nada baixado):** CC BY 4.0; 3,0 GB descompactado (zip de 1,7 GB). A página lista as derivações como I, II, III, AVL, AVR, AVF, o que conflita com o registro de 05/11 (AVR, AVL, AVF); ordem real **não verificada**.
- 2026-11-07 (Claude do Tom): **ambiente montado no computador do Tom.** O Python 3.11.9 falhou no `requirements.txt` (numpy 2.5.4 exige 3.12+); instalado o Python 3.12.10 (winget); projeto clonado em caminho curto fora do OneDrive (a cópia antiga no OneDrive foi mantida); `.venv` criado com o comando do `requirements.txt`. Hardware na seção 6.
- 2026-11-07 (Claude do Tom): **dados baixados no computador do Tom:** `exams.csv` (35.500.900 bytes, md5 conferido) e partes 0 a 3 (zips com tamanho e md5 iguais aos do Zenodo; `testzip` sem erro; cada HDF5 com 3.932.518.664 bytes, `tracings` 20.001 × 4.096 × 12 float32, 1 linha de preenchimento `exam_id` = 0, todos os outros ids na tabela e apontando para a própria parte). Pela tabela, as partes 0 a 3 somam 80.000 exames de 71.477 pacientes (treino 55.679, validação 8.180, teste 16.141 exames, com o split por paciente de sempre); o notebook ainda descarta exames zerados. Sobraram ~237 GB de disco.
- 2026-11-07 (Claude do Tom): **ordem das derivações da CODE-15% medida** (Einthoven, 2.997 exames da parte 0): DI, DII, DIII, AVR, AVL, AVF. Confirma o registro anterior. No CODE-TEST a ordem é DI, DII, DIII, AVL, AVF, AVR.
- 2026-11-07 (Claude do Tom): **pendências novas:** T8 (escala dos sinais entre as bases) e T9 (RAM: 4 partes em float32 ≈ 15,7 GB não cabem como array inteiro no Computador do Tom; alimentar a CNN em lotes lendo do HDF5).
- 2026-11-07 (Claude do Tom): o Tom pediu, uma única vez, que a IA fizesse o commit e o push do `.gitignore` (commit `fc24f3d`). A regra "a IA nunca faz commit nem push" continua valendo.
- 2026-11-07 (Claude do Tom): **T3 e T9 (parte técnica):** criados `ecg_dados.py`, `a0_baselines.py` e `resultados/experimentos.csv`. A0 rodado (treino → validação, partes 0 a 3, 55.679 e 8.180 exames; teste não aberto): macro A0b AUROC 0,705, AUPRC 0,046, F1 0,095; A0a AUROC 0,500, acurácia 0,980. Lista de exames zerados nas partes 0 a 3: 411, **só listados, nenhum descartado** (T1). Gerador em lotes testado: X (64, 4096, 12) float32, ~62 ms por lote, pico ~0,5 GB; 640 exames conferidos; 0 pacientes em mais de um grupo. Decisão do Tom: gerador lendo o HDF5 em float32, sem float16. Tempo de treino real: não medido.
- 2026-11-07 (Claude do Tom): o Tom pediu, por escrito, que a IA fizesse o commit e o push desta rodada de atualizações do `CLAUDE.md` (só se o `origin/main` não tivesse commits novos; sem `--force`). A regra "a IA nunca faz commit nem push" continua valendo para o resto.
- 2026-11-07 (Claude do Tom): **T5 (preparo) e T8:** PTB-XL baixado e conferido (2.635.735.664 bytes, SHA256 ok). Ordem das derivações do PTB-XL medida: igual à CODE-15% (página do PhysioNet trocada). Escala medida sem normalizar: PTB-XL ~0,4-0,5× a CODE-15% por exame, CODE com offsets e picos grandes; unidade da CODE não verificada. Contagens de rótulos por `likelihood`: SB, ST e AF vêm com `likelihood` 0, então o limite muda tudo (decisão da dupla, não escolhida). Idade e sexo do PTB-XL comparados com a CODE-15%. `preprocessar_ptbxl` e `GeradorPTBXL` testados (~2,2 s por lote, pico ~0,45 GB). Nenhum modelo rodado, nenhuma métrica calculada, nenhum rótulo usado para decidir pré-processamento.
- 2026-11-07 (Claude do Tom): **T4, A1 e A2 provisórios (**provisório, a confirmar com a Renata**):** criado `a1_cnn.py`. Dados: só a parte 0, via `GeradorECG`; descartados só os exames com 12 derivações zeradas, só do treino e da validação (27 e 4); limiar = F1 máximo na validação; A1 sem normalização. Teste interno, PTB-XL e CODE-TEST não foram abertos. A1: macro AUROC 0,976, AUPRC 0,656, F1 0,696 na validação (A0b no mesmo conjunto: 0,697, 0,046, 0,103); 14 épocas em 20,3 min. A2 provisório (normalização por exame e derivação, calculada só fora do preenchimento de zeros): macro AUROC 0,985, AUPRC 0,682, F1 0,717; 9 épocas em 17,6 min. AF do A2 deu AUROC 0,983 (> 0,98): investigado (sem vazamento achado; IC95 0,965-0,993; poucos positivos). Uma seed; diferença A1 vs A2 dentro do intervalo de erro. Nenhuma decisão da seção 5.6 foi fechada.
- 2026-11-07 (Claude do Tom): **A3 (**provisório, a confirmar com a Renata**):** A2 com as partes 0 a 3 (uma mudança). Treino 55.561 e validação 8.158 exames (118 + 22 todo-zero descartados); `bench` estimou 10,1 min por época, real 3,7 a 7,3 min; 7 épocas em 37,5 min, melhor época 4. Validação completa: macro AUROC 0,989, AUPRC 0,691, F1 0,709 (A0b no mesmo conjunto: 0,705, 0,046, 0,095). Nos mesmos 2.069 exames de A1/A2: 0,990, 0,721, 0,717 (F1 0,739 com limiar reescolhido). Checagem de vazamento: nada achado; 8 exames de validação têm sinal idêntico a exames de treino de outros pacientes, todos sem alteração. Uma seed. Teste interno, PTB-XL e CODE-TEST não abertos. Nenhuma decisão da seção 5.6 fechada.
- 2026-11-07 (Claude do Tom): **preparo para o teste externo (**provisório, a confirmar com a Renata**):** (1) lida a documentação do `likelihood` do PTB-XL (página do PhysioNet e artigo): para ritmo o 0 significa sem informação; critério proposto só pela documentação, não escolhido. (2) Criados `congelado_candidato_A3.json` (status "candidato") e `gerar_congelado_candidato.py`. (3) Criado `avaliar_externo.py` com trava; reproduziu o A3 na validação (AUROC 0,989, AUPRC 0,691, F1 0,709); teste interno, PTB-XL e CODE-TEST não foram lidos. (4) Diagnóstico dos 388 exames de sinal idêntico em treino e validação: 166 grupos do mesmo paciente e 14 de pacientes diferentes sem sinal de serem a mesma pessoa; nada descartado. Nenhuma decisão da seção 5.6 fechada.
- 2026-11-08 (Claude do Tom): **caminhos de teste do `avaliar_externo.py` (**provisório, a confirmar com a Renata**):** implementados `interno`, `ptbxl` e `code_test` (e `GeradorCodeTest`, `GeradorPTBXL(normalizar=True)`, `exames_todo_zero` em `ecg_dados.py`), `testar_formato.py` e `congelar.py` (este nunca foi executado). JSON continua "candidato", com limite de likelihood "presente", exclusão AFLT/IRBBB/ILBBB ligada e folds "todos" como padrões provisórios. Nenhum modelo foi aplicado e nenhuma métrica foi calculada no teste interno, no PTB-XL ou no CODE-TEST; só formato (2 lotes de cada) e contagem de rótulos. Teste interno: 16.109 exames após excluir 32. PTB-XL: 1 registro com derivação zerada. Nenhuma decisão da seção 5.6 fechada.
- 2026-11-08 (Claude do Tom, sessão autônoma pedida pelo Tom; **provisório, a confirmar com a Renata**): Parte B só com a tabela (B0/B1, horizonte principal de 2 anos declarado antes, 1 e 3 anos como sensibilidade), análise de erros do A3 na validação, seeds 43 e 44 do A3, figuras, `roteiro_slides.md`, `decisoes_pendentes.md` e `RELATORIO_NOITE.md`. Só treino e validação; teste interno, PTB-XL e CODE-TEST não abertos; `congelar.py` não rodado; JSON candidato e modelo A3 com SHA256 iguais antes e depois. `a1_cnn.py` ganhou `--seed` e `--peso-por-classe` (A4, não treinado). O computador parou ~3h45 no meio da seed 43 (causa não verificada); seed 45 e A4 ficaram para depois. Nenhuma decisão da seção 5.6 fechada.
- 2026-11-08 (Claude do Tom): o Tom pediu, por escrito, commit local ao fim de cada bloco e push (só se o `origin/main` não tivesse commits novos; sem `--force`). Feito assim nesta sessão.
- 2026-11-08 (Tom e Renata decidiram; Claude do Tom só registrou): **decisões para o teste:** modelo = A3 seed 42, sem extensões (A4 e outras seeds fora); PTB-XL com limite de `likelihood` `presente`, exclusão AFLT→AF / IRBBB→RBBB / ILBBB→LBBB ligada, todos os 21.799 registros e bootstrap por `patient_id`; exames zerados = só os de 12 derivações zeradas, nos três testes, pelo sinal. `congelado_candidato_A3.json` atualizado (todos os "a decidir" viraram "decidido"; status segue "candidato"). SHA256 de `A3_melhor.keras` igual antes e depois (`bae64fa9...`). `congelar.py` não rodado; teste interno, PTB-XL e CODE-TEST não abertos e nenhum `predict`. Pendência: a exclusão todo-zero só existe no caminho `interno` do `avaliar_externo.py`.

## 11. Divisão de tarefas

- Ver a tabela de tarefas na seção 0. Preencher a coluna "Quem" ao combinar.
