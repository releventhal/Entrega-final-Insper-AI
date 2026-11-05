# PROJETO: Classificação de batimentos cardíacos (arritmia) avaliada do jeito honesto

> **Para qualquer IA que ler este arquivo:** este é o documento único de contexto do projeto. Leia inteiro antes de responder ou escrever código. Ele existe para que duas duplas de pessoa+IA (Tom e Renata, cada um com a sua IA) sigam o MESMO plano. Se algo aqui conflitar com o que o usuário pedir na hora, avise o conflito e pergunte, não mude o plano em silêncio. Este arquivo pode ser renomeado para `CLAUDE.md` quando o projeto estiver num repositório.

---

## 1. Contexto

- **Quem:** Tom (Tomaz Santos Almeida), estudante de Ciência da Computação no Insper, e sua dupla, Renata.
- **Onde:** Trilha de Trainees da liga estudantil **Insper AI** (IA, Machine Learning, Deep Learning).
- **O quê:** projeto final da trilha, em dupla. Segundo as orientações oficiais (`entrega_final.pdf`):
  - Cada grupo propõe, desenvolve e apresenta um projeto de **deep learning** em UMA área: **saúde, segurança ou transporte público e mobilidade**.
  - Objetivo: achar uma boa ideia, planejar a solução, **justificar as escolhas** e **criá-la até o fim**.
  - Entrega intermediária: primeira versão, não precisa estar 100%, apresentada presencialmente com feedback.
  - Entrega final: **notebook + paper (relatório)**. A nota final é baseada na **apresentação**.
  - Não há exigência de demo, aplicativo ou câmera.
  - A lista de "cases de sucesso" é **só inspiração, NÃO pode ser copiada**.
- **Prazos:** intermediária **11/11** (o PDF diz 12/11, mas a dupla confirmou que a nossa é dia 11), entrega (notebook + paper) **17/11**, apresentação final **18/11** (membros, diretores e trainees). Documento escrito em 04/11/2026.
- **Confirmado pelo Tom:** usar MLP é permitido dentro do que o projeto pede.

### O que a dupla já viu nas aulas (use SÓ isso no núcleo)
Regressão linear (A1), escalando o modelo (A2), classificação (A3), do neurônio à rede (A4), MLP + backpropagation (A5), treinando na prática: epoch, batch, otimizadores, curvas de loss (A6), análise de modelos: métricas, matriz de confusão, erros, regularização (A7), Claude Code (A8), CNNs (A9, A10) e transfer learning (A10).
**Ainda NÃO foi visto:** modelos de sequência (RNN/LSTM), linguagem, aprendizado por reforço, Grad-CAM. Qualquer coisa que dependa disso é **bônus** e deve ser marcada como tal.
Experiência prévia: MLP em Keras no desafio MNIST; projeto em grupo de modelagem de carros usados (EDA, regressão, classificação).

### Regra da trilha sobre uso de IA (Aula 8)
Pode usar IA para escrever código. **Não dá para delegar a decisão.** Arquitetura, métricas, divisão dos dados e interpretação são da dupla e precisam ser entendidas e explicadas por elas. Portanto: explique cada escolha técnica em linguagem simples antes ou junto do código.

---

## 2. A ideia em uma frase

Modelos que classificam tipos de batimento cardíaco costumam mostrar ~98-99% de acerto, mas isso é em boa parte ilusório: batimentos do mesmo paciente caem no treino e no teste, e o modelo aprende o **paciente**, não a **arritmia**. O projeto **mostra essa diferença** e tenta **melhorar o resultado honesto**.

### Pergunta de pesquisa
Quanto do desempenho "bonito" some quando o teste usa pacientes que o modelo nunca viu, principalmente nas classes raras e clinicamente importantes? E o que ajuda a recuperar parte dessa queda?

### Hipótese a testar
O split aleatório dá número alto; o split por paciente (protocolo de de Chazal et al.) derruba o desempenho, principalmente na classe **S (supraventricular)**. Técnicas como peso por classe, oversampling e o intervalo entre batimentos recuperam parte disso.

### Narrativa da apresentação
"Mostramos por que os resultados populares são enganosos e construímos um classificador avaliado do jeito que importaria num hospital."

---

## 3. Dados

### Fonte escolhida: MIT-BIH Arrhythmia Database (PhysioNet), versão ORIGINAL
- 48 gravações de ~30 min, de 47 pacientes, 360 Hz, 2 derivações por gravação. A literatura costuma usar a derivação II (MLII).
- Cada gravação tem arquivo de anotações (`.atr`) com a posição e o tipo de cada batimento, marcado por cardiologistas.
- **Por que não o Kaggle:** a versão pré-processada (`mitbih_train.csv` / `mitbih_test.csv`) tem batimentos soltos de 187 valores, 5 classes, 87.554 treino / 21.892 teste. **Não confirmado**, mas pelo que foi visto ela não traz o identificador do paciente, então não permite split por paciente. Por isso o dataset é montado a partir dos arquivos originais.
- Leitura em Python: biblioteca `wfdb` (`wfdb.rdrecord`, `wfdb.rdann`). Existe também a biblioteca `pyheartlib`, que gera datasets de batimentos com split inter-paciente; é uma alternativa a testar no "teste de 30 minutos", mas a dupla precisa entender o que ela faz antes de usá-la.

### Classes (padrão AAMI EC57)
| Classe | Significado | Símbolos originais (conferir na documentação do PhysioNet) |
|---|---|---|
| N | Normal e equivalentes | N, L, R, e, j |
| S | Supraventricular ectópico | A, a, J, S |
| V | Ventricular ectópico | V, E |
| F | Fusão ventricular + normal | F |
| Q | Desconhecido / marcapasso | /, f, Q |

Contagem total na literatura (MIT-BIH completo): N 90.631, S 2.781, V 7.236, F 803, Q 8.043 (Q inclui batimentos de marcapasso). **Cerca de 83% são normais**, daí o desbalanceamento forte.

### Split por paciente (de Chazal et al., 2004)
44 gravações (excluídas as 4 de marcapasso: **102, 104, 107, 217**).
- **DS1 (treino):** 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230
- **DS2 (teste):** 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234
- Observação da literatura: as gravações 201 e 202 são do mesmo paciente e ficam em conjuntos diferentes; todas as outras são de pacientes únicos.
- Contagens aproximadas (variam em 1-2 entre artigos): DS1 = N 45.866 / S 944 / V 3.788 / F 415 / Q 8. DS2 = N 44.259 / S 1.837 / V 3.221 / F 388 / Q 7.
- A classe Q some nesse split. **Avaliar N, S, V, F.**
- Validação: precisa ser separada **por paciente** dentro do DS1 (nunca por batimento).

### Tabela final (um único dataset, usado nos dois splits)
Uma linha por batimento. Colunas mínimas: `paciente` (id da gravação), `classe` (N/S/V/F), `sinal` (janela do batimento), e opcionalmente `rr_anterior`, `rr_posterior` (intervalos entre batimentos).

---

## 4. Plano técnico

### Etapas
> Cronograma ajustado em 05/11 porque a intermediária passou para **11/11** (ver log, seção 7).

1. **Teste de 30 minutos (05/11):** instalar `wfdb`, baixar uma gravação (ex.: 100), plotar alguns segundos com as anotações, confirmar que dá para ler símbolo e posição de cada batimento. Se travar, parar e reportar ANTES de seguir.
2. **Montar a tabela (06-07/11):** cortar uma janela ao redor de cada pico R, mapear os símbolos para as 5 classes, guardar o id do paciente, aplicar as listas DS1/DS2. Fazer EDA: contagem por classe e batimento médio de cada tipo.
3. **Baselines e primeiros modelos (08-09/11):** (a) baseline ingênuo que sempre chuta "N"; (b) MLP com **split aleatório** (reproduzir o resultado "bonito"); (c) a MESMA MLP com **split por paciente**.
4. **Métricas e intermediária (10-11/11):** métricas por classe, matriz de confusão, slides. Apresentação intermediária **11/11**.
5. **Reforço (12-15/11):** peso por classe, oversampling, intervalo RR como entrada, análise dos erros da classe S.
6. **Fechamento (16/11):** notebook e paper finais. Entrega 17/11. Ensaiar para 18/11.

### Experimentos obrigatórios (núcleo)
- E0: baseline "sempre N".
- E1: MLP, split aleatório por batimento.
- E2: MLP, split por paciente (DS1 → DS2).
- E3: E2 + peso por classe.
- E4: E2 + oversampling (**só no treino, depois do split**).
- E5: E2 + features de ritmo (intervalo RR).
- Análise de erros: onde a classe S falha, por paciente e por tipo de confusão.

### Métricas (obrigatório)
Recall, precision e F1 **por classe**, F1 macro, matriz de confusão. **Acurácia geral não pode ser a métrica principal** (83% dos batimentos são N). Mostrar também o desempenho por paciente quando possível.

### Bônus (só se o núcleo estiver pronto; marcar como bônus no texto)
- Conv1D (CNN 1D).
- Grad-CAM / saliency para mostrar que parte da onda o modelo usou. **Não visto nas aulas.**
- Focal loss (não verificado se é compatível com o que foi visto).
- Teste em segundo dataset: **INCART** (PhysioNet; 75 gravações de 30 min, 32 pacientes, 12 derivações, 257 Hz, licença ODC-By 1.0). Exige reamostrar para 360 Hz, escolher derivação e mapear rótulos.
- Alternativa descartada por ora: PTB-XL (21.837 ECGs de 18.885 pacientes, 10 s, multi-label, com folds já separados por paciente). É outro problema, mais próximo de laudo.

### Stack
- Python no Google Colab (um notebook só, na entrega final).
- `wfdb` (leitura), `numpy`, `pandas`, `matplotlib`, `scikit-learn` (métricas, class weights), `Keras/TensorFlow` (MLP; a dupla já usou Keras no MNIST).
- Seeds fixas em tudo. Um repositório só, um notebook principal.

### Decisões ainda abertas (decidir junto com a dupla, e registrar na seção 7)
- Derivação do sinal (sugestão inicial: MLII; a validar).
- Tamanho da janela ao redor do pico R (a literatura varia; a validar com a dupla).
- Normalização por batimento ou global (se global, calcular estatísticas SÓ no treino).
- Arquitetura da MLP e hiperparâmetros (seguir o método da Aula 6: mudar uma coisa por vez e manter tabela de experimentos).
- **Proposta (Claude da Renata, 05/11), aguardando a dupla:** tirar a Conv1D do bônus e colocar como E6 no núcleo, porque CNNs foram vistas nas Aulas 9 e 10.
- **Proposta (Claude da Renata, 05/11), aguardando a dupla:** no E5, usar o RR **normalizado pelo RR médio do próprio paciente** (ex.: RR atual ÷ média local), além do RR cru. Motivo: um RR "curto" depende da frequência cardíaca de cada pessoa, então o valor cru tende a não generalizar entre pacientes.
- **Proposta (Claude da Renata, 05/11), aguardando a dupla:** no split por paciente, rodar 3 a 5 seeds e reportar média ± desvio, porque o resultado inter-paciente oscila bastante.
- **Confirmado nos dados (05/11):** a gravação 232 tem **75% dos batimentos S do DS2** (1.381 de 1.836); 6 das 22 gravações do DS2 não têm nenhum S. No DS1, a 209 tem 41% dos S. O recall de S no DS2 depende muito de um paciente só, então a análise por paciente é obrigatória.
- **Escolhas provisórias da montagem (Claude da Renata, 05/11), a confirmar com o Tom:** derivação MLII; janela larga de 0,5 s antes + 0,5 s depois do pico R guardada no dataset (o tamanho final é recortado na hora do modelo); sinal em mV sem normalização (decidir na seção 5); RR com todos os batimentos da gravação; 44 gravações usadas nos dois splits. Detalhes na seção 3 do notebook.

---

## 5. Regras de ouro (qualquer IA deve obedecer)

1. **Nunca misturar batimentos do mesmo paciente entre treino, validação e teste** nos experimentos "por paciente". Isso inclui oversampling, normalização e seleção de hiperparâmetros.
2. **Nunca usar só acurácia.** Sempre reportar por classe.
3. **Oversampling, peso por classe e qualquer ajuste só no treino.** O teste (DS2) nunca é usado para escolher nada.
4. **Não inventar dataset, número, licença ou resultado.** Se não verificou, escreva "não verificado". Não afirmar resultados antes de rodar.
5. **Não usar técnica fora das aulas no núcleo.** Se usar, rotular como bônus.
6. **Explicar cada escolha em linguagem simples.** A dupla precisa defender na apresentação.
7. **Não mudar o plano em silêncio.** Qualquer mudança vira uma linha na seção 7 (log de decisões).
8. **O texto do paper é da dupla.** A IA pode ajudar a estruturar e revisar, mas os resultados e interpretações precisam bater com o que foi realmente rodado.
9. **Uma coisa por vez** nos experimentos, com tabela de experimentos (configuração, epochs, métricas por classe).
10. Responder em **português**, tom natural e direto. O Tom gosta de ser chamado de "mestre". A Renata é chamada de Renata.

### Combinados de trabalho (definidos pela Renata em 05/11)
- **A IA nunca faz commit nem push.** Quem commita são as pessoas. Quando terminar uma etapa, a IA avisa ("pode commitar") e sugere uma mensagem de commit curta. A ideia é ter commits frequentes e pequenos.
- **Toda mudança que a IA fizer no repositório vira uma linha no log (seção 7)**, com data e quem fez (ex.: "Claude da Renata"). Assim a IA da outra pessoa entende o que mudou ao abrir o projeto.
- **Notebook principal:** `projeto_arritmia.ipynb`. Regras de organização:
  - Cada título de seção fica numa **célula markdown própria, só com o título** (`##` para seção, `###` para subseção). O texto explicativo vai na célula de baixo. Assim dá para recolher a seção no VS Code e ver só os títulos.
  - Seções numeradas na ordem do plano. Cada seção abre com 1-3 frases dizendo o que ela faz e por quê.
  - Direto e completo, sem prolixidade. Explicações em linguagem simples, para a dupla defender na apresentação.
  - A seção 0 tem uma tabela de **status** (o que está pronto, o que falta) e a seção de experimentos tem a **tabela de experimentos**. As duas precisam ser atualizadas sempre que algo mudar.
  - Código não rodado é marcado como tal. Nenhum número entra no texto antes de ter sido rodado.

---

## 6. Estado de verificação

| Item | Status |
|---|---|
| Listas DS1/DS2 e exclusão de 102, 104, 107, 217 | Visto em vários artigos |
| Contagens por classe nos splits | **Conferido em 05/11**: nosso dataset tem DS1 = N 45.824 / S 943 / V 3.788 / F 414 / Q 8 e DS2 = N 44.218 / S 1.836 / V 3.219 / F 388 / Q 7 (100.645 batimentos). S/V/F/Q a ±2 da literatura; N ~40 a menos por descartarmos o 1º e o último batimento de cada gravação. Nessas 44 gravações, **89,5% são N** (o "83%" da seção 3 inclui as de marcapasso) |
| Kaggle sem id de paciente | **Não confirmado** |
| Licença do MIT-BIH | **Não verificada** |
| Mapeamento exato símbolo → classe AAMI | Significado de cada símbolo **conferido** na tabela de anotações da `wfdb` (05/11). Os 15 símbolos da tabela são os únicos de batimento nas 44 gravações; o resto (`+ ~ ! " x \| [ ]`) não é batimento e foi descartado. O agrupamento em classes AAMI segue a literatura; a norma EC57 em si não foi consultada |
| Gravação 114 | MLII é o **2º canal** (V5 é o 1º). O código escolhe o canal pelo nome |
| `wfdb` lê gravações e anotações | **Testado em 05/11** (gravação 100, baixada direto do PhysioNet): 360 Hz, MLII + V5, anotações no pico R |
| INCART: 75 gravações, 32 pacientes, ODC-By 1.0 | Visto na página do PhysioNet |
| Grad-CAM e focal loss nas aulas | Grad-CAM não encontrado; focal loss não verificado |
| Acesso ao PhysioNet pelo ambiente de cada IA | **Desconhecido**; o Colab da dupla é o caminho seguro |
| Python no computador da Renata | Anaconda em `C:\Users\rehle\anaconda3` (Python 3.12.7), **fora do PATH**: chamar pelo caminho completo. Tem numpy 1.26.4, pandas 2.2.2, scikit-learn, Jupyter e `wfdb` 4.1.2. **Sem TensorFlow ainda.** O Claude da Renata roda o notebook com `python -m jupyter nbconvert --execute --inplace` |

---

## 7. Log de decisões (qualquer IA ou pessoa acrescenta ao final, com data)

- 2026-11-04: Projeto escolhido: arritmia por batimento, split aleatório vs por paciente. Um repositório, um notebook.
- 2026-11-04: Dataset será montado a partir do MIT-BIH original; o notebook conterá o código de montagem (a dupla precisa entender e defender).
- 2026-11-04: Liga confirmou (via Tom) que MLP é aceito como deep learning.
- 2026-11-05 (Claude da Renata): intermediária passou de 12/11 para **11/11**. Cronograma da seção 4 puxado um dia para trás.
- 2026-11-05 (Claude da Renata): criados os "Combinados de trabalho" na seção 5 (a IA não commita, toda mudança vai para este log, regras do notebook).
- 2026-11-05 (Claude da Renata): adicionadas três propostas e um ponto a conferir em "Decisões ainda abertas" (Conv1D no núcleo, RR normalizado, várias seeds, gravação 232). **Ainda não decididas pela dupla.**
- 2026-11-05 (Claude da Renata): criado `projeto_arritmia.ipynb` com a estrutura de seções e o código do teste de 30 minutos (seções 1 e 2).
- 2026-11-05 (Claude da Renata): **teste de 30 minutos concluído** no Anaconda da Renata. A leitura funciona; resultado registrado no fim da seção 2 do notebook. Criado `.gitignore` (a pasta `mitdb/` com os dados baixados não vai para o repositório).
- 2026-11-05 (Claude da Renata): **cuidado com `pip install` no ambiente local.** Instalar o `wfdb` mais novo atualizou numpy/pandas e quebrou o Anaconda base; as versões originais foram restauradas. Regra: instalar pacotes com `-c` (arquivo de restrições gerado por `pip list --format=freeze`) para não atualizar nada. A célula de instalação do notebook só instala o `wfdb` se ele não existir.
- 2026-11-05 (Claude da Renata): **etapa 2 concluída (seções 3 e 4 do notebook, rodadas).** Dataset montado (`df` com os metadados + `X` com as janelas, 100.645 × 360). EDA: contagens por classe comparadas com a literatura, batimentos S por paciente, batimento médio por classe e RR por classe. Achados: (1) S tem formato quase igual ao N; (2) com o RR **relativo ao ritmo da gravação**, S fica bem separado de N, o que reforça a proposta do RR normalizado no E5; (3) a 232 concentra 75% dos S do teste.
- 2026-11-05 (Claude da Renata): adicionado um **Glossário** no começo do notebook (classes N/S/V/F/Q explicadas em linguagem simples, pico R, RR, MLII, DS1/DS2, tipos de split) e um "como ler" nos gráficos 4.3 e 4.4, a pedido da Renata.

## 8. Divisão de tarefas (a preencher)

- Tom: _a definir_
- Renata: _a definir_
