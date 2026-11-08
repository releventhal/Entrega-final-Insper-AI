# Relatório da sessão autônoma de 08/11/2026 (Claude do Tom)

Tudo aqui é **provisório, a confirmar com a Renata**. Só treino e validação foram usados. O teste interno, o PTB-XL e o CODE-TEST **não foram abertos, lidos nem avaliados**. `congelar.py` não foi rodado; a flag `--confirmo-congelado` não foi usada; o JSON candidato não foi alterado.

Observação sobre git: o `CLAUDE.md` diz "a IA nunca faz commit nem push". Nesta sessão o Tom pediu por escrito commit local ao fim de cada bloco e push (só se o `origin` não tiver commits novos, nunca `--force`). Sigo o pedido e registro no log do `CLAUDE.md`.

Observação sobre ferramentas: TaskCreate/TaskUpdate não estão disponíveis nesta sessão; a lista de tarefas fica neste relatório.

## Bloco 0, preparação (início 10:17)
- Hora de início: **10:17** (relógio do computador). Corte de treinos novos: 14:47. Fim do orçamento: 15:17.
- `git fetch` + `git status`: `main` igual a `origin/main` (último commit `a3ef90e`), árvore limpa.
- `.venv`: Python 3.12.10, TensorFlow 2.21.0, Keras 3.15.1, scikit-learn 1.9.1, matplotlib 3.11.2. Funciona.
- Disco C: 235 GB livres. RAM livre na hora: ~6,3 GB de 15,7 GB.
- **SHA256 inicial:**
  - `modelos/A3_melhor.keras`: `bae64fa994934072434853ce6c5f6dfffd485a1f126336be1d6e8d526ca12a89` (igual ao campo `sha256` do JSON)
  - `congelado_candidato_A3.json`: `1c145b6a553a47b53a6330f4de44608c8e54aa251183627d7607673a97ecf935`

## Bloco 1, Parte B só com a tabela (10:19-10:23)
Script: `parte_b_tabular.py` (nenhum sinal lido). Saídas em `resultados/parte_b_*.csv`.

**Dados.** Tabela inteira com o split de sempre; o grupo teste é cortado logo na carga. Treino + validação: 187.069 pacientes, 186.975 com `death`/`timey` (um exame por paciente, conferido: `patient_id` único). Esse exame é o de menor idade do paciente em 99,8% dos casos (treino+validação).

**Unidade de `timey`: não verificado pela documentação.** A página do Zenodo (lida em 08/11) define `timey` como "time to the death of the patient. If not, it is the follow-up time", sem dizer a unidade. Os valores (0 a 7,68; mediana 3,45; óbitos com mediana 1,69) são compatíveis com anos, mas isso é inferência, não documentação. Usei como anos, igual ao `CLAUDE.md`.

**Regra do horizonte** (entra quem morreu até X anos ou foi acompanhado X anos; quem ficou vivo e foi acompanhado menos que X sai):

- 2 anos (**principal, declarado antes de rodar**): treino 129.439 elegíveis, 3.368 óbitos, 33.967 saíram por censura; validação 18.711 elegíveis, **450 óbitos**, 4.858 saíram.
- 1 ano (sensibilidade): treino 157.208 / 2.021 / 6.198; validação 22.753 / 251 / 816.
- 3 anos (sensibilidade): treino 100.538 / 4.382 / 62.868; validação 14.481 / 585 / 9.088.

**B0 (idade + sexo) vs B1 (B0 + `nn_predicted_age` + diferença), validação, IC 95% por bootstrap por paciente (1.000 reamostras, seed 42):**

- 2 anos: AUROC B0 0,794 (0,775-0,814), B1 0,804 (0,785-0,822); diferença B1 − B0 IC 0,003 a 0,016. AUPRC B0 0,082 (0,070-0,095), B1 0,086 (0,073-0,102); diferença IC −0,002 a 0,014 (inclui zero). Taxa de óbito na validação: 450/18.711 = 2,4%.
- 1 ano: AUROC 0,810 vs 0,814 (diferença IC −0,004 a 0,013); AUPRC 0,038 vs 0,041.
- 3 anos: AUROC 0,800 vs 0,811 (diferença IC 0,005 a 0,016); AUPRC 0,130 vs 0,136.
- Coeficientes do B1 (variáveis padronizadas): idade 0,70, idade do ECG 0,69, diferença −0,09, sexo masculino 0,49. **Atenção:** diferença = idade do ECG − idade, então as três variáveis são colineares e os coeficientes isolados não são interpretáveis; o que vale é o desempenho.

**Por faixa de idade (2 anos, validação; AUROC B0 → B1; IC da diferença):**
- < 40: 5.996 pacientes, 24 óbitos; 0,589 → 0,673 (−0,002 a 0,179). Poucos óbitos: muito incerto.
- 40-59: 5.552, 66 óbitos; 0,650 → 0,732 (0,047 a 0,126).
- 60-69: 2.742, 64 óbitos; 0,668 → 0,664 (−0,057 a 0,053).
- 70+: 4.421, 296 óbitos; 0,619 → 0,617 (−0,024 a 0,018).
Dentro de cada faixa o AUROC cai (0,59-0,73) porque a idade, que separa muito no geral, varia pouco dentro da faixa.

**Calibração (decis de risco previsto, 2 anos):** B0 e B1 acompanham a taxa observada (decil 10: previsto 0,097 / observado 0,091 no B0; 0,098 / 0,088 no B1); nos decis 5 a 8 os dois superestimam um pouco (ex.: B0 decil 8: 0,040 previsto, 0,030 observado). Tabela em `resultados/parte_b_calibracao_2anos.csv`.

**Paradoxo de Simpson** (mesma definição do notebook: morte em qualquer tempo, primeiro exame, grupos −8 / ±8 / +8 anos; **aqui só treino + validação**, o notebook usou a tabela inteira):
- Sem controle: ECG 8+ mais velho 3,00% (45.084 pacientes, idade média 42,2); semelhante 3,48% (107.336; 50,4); ECG 8+ mais jovem 4,68% (34.555; 62,9). O `CLAUDE.md` diz 3,0% vs 4,6%: **bate** (4,68% vs 4,6%; diferença esperada por excluir o teste).
- Dentro da faixa 60-69: mais jovem 3,18% / semelhante 3,70% / mais velho 6,40%. O `CLAUDE.md` diz 3,0% / 3,6% / 6,5%: **bate no padrão**, com diferenças de 0,1-0,2 ponto (mesmo motivo).
- Em todas as 6 faixas de idade a ordem é a do artigo (mais velho > semelhante > mais jovem). Ex.: 16-39: 0,21% / 0,30% / 0,91%; 80+: 12,7% / 14,5% / 19,8% (só 353 pacientes no grupo mais velho).

**O que os números mostram (sem concluir além disso):** a idade e o sexo sozinhos já dão AUROC ~0,79 para óbito em 2 anos. Acrescentar a idade do ECG melhora o AUROC em cerca de 0,01 (IC não inclui zero em 2 e 3 anos; inclui em 1 ano), e o ganho em AUPRC tem IC que inclui zero. Nas faixas de idade, só em 40-59 o ganho tem IC fora de zero; em 60-69 e 70+ não aparece ganho. Uma seed só faz sentido para a logística (determinística). B2/B3 (CNN com sinal) não foram feitos.
