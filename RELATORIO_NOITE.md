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

## Bloco 2, análise de erros do A3 (seed 42) na validação (10:24-10:28)
**Descritivo, posterior ao treino, não muda o modelo.** Script `analise_erros_A3.py`; saídas em `resultados/erros_A3/`. Modelo `modelos/A3_melhor.keras`, limiares lidos do JSON (o JSON não foi alterado). Mesma validação do A3: 8.158 exames de 7.206 pacientes. As previsões recalculadas com o modelo são idênticas às salvas (diferença máxima 0).

**1) Contagens (VP / FP / FN / VN):** 1dAVb 65 / 62 / 76 / 7.955; RBBB 216 / 80 / 11 / 7.851; LBBB 109 / 34 / 21 / 7.994; SB 80 / 45 / 38 / 7.995; ST 150 / 62 / 21 / 7.925; AF 117 / 42 / 58 / 7.941. O 1dAVb erra mais por perder positivos (76 FN de 141) e o AF também (58 FN de 175).

**2) Subgrupos (sensibilidade / precisão, `resultados/erros_A3/subgrupos.csv`):**
- 1dAVb por idade: < 40: 0,31 / 0,40 (13 positivos); 40-59: 0,59 / 0,53 (17); 60-69: 0,48 / 0,48 (29); 70+: 0,45 / 0,54 (82). Por sexo: mulheres 0,54 / 0,58 (72); homens 0,38 / 0,43 (69).
- AF: < 40: 0,50 / 0,80 (só 8 positivos); 70+: 0,68 / 0,74 (123).
- SB 70+: sensibilidade 0,54 (35 positivos) contra 0,77-0,79 em < 40 e 60-69.
- RBBB fica entre 0,94 e 0,97 de sensibilidade em todas as faixas e nos dois sexos.
- Com 2 a 30 positivos por célula, essas diferenças são muito incertas (sem IC calculado por subgrupo).

**3) Concentração de erros por paciente:** na validação quase todo paciente tem 1 exame (8.158 exames, 7.206 pacientes), então os erros **não se concentram** em poucas pessoas. Ex.: os 76 FN do 1dAVb vêm de 72 pacientes (máximo 2 por paciente); metade deles vem de 34 pacientes (se cada erro fosse de um paciente diferente seriam 38). Os 58 FN de AF vêm de 52 pacientes (metade em 23; o mínimo possível seria 29). Nenhum paciente tem mais de 2 erros do mesmo tipo. Diferente da fase 1 (75% dos S do teste de uma gravação).

**4) Confusões e exames com mais de um rótulo:**
- Entre exames com a alteração da linha e sem a da coluna, a fração prevista como a coluna é baixa (≤ 2,3%): LBBB→RBBB 1,6%; RBBB→LBBB 0,4%; SB→ST 0%; ST→SB 0%; 1dAVb→LBBB 2,3%; RBBB→SB 2,3%. Taxa de "positivo" em exames sem nenhuma das 6: 0,4% a 1,0% por alteração. Ou seja, as alterações quase não se confundem entre si; os FP vêm sobretudo de exames sem nenhuma das 6.
- Exames com algum erro: 4,0% dos 7.287 sem nenhuma alteração; 21,4% dos 786 com uma; **65,9% dos 85 com duas ou mais**. Os exames multi-rótulo erram muito mais (são raros: 1% da validação).

**5) 1dAVb (o mais fraco):** quantis das notas (5%, 25%, 50%, 75%, 95%): positivos 0,028 / 0,178 / 0,431 / 0,669 / 0,899; negativos 0,000 / 0,000 / 0,0001 / 0,002 / 0,082. Limiar 0,489: a mediana dos positivos fica abaixo do limiar, por isso a sensibilidade é ~0,46. Erros por idade: < 40: 9 FN de 13; 40-59: 7 de 17; 60-69: 15 de 29; 70+: 45 de 82 (FP: 6, 9, 15, 32).

**6) Calibração e F1 contra o limiar** (`figuras/d_calibracao_A3.png`, `resultados/erros_A3/f1_vs_limiar.csv`): nas faixas de nota com 10+ exames a curva segue a diagonal de forma grosseira; 1dAVb fica abaixo dela de 0,4 para cima (nota 0,80-0,90: 50% de positivos), ou seja, o modelo é "confiante demais" no 1dAVb. As faixas do meio têm 8 a 60 exames, então a curva é ruidosa. O F1 é plano perto do máximo: fica a até 0,05 do máximo numa faixa larga de limiares (ex.: RBBB de 0,16 a 0,56; LBBB de 0,20 a 0,74; 1dAVb de 0,14 a 0,56). A escolha exata do limiar pesa pouco.

**7) Com e sem exames especiais (macro AUROC / AUPRC / F1):** todos (8.158): 0,9892 / 0,6908 / 0,7087; sem os 21 de derivação parcialmente zerada: 0,9893 / 0,6907 / 0,7091; sem os 56 de sinal idêntico a outro exame de treino ou validação (8 deles idênticos a um de treino): 0,9893 / 0,6914 / 0,7104; sem os dois: 0,9893 / 0,6912 / 0,7106. Diferenças ≤ 0,002: esses exames **não mudam** o resultado. Desses, 2 parciais e 6 idênticos têm alguma das 6 alterações. (Hash MD5 do sinal bruto; os sinais de treino foram lidos só nas linhas de treino.)

**8) Estresse de escala (sinal bruto × 0,45 e × 2, depois a normalização do A2):** diferença máxima das notas: × 0,45 → 3,3e-5 (média 2,3e-8), 1 decisão trocada em 48.948 (uma nota colada no limiar); × 2 → 0 (multiplicar por 2 é exato em ponto flutuante). Como esperado, a normalização por exame torna o modelo **insensível a um fator de escala** no sinal inteiro. Isso NÃO cobre diferenças de forma, de ruído, de offset por derivação ou de filtros entre aparelhos (T8 continua aberta para isso).

## Bloco 3, robustez do A3 entre seeds (seed 43: 10:18-14:32; seed 44: iniciada 14:33)
Mesma configuração do A3 (arquitetura, normalização do A2, partes 0 a 3, mesma exclusão dos todo-zero, limiar F1 máximo na validação, early stopping paciência 3, limite de 90 min e 12 épocas), mudando **só a seed** (pesos iniciais e ordem do embaralhamento). `a1_cnn.py --seed N`; modelo em `modelos/A3_seed43/` (o modelo da seed 42 não foi tocado; SHA256 conferido no fim).

**O que aconteceu com o tempo (importante):**
- Seed 43: épocas 1 a 7 levaram 174-309 s cada (melhor época = 7, val_loss 0,0294, checkpoint gravado às 10:45). A **época 8 levou 13.630 s (3,8 h)**: o computador parou entre 10:45 e 14:32. Causa **não verificada** (hipótese: suspensão do Windows ou falta de memória; o Claude Code avisou "system is running low on memory" e encerrou um processo de espera meu). O limite de 90 min disparou ao fim da época 8 e o treino parou. Ou seja, **a seed 43 terminou pelo limite de tempo, não pelo early stopping**: depois da melhor época (7) só rodou 1 época, então não sabemos se ela melhoraria nas épocas 9 a 12. O modelo usado é o da época 7 (o melhor salvo), como no A3. Tempo registrado no CSV: 254,4 min (inclui a parada).
- Por causa da parada, a sessão perdeu ~3h45 de orçamento. A fila de treinos (`logs/fila_treinos.sh`) iniciou a seed 44 às 14:33 (antes do corte de 14:47, permitido pela regra) e eu interrompi a fila logo depois, para que **nada mais começasse** (seed 45 e A4 não foram iniciados). Resultado da seed 44: ver o fim deste relatório.

**Resultado com 2 seeds (42 e 43), validação completa de 8.158 exames** (média; desvio padrão amostral; valores de cada seed), `resultados/seeds_A3.csv`, `figuras/h_seeds_A3.png`:
- Macro AUROC 0,9895 (desvio 0,0005; 0,9892 e 0,9899). Macro AUPRC 0,6974 (0,0093; 0,6908 e 0,7040). Macro F1 0,7109 (0,0032; 0,7087 e 0,7132).
- Por alteração (AUROC / AUPRC / F1, média): 1dAVb 0,974 / 0,444 / 0,504; RBBB 0,995 / 0,850 / 0,831; LBBB 0,994 / 0,847 / 0,791; SB 0,993 / 0,640 / 0,649; ST 0,995 / 0,689 / 0,771; AF 0,987 / 0,715 / 0,721. O AUROC quase não varia entre as seeds (desvio ≤ 0,002); a AUPRC e o F1 variam mais nas alterações difíceis (SB: AUPRC 0,618 e 0,662; AF: F1 0,701 e 0,741; 1dAVb: F1 0,485 e 0,524).
- **A2 → A3 contra o desvio entre seeds** (nos mesmos 2.069 exames da parte 0, limiar da validação completa de cada modelo): A2 (1 seed) 0,985 / 0,682 / 0,717; A3 seed 42 0,990 / 0,721 / 0,717; A3 seed 43 0,990 / 0,726 / 0,730. A diferença A2 → A3 em AUROC (+0,005) e AUPRC (+0,04) é maior que a diferença entre as duas seeds do A3 (AUROC 0,0006; AUPRC 0,005). **O que isso permite concluir:** pouco. São só 2 seeds (o desvio de 2 valores é muito instável), e o A2 tem uma seed só, então não sabemos quanto o A2 varia de seed para seed (com 1 parte de treino ele pode variar mais). No F1 a diferença A2 → A3 (0 a +0,013) é do tamanho da variação entre seeds. Não é um teste estatístico.
- Linhas acrescentadas em `resultados/experimentos.csv` (experimento "A3 provisório ... | seed 43").

## Bloco 4, A4 (perda ponderada, fora das aulas): **não rodado**
Implementado e conferido em `a1_cnn.py` (`--peso-por-classe`; peso = negativos/positivos de cada alteração, só no treino, limitado a 10; com peso 1 a perda é idêntica à `binary_crossentropy`), mas **não treinado**: o orçamento acabou por causa da parada do computador. Comando para rodar depois (≈ 30-40 min): `.venv\Scripts\python a1_cnn.py treinar A4 --normalizar --partes 0,1,2,3 --max-min 90 --max-epocas 12 --peso-por-classe` (salva `modelos/A4_melhor.keras`, não toca no A3 nem no JSON). Observação: com a perda ponderada, o early stopping monitora a perda ponderada da validação.

## Bloco 5, figuras e roteiro (10:25-10:31 e 14:34)
`figuras.py` gera em `figuras/` (PNG, ~0,6 MB no total): `a_modelos_macro.png` (A0b/A1/A2/A3 nos mesmos 2.069 exames), `b_por_alteracao_A0b_A3.png`, `c_roc_pr_A3.png`, `d_calibracao_A3.png`, `e_loss_A3.png`, `f_simpson.png`, `g_B0_B1_faixas.png`, `h_seeds_A3.png` (2 seeds). Cores das alterações = as do notebook (seção 1). `roteiro_slides.md`: 16 slides com título, ideia, figura e fala; todos os números conferidos nos CSVs (o único número que eu tinha tirado "a olho" de um gráfico, a precisão do 1dAVb com sensibilidade de 0,5, foi recalculado: 0,45). O slide 2 (fase 1) e o diagrama da CNN (slide 5) não têm figura pronta.

## Bloco 6, documentos finais e conferência (14:35)
- `decisoes_pendentes.md`: 9 itens (congelar o A3; A4; limite de `likelihood`; exclusão AFLT/IRBBB/ILBBB; registros do PTB-XL; exames zerados nos testes; horizonte da Parte B; número de partes; outros). Nada decidido.
- `CLAUDE.md`: bloco "Atualização de 08/11" na seção 0, status da T7, scripts novos na seção 6 e duas linhas no log (seção 10). Nada da Renata foi apagado.
- **SHA256 final (iguais aos iniciais):**
  - `modelos/A3_melhor.keras`: `bae64fa994934072434853ce6c5f6dfffd485a1f126336be1d6e8d526ca12a89`
  - `congelado_candidato_A3.json`: `1c145b6a553a47b53a6330f4de44608c8e54aa251183627d7607673a97ecf935`
- Nenhum arquivo apagado; split e lista dos 411 exames zerados intactos (a lista só foi lida).
