# Parte A: rascunho para o paper (sessão 2 do Claude do Tom, 14/11)

> **Rascunho de apoio.** O texto final é da dupla e quem fecha o paper é a Renata. Cada número traz entre colchetes o arquivo de onde veio; todos foram conferidos por código em `resultados/sessao2_conferencia_numeros_parte_a.csv` (167 números, 166 ok) ou lidos direto do arquivo indicado. A fundamentação médica das 6 alterações está em `fundamentacao_medica.md` (Renata) e não é repetida aqui. "Validação" = conjunto que também escolheu época, hiperparâmetros e limiares, portanto **otimista**; o número honesto é o dos testes, abertos uma única vez em 08/11 com o A3 congelado.

## 1. Dados e divisão por paciente
- CODE-15% (Zenodo 10.5281/zenodo.4916206, CC BY 4.0), partes 0 a 3 dos sinais (12 derivações, 400 Hz, 4.096 amostras). Divisão **por paciente** (70/10/20, sorteio sobre a lista completa de `patient_id`, seed 42): nenhum paciente aparece em dois grupos.
- Treino 55.561 exames e validação 8.158 exames, depois de excluir só os exames com as 12 derivações zeradas [resultados/experimentos.csv, colunas n_treino e n_val do A3].
- Teste interno: 16.109 exames de 14.360 pacientes (32 de 16.141 excluídos por terem as 12 derivações zeradas) [resultados/bruto_interno_20261108_466e7a3.log].
- Testes externos: PTB-XL (Alemanha, principal; 21.799 registros de 18.869 pacientes, reamostrado de 500 para 400 Hz) e CODE-TEST (gabarito de cardiologistas, secundário; 827 exames) [CLAUDE.md, seção 4; resultados/bruto_ptbxl_20261108_466e7a3.log; resultados/bruto_code_test_20261108_466e7a3.log].

## 2. Modelos A0 a A3 (validação)
Tarefa multi-rótulo: 6 saídas sigmoid (1dAVb, RBBB, LBBB, SB, ST, AF). Limiar por alteração = F1 máximo na validação. Uma mudança por vez:

| Modelo | O que muda | Exames de validação | AUROC macro | AUPRC macro | F1 macro |
|---|---|---|---|---|---|
| A0a | sempre "não" | 8.180 | 0,500 | 0,020 | 0,000 |
| A0b | regressão logística, idade + sexo | 8.158 | 0,705 | 0,046 | 0,095 |
| A1 | CNN 1D (4 convoluções, ~406 mil parâmetros), parte 0, sem normalização | 2.069 | 0,976 | 0,656 | 0,696 |
| A2 | A1 + normalização por exame e derivação | 2.069 | 0,985 | 0,682 | 0,717 |
| A3 | A2 + partes 0 a 3 no treino | 2.069 (mesmos do A1/A2) | 0,990 | 0,721 | 0,717 |
| A3 | idem | 8.158 | 0,989 | 0,691 | 0,709 |

[resultados/experimentos.csv; linhas MACRO de "A0a sempre negativo", "A0b idade+sexo (mesmo conjunto do A3)", "A1 CNN (provisório, parte 0)", "A2 provisório ...", "A3 provisório ... | só os 2069 exames ... (limiar da validação completa)" e "A3 provisório (partes 0 a 3; ...)"]

- A0a tem acurácia 0,980 e sensibilidade 0: acurácia engana com doenças raras [resultados/experimentos.csv].
- A3 por alteração (AUROC / AUPRC / F1): 1dAVb 0,972 / 0,436 / 0,485; RBBB 0,995 / 0,851 / 0,826; LBBB 0,994 / 0,844 / 0,798; SB 0,993 / 0,618 / 0,658; ST 0,995 / 0,695 / 0,783; AF 0,986 / 0,700 / 0,701 [resultados/experimentos.csv]. O 1dAVb é o mais difícil.
- Três seeds do A3 (42, 43, 44): AUROC 0,9897 ± 0,0004, AUPRC 0,697 ± 0,007, F1 0,713 ± 0,004 (média ± desvio, ddof 1) [resultados/experimentos.csv, linhas "seed 43" e "seed 44"]. Com uma seed por passo e intervalos sobrepostos, **não dá para afirmar que cada passo A1 → A2 → A3 ajudou**.
- Modelo escolhido para o teste (decisão da dupla, 08/11): A3 seed 42, sem extensões [congelado_candidato_A3.json].

## 3. Testes do A3 congelado (abertos uma única vez, 08/11)
Mesmo modelo, mesmos limiares, mesma normalização; nada foi escolhido olhando estes números.

| Conjunto | AUROC macro (IC 95%) | AUPRC macro (IC 95%) | F1 macro (IC 95%) |
|---|---|---|---|
| Teste interno (CODE-15%) | 0,989 (0,987-0,991) | 0,710 (0,686-0,733) | 0,705 (0,687-0,722) |
| PTB-XL (principal) | 0,970 (0,968-0,973) | 0,788 (0,776-0,799) | 0,746 (0,735-0,756) |
| CODE-TEST (secundário) | 0,998 (0,996-0,999) | 0,916 (0,870-0,960) | 0,812 (0,747-0,865) |

[resultados/avaliacao_interno_20261108_466e7a3.csv; resultados/avaliacao_ptbxl_20261108_466e7a3.csv; resultados/avaliacao_code_test_20261108_466e7a3.csv; IC por bootstrap de pacientes (exames no CODE-TEST), 1.000 reamostras]

- No PTB-XL a queda se concentra em SB (AUROC 0,920) e 1dAVb (0,924); RBBB, LBBB, ST e AF ficam em 0,990 ou mais [resultados/avaliacao_ptbxl_20261108_466e7a3.csv]. A sensibilidade do 1dAVb fica perto de 0,50 nos três testes [mesmos arquivos].
- Análise posterior ao teste, exploratória (não muda o modelo): os SB do PTB-XL com frequência entre 55 e 59,9 bpm recebem quase todos nota baixa; **hipótese apoiada**: o rótulo SB da CODE-15% corresponde a FC < 50 e o `SBRAD` do PTB-XL inclui 50-60 bpm; o critério oficial da CODE-15% é **não verificado** [RELATORIO_TESTES.md, seção 7].
- CODE-TEST, F1 macro: A3 0,812 contra residentes de cardiologia 0,862, residentes de emergência 0,829 e estudantes 0,817, com todas as diferenças de IC incluindo zero; a comparação com os dois cardiologistas é circular (o gabarito é a concordância deles) [RELATORIO_TESTES.md, seção 7]. AF tem só 13 positivos no CODE-TEST [resultados/avaliacao_code_test_20261108_466e7a3.csv].

## 4. Robustez (só validação, só inferência; o modelo não muda)
- **Estresse** (IC com 200 reamostras de pacientes): multiplicar a amplitude por 0,5 ou 2 não muda nenhuma nota (efeito da normalização por exame); ruído de 1,0 desvio tira 0,0070 do AUROC macro (IC 0,0050 a 0,0094); zerar a V1, a pior derivação, tira 0,0011 [resultados/robustez_02_estresse.csv].
- **Recorte:** mantendo só 25% do sinal (o resto vira zero), o AUROC macro cai 0,0055 (IC 0,0039 a 0,0073), mas o F1 com os limiares fixos vai a 0,004 (1.059 decisões trocadas); com 50%, F1 0,315 [resultados/robustez_02_estresse.csv]. A ordem das notas quase se mantém; as notas é que ficam abaixo dos limiares.
- **Pooling/preenchimento:** recalculando a média do pooling só sobre os trechos com sinal (fora do modelo), a fração de positivos acima do limiar no recorte de 25% volta a 45% (1dAVb), 93% (RBBB), 82% (LBBB), 67% (AF), contra 45%, 95%, 83% e 67% no sinal original; em SB (46% contra 68%) e ST (58% contra 87%) a volta é parcial [resultados/diagnostico_pooling.csv]. **Hipótese sustentada, não provada** (zeros diluem o pooling global); sem IC; outras causas não testadas.
- **Ensemble das 3 seeds:** AUPRC macro 0,7108 contra 0,6908 da seed 42 (a pior das três; sobre a média das seeds o ganho é de cerca de 0,014) [resultados/robustez_01_ensemble.csv]. Fica como extensão; o modelo testado é a seed 42.

## 5. Baselines de árvore e floresta (fora das aulas)
Atributos calculados do sinal (frequência, variabilidade do RR, amplitude, largura do QRS aproximada) e, no A0e, o `p_r_peak` do neurokit2 (**não é o PR clínico**: vai do pico da P ao pico do R).

| Modelo | AUROC macro | AUPRC macro | F1 macro |
|---|---|---|---|
| A0c árvore (só ECG) | 0,911 | 0,402 | 0,484 |
| A0d floresta (só ECG) | 0,943 | 0,531 | 0,557 |
| **A0e-1 floresta (+ `p_r_peak`; baseline oficial)** | 0,976 | 0,567 | 0,600 |
| A3 (CNN) | 0,989 | 0,691 | 0,709 |

[resultados/experimentos.csv]

- No 1dAVb, a floresta passa de AUROC 0,749 (A0d) para 0,942 (IC 0,922-0,957) com o `p_r_peak`, e a AUPRC de 0,053 para 0,262 (0,204-0,337); o A3 continua acima (0,972 / 0,436) [resultados/arvore_rf_pr_ic.csv; resultados/experimentos.csv].
- Apêndice: em validação cruzada de 5 folds agrupada por paciente (treino + validação), a A0e-1 dá AUROC macro 0,975 (dp 0,001) e AUPRC 0,557 (dp 0,020), igual ao split único [resultados/sgpu_rf_A0e1_groupkfold_resumo.csv].

## 6. Resultados negativos (redes maiores e ensemble)
- **A6 (CNN + LSTM bidirecional, fora das aulas)**, early stopping completo (8 épocas, melhor 5): AUROC macro 0,912, AUPRC 0,445, F1 0,500, abaixo do A3 com IC da diferença abaixo de zero (AUPRC −0,245; IC −0,278 a −0,213) [resultados/noite_A6_completo_avaliacao.csv].
- **A5 (ResNet 1D pequena, ~3,4 milhões de parâmetros, fora das aulas)**, early stopping completo (12 épocas, melhor 9; treino interrompido e retomado na CPU) [resultados/A5_completo_log_por_epoca_consolidado.csv]: AUROC macro 0,989 contra 0,989 do A3 (diferença −0,000; IC −0,003 a +0,002) e AUPRC 0,705 contra 0,691 (diferença +0,014; IC −0,006 a +0,035, inclui zero) [resultados/noite_A5_completo_avaliacao.csv]. **Empate**: pela regra combinada antes (AUPRC macro acima do A3 com IC acima de zero), o A5 não substitui o A3. A única diferença fora de zero é em AF (AUPRC 0,797 contra 0,700; diferença +0,097, IC +0,045 a +0,155) [mesmo arquivo], uma alteração entre seis, numa validação que também escolheu a época: não é evidência de ganho geral.
- **Ensemble A3 + A6** (média das notas): AUPRC macro 0,676 contra 0,691 do A3 (diferença −0,015; IC −0,029 a −0,001): piora [resultados/sgpu_ensemble_A3_A6.csv]. Uma frase de apêndice.

## 7. Pontos de operação (ilustração de triagem; validação, limiar escolhido nela)
Se a rede fosse usada para priorizar laudos na telessaúde de Minas Gerais, o ponto de corte seria escolhido pela especificidade tolerada. Na validação, exigindo especificidade de 99%, o A3 acharia 48% dos 1dAVb e 75% das AF; com 95%, de 88% (1dAVb) a 100% (ST) [resultados/sessao2_pontos_operacao.csv; figuras/sessao2_pontos_operacao.png]. Na mesma tabela, o A5 aparece como `A5_completo_parcial14`, que tem previsões idênticas às do A5 final (mesmo checkpoint da época 9; diferença máxima 0,0). **São números ilustrativos e otimistas** (limiar escolhido na própria validação), não números de teste.

## 8. Limitações
- Tudo o que não é a seção 3 é validação, que também escolheu épocas, hiperparâmetros e limiares (otimista).
- Uma seed por configuração, exceto o A3 (3 seeds); só 4 das 18 partes no treino da Parte A.
- Origem exata dos rótulos da CODE-15% não documentada (provável: laudo de cardiologista + classificador de texto, como na coorte CODE; ver `fundamentacao_medica.md`).
- Definição de SB possivelmente diferente entre CODE-15% e PTB-XL (hipótese apoiada, seção 3); 1dAVb com sensibilidade perto de 0,5.
- CODE-TEST sem identificador de paciente: sobreposição com a CODE-15% não verificável; AF com 13 positivos.
- Modelos abertos pré-treinados (ECG-JEPA, ST-MEM) usaram a CODE-15 no pré-treino (confirmado nos papers) e o ECG-FM usou o PTB-XL: compará-los nas nossas bases exigiria cuidado com vazamento [literatura_modelos_abertos.md].
- Grad-CAM e calibração só como diagnóstico; o mecanismo da queda no recorte é hipótese.
