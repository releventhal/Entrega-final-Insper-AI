# Parte B: rascunho para o paper (Claude da Renata, 15/11)

> **Rascunho de apoio, mesmo formato do `parte_a_rascunho.md` do Tom.** O texto final é da dupla. Cada número traz entre colchetes o arquivo de onde veio. As seções 1 a 8 são **validação** (que também escolheu a época das CNNs e o modelo final, portanto otimista). **O resultado principal é o do TESTE, na seção 9** (aberto uma vez em 15/11, com o B3z congelado). A fundamentação médica está em `fundamentacao_medica.md`.

## 1. Pergunta e desenho
- **Pergunta:** o ECG prevê morte em até 2 anos **além** do que a idade e o sexo já dizem? Quanto disso são as 6 alterações conhecidas (Parte A)?
- **Dados:** CODE-15% (Zenodo 10.5281/zenodo.4916206, CC BY 4.0), **as 18 partes** dos sinais. Só o **primeiro exame** de cada paciente, o único com `death` e `timey` [CLAUDE.md, seção 4].
- **Regra do horizonte (2 anos, declarado antes de rodar, 08/11):** entra quem morreu até 2 anos ou foi acompanhado os 2 anos inteiros; quem ficou vivo e foi acompanhado menos tempo sai [parte_b_tabular.py, `rotular`].
- **Divisão por paciente** (a mesma da Parte A, 70/10/20, seed 42). Exames com as 12 derivações zeradas saem, decidido pelo sinal (269 no treino e 56 na validação) [CLAUDE.md, log de 14/11].
- **Elegíveis:** treino 129.221 pacientes (3.356 óbitos, 2,6%); validação 18.665 (450 óbitos, 2,4%) [resultados/parte_b_cnn_p0-17.csv, colunas n_treino, obitos_treino, n_val, obitos_val].
- **IC 95%:** bootstrap de pacientes, 1.000 reamostras, seed 42, **pareado** (as mesmas reamostras para todos os modelos, o que permite o IC da diferença) [parte_b_cnn.py, `bootstrap`].

## 2. Modelos (uma mudança por vez)

| Modelo | O que usa | Como |
|---|---|---|
| B0 | idade + sexo | regressão logística |
| B1 | B0 + idade do ECG de Lima et al. (`nn_predicted_age`) e a diferença para a idade real | regressão logística |
| B4 | B0 + as 6 alterações do **laudo** (0/1) | regressão logística |
| B4p | B0 + as 6 notas do **A3 congelado** (logit) | regressão logística |
| B2t | as 128 características do A3 congelado (transfer learning), só o ECG | camada final (= regressão logística) |
| B3 | B2t + idade + sexo | idem |
| B3z | CNN do zero (a arquitetura do A3) com idade e sexo juntados depois do pooling, treinada de ponta a ponta | Keras, GPU L4, 10 épocas em 51,6 min, melhor época 7 pelo AUROC de validação |
| B5 | B3z com as convoluções começando de uma CNN pré-treinada para prever a idade | idem, 8 épocas, melhor época 5 |

[parte_b_cnn.py; parte_b_alteracoes.py; parte_b_idade.py; CLAUDE.md, log de 14/11]

## 3. Resultado principal (validação, 18 partes)

| Modelo | AUROC (IC 95%) | AUPRC (IC 95%) |
|---|---|---|
| B0 | 0,794 (0,773-0,813) | 0,082 (0,070-0,096) |
| B1 | 0,803 (0,784-0,822) | 0,086 (0,073-0,102) |
| B4 | 0,808 (0,787-0,827) | 0,095 (0,080-0,115) |
| B4p | 0,818 (0,799-0,837) | 0,110 (0,092-0,133) |
| B2t | 0,735 (0,711-0,759) | 0,076 (0,063-0,093) |
| B3 | 0,824 (0,806-0,843) | 0,113 (0,094-0,136) |
| **B3z** | **0,831 (0,812-0,850)** | **0,123 (0,101-0,151)** |

| Diferença | AUROC (IC 95%) | AUPRC (IC 95%) |
|---|---|---|
| B3z − B0 | +0,036 (+0,024 a +0,049) | +0,041 (+0,024 a +0,064) |
| B3z − B1 | +0,027 (+0,016 a +0,040) | +0,037 (+0,017 a +0,058) |
| B3z − B4 | +0,023 (+0,012 a +0,035) | +0,028 (+0,010 a +0,048) |
| **B3z − B4p** | **+0,013 (+0,003 a +0,023)** | +0,013 (−0,002 a +0,029) |
| B4 − B0 | +0,013 (+0,007 a +0,019) | +0,013 (+0,003 a +0,027) |
| B4p − B0 | +0,023 (+0,013 a +0,033) | +0,028 (+0,014 a +0,045) |
| B4p − B4 | +0,010 (+0,003 a +0,017) | +0,015 (+0,004 a +0,028) |
| B3z − B3 | +0,007 (−0,003 a +0,017) | +0,010 (−0,009 a +0,031) |

[resultados/parte_b_cnn_p0-17.csv; resultados/parte_b_alteracoes_p0-17.csv; AUROC recalculado das notas salvas em projeto_ecg.ipynb, seção 8.5]

**Leitura:** o ECG acrescenta à idade e ao sexo (+0,036), e a CNN fica acima do modelo que só usa as 6 alterações vistas pelo A3 (+0,013, IC fora de zero em AUROC, não em AUPRC). As notas contínuas do A3 valem mais que o "sim/não" do laudo (+0,010). Sem idade e sexo, o transfer do A3 é fraco (B2t 0,735).

## 4. Mais dados mudaram a conclusão
- Com **8 partes** (66.417 / 9.549 pacientes, 1.680 / 215 óbitos): B3z 0,805 contra B4p 0,802, **B3z − B4p +0,003 (IC −0,014 a +0,020)**: não dava para dizer que a CNN via algo além das 6 alterações [resultados/parte_b_alteracoes_p0-7.csv].
- Com **18 partes:** +0,013 (IC +0,003 a +0,023) [resultados/parte_b_alteracoes_p0-17.csv].
- Nos 9.549 pacientes de validação em comum, o B3z com 18 partes ganha do B3z com 8 partes: +0,013 (IC +0,0003 a +0,029) [resultados/parte_b_comparar.csv].

## 5. Risco escondido e triagem (validação, 18 partes)
- **Sem nenhuma das 6 alterações no laudo** (16.832 pacientes, 347 óbitos): B0 0,802; B4p 0,812 (+0,010; IC +0,001 a +0,017); **B3z 0,829 (+0,026; IC +0,013 a +0,040)**. O B4 é igual ao B0 por construção.
- **Laudo "ECG normal"** (6.594 pacientes, 103 óbitos): B0 0,802; B3z 0,803 (+0,001; IC −0,020 a +0,024): **sem ganho**. Leitura: o risco "além das 6" vem de pacientes com outras anormalidades no laudo, não de ECGs normais.
- **Triagem** (fração dos óbitos em 2 anos entre os X% de maior nota, todos os 18.665):

| Chamando os | B0 | B4 | B4p | B3z | B3z − B0 (IC 95%) |
|---|---|---|---|---|---|
| 5% | 20,7% | 22,7% | 28,2% | 30,7% | +0,100 (+0,059 a +0,149) |
| 10% | 37,1% | 40,4% | 42,2% | 47,8% | +0,107 (+0,059 a +0,154) |
| 20% | 63,8% | 66,9% | 64,7% | 67,8% | +0,040 (+0,002 a +0,085) |

[resultados/parte_b_analises_p0-17.csv]

## 6. Por faixa de idade (validação, 18 partes; calculado no notebook)

| Faixa | Pacientes | Óbitos | AUROC B0 | AUROC B4p | AUROC B3z | B3z − B0 (IC 95%) | B3z − B4p (IC 95%) |
|---|---|---|---|---|---|---|---|
| < 40 | 5.982 | 24 | 0,589 | 0,684 | 0,769 | +0,180 (+0,046 a +0,329) | +0,085 (+0,005 a +0,174) |
| 40-59 | 5.541 | 66 | 0,650 | 0,735 | 0,759 | +0,109 (+0,040 a +0,183) | +0,023 (−0,026 a +0,070) |
| 60-69 | 2.739 | 64 | 0,668 | 0,715 | 0,754 | +0,086 (+0,004 a +0,165) | +0,039 (−0,010 a +0,088) |
| 70+ | 4.403 | 296 | 0,619 | 0,663 | 0,685 | +0,066 (+0,030 a +0,102) | +0,022 (−0,006 a +0,051) |

[projeto_ecg.ipynb, seção 8.7, a partir de resultados/parte_b_cnn_p0-17_previsoes_validacao.csv e parte_b_alteracoes_p0-17_previsoes_validacao.csv]. 4 faixas × 2 comparações: é pista, não conclusão. Com 8 partes, o ganho sobre o B0 só aparecia abaixo de 60 anos.

## 7. Sobrevivência (Kaplan-Meier, fora das aulas; todos os 23.513 pacientes de validação, inclusive censurados)
- Sobrevida em 5 anos por quartil da nota, Q1 → Q4: B0 0,996 / 0,985 / 0,964 / 0,874; B3z 0,997 / 0,991 / 0,964 / 0,867 [resultados/parte_b_km_p0-17.csv; figuras/parte_b_km_p0-17.png].
- Curvas quase iguais por quartil, porque a vantagem do B3z está no topo (5-10%). O Q4 do B3z tem mais óbitos (600 contra 573) e idade média menor (73,4 contra 76,2) [idem].

## 8. Razões de chance das alterações (B4 sem penalização, ajustado por idade e sexo, treino com 18 partes; IC por 500 bootstraps)
idade 1,85 por década (1,80-1,91); homem 1,63 (1,52-1,74); **ST 2,91 (2,48-3,38); AF 2,53 (2,13-2,88); LBBB 1,53 (1,25-1,79)**; 1dAVb 1,16 (0,98-1,47); RBBB 1,12 (0,97-1,36); SB 0,66 (0,31-0,73) [resultados/parte_b_alteracoes_or_p0-17.csv].
- **Comparação com a coorte CODE inteira** (Paixão et al., *Hearts* 2021, 1,56 milhão de pacientes, Cox ajustado por idade, sexo e comorbidades): FA HR 2,10; BRE 1,69; BRD 1,32; BAV 1º grau com pior sobrevida [fundamentacao_medica.md]. A ordem FA > BRE > BRD bate; medidas diferentes (OR em 2 anos × HR em todo o seguimento) e sem ajuste por comorbidades.
- SB com OR < 1 e ST com OR alto: explicação **não verificada** (hipóteses em fundamentacao_medica.md).

## 9. Idade pelo ECG, pré-treino e o modelo final
- **Idade pelo ECG** (mesma CNN, saída numérica, 163.137 pacientes de treino): MAE 8,08 anos (r 0,849) contra 8,43 (r 0,839) do `nn_predicted_age`, nos 23.513 pacientes de validação; diferença −0,35 (IC −0,43 a −0,27) [resultados/parte_b_idade_p0-17.csv]. Não é comparação justa para "superar" Lima et al. (treinamos nesta amostra; para eles o erro é a informação).
- **B5:** 0,830 na validação completa contra 0,831 do B3z (−0,001; IC −0,010 a +0,009) [CLAUDE.md, log de 14/11]: o pré-treino não ajudou.
- **Regra do modelo final (registrada em 14/11, antes dos resultados da GPU):** maior AUROC na validação, nos 9.549 pacientes em comum: B3z 18 partes 0,818; B5 0,815; B3z 8 partes 0,805 → **B3z com 18 partes** [resultados/parte_b_comparar.csv]. Status: **candidato** [congelado_candidato_parte_b.json].
- **Seeds 43 e 44 do B3z (estabilidade, validação):** AUROC 0,831 / 0,828 / 0,830 (seeds 42 / 43 / 44; média 0,830, desvio 0,001); B3z − B4p +0,013 (+0,003 a +0,023) / +0,011 (+0,003 a +0,019) / +0,012 (+0,004 a +0,021) [resultados/parte_b_seeds_p0-17.csv].

## 9b. TESTE da Parte B (aberto UMA vez, 15/11; B3z p0-17 congelado)
Congelado depois da confirmação de Renata e Tom [congelado_candidato_parte_b.json, status "congelado"]. B0, B1, B4 e B4p ajustados só no treino. Antes, o mesmo script reproduziu as notas salvas da validação [resultados/parte_b_colab_teste_p0-17.log]. Teste: **36.880 pacientes, 895 óbitos em 2 anos** (46.672 primeiros exames; 94 todo-zero e 9.698 censurados fora). Rótulos e grupo recalculados de forma independente a partir do `exams.csv`: iguais; 0 pacientes em comum com treino/validação.

| Modelo | AUROC (IC 95%) | AUPRC (IC 95%) |
|---|---|---|
| B0 | 0,793 (0,779-0,806) | 0,089 (0,079-0,101) |
| B1 | 0,804 (0,790-0,817) | 0,096 (0,086-0,110) |
| B4 | 0,804 (0,791-0,817) | 0,101 (0,089-0,115) |
| B4p | 0,819 (0,806-0,831) | 0,113 (0,100-0,129) |
| **B3z** | **0,832 (0,820-0,844)** | **0,128 (0,114-0,146)** |
| B3z − B0 | +0,039 (+0,030 a +0,049) | +0,039 (+0,027 a +0,053) |
| B3z − B1 | +0,028 (+0,020 a +0,037) | +0,031 (+0,017 a +0,045) |
| B3z − B4 | +0,028 (+0,019 a +0,036) | +0,027 (+0,015 a +0,040) |
| **B3z − B4p** | **+0,013 (+0,006 a +0,019)** | **+0,014 (+0,005 a +0,024)** |
| B1 − B0 | +0,011 (+0,007 a +0,015) | +0,007 (+0,002 a +0,015) |
| B4 − B0 | +0,012 (+0,008 a +0,016) | +0,012 (+0,004 a +0,022) |
| B4p − B0 | +0,026 (+0,019 a +0,033) | +0,025 (+0,014 a +0,036) |
| B4p − B4 | +0,015 (+0,010 a +0,020) | +0,013 (+0,004 a +0,021) |

[resultados/parte_b_teste_p0-17.csv; AUROC recalculado de resultados/parte_b_teste_bruto_p0-17.csv]

- **Risco escondido (teste):** sem nenhuma das 6 no laudo (33.152 pacientes, 659 óbitos): B3z 0,829, B3z − B0 +0,030 (+0,020 a +0,039); laudo "ECG normal" (12.804, 150 óbitos): B3z 0,807, **B3z − B0 +0,023 (+0,006 a +0,040)**, diferente da validação (+0,001; −0,020 a +0,024), com intervalos que se sobrepõem [resultados/parte_b_teste_analises_p0-17.csv].
- **Triagem (teste):** óbitos nos 5% / 10% / 20% de maior nota: B0 23,2% / 40,1% / 60,9%; B4p 30,5% / 46,6% / 65,5%; **B3z 30,3% / 48,8% / 68,6%**; B3z − B0 nos 10%: +0,087 (+0,056 a +0,120) [idem].
- **Leitura:** o teste reproduz a validação (0,831 → 0,832), e o ganho além das 6 alterações agora tem IC fora de zero em AUROC e AUPRC.

## 10. Limitações
- Seções 1-8 são validação (que escolheu a época e o modelo final); o teste (9b) foi aberto uma vez e confirma.
- O modelo testado é a seed 42; as seeds 43 e 44 só foram avaliadas na validação.
- O ganho "além das 6 alterações" é pequeno (+0,013) e só apareceu com as 18 partes.
- Óbito por cruzamento probabilístico de registros, sem causa da morte; `timey` sem unidade documentada (compatível com anos).
- Uma população só (atenção primária de Minas Gerais); a Parte B não tem teste externo.
- Rótulos do laudo: processo descrito para a coorte CODE (Paixão et al. 2021); que a CODE-15% seguiu o mesmo processo é provável, não verificado.
