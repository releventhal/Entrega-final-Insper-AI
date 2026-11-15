# Fundamentação médica (rascunho de apoio para o paper; o texto final é da dupla)

> Escrito pelo Claude da Renata em 14/11/2026. Cada afirmação tem a fonte ao lado. **Lido na íntegra** = o texto completo foi aberto e lido nesta data; **só resumo/busca** = só o resumo ou trechos de busca foram vistos, então conferir antes de citar. Nada aqui foi inventado; o que não tem fonte está marcado como "não verificado".

## 1. O problema real em Minas Gerais

- A **Rede de Teleassistência de Minas Gerais (RTMG/TNMG)** liga as unidades de atenção primária a cardiologistas dos hospitais universitários. O profissional local faz o ECG digital e o envia pela internet, e um cardiologista lauda à distância. Segundo Paixão et al. (2021, lido na íntegra), a rede cobria **817 dos 853 municípios** de Minas Gerais e cerca de 400 em outros estados, e já tinha recebido **mais de 5 milhões de ECGs**. Alkmim et al. (Bull. WHO 2012; só resumo/busca) descrevem a origem da rede (financiada pelo governo do estado a partir de 2005).
- Na coorte CODE, cada ECG foi laudado **por um único cardiologista**, entre 14 cardiologistas treinados (Paixão et al. 2021). Em escala de milhões de exames, um apoio automático pode ajudar a **priorizar** a fila (quem precisa de atenção antes), não a substituir o médico. Essa é a moldura prática do nosso projeto: triagem e apoio, não diagnóstico autônomo.
- A **CODE-15%** (nossa base) é uma amostra de 15% dessa coorte (Ribeiro et al. 2020; Lima et al. 2021).

## 2. De onde vêm os rótulos (fecha em parte uma pendência da seção 9 do CLAUDE.md)

- Na coorte CODE, Paixão et al. (2021, lido na íntegra) descrevem o processo assim. O laudo do cardiologista é texto livre, e um classificador de texto (rede recorrente) reconhece nele as alterações. Dois programas automáticos (Glasgow e Minnesota) também interpretam o traçado. O diagnóstico é aceito sem revisão quando o laudo concorda com um dos dois programas; os casos em que só o cardiologista ou só os dois programas apontam a alteração são revisados manualmente. O classificador de texto teve F1 de 87,0% para FA, 90,9% para BRD, 88,6% para BRE e 83,6% para BAV, contra 4.557 laudos rotulados por dois cardiologistas.
- **Ressalva:** isso está descrito para as alterações daquele artigo (FA, bloqueios de ramo, BAV, pré-excitação). Que os 6 rótulos da CODE-15% (incluindo SB e ST) seguiram exatamente o mesmo processo é **provável, mas não verificado** (ver Ribeiro et al. 2020, métodos, ainda não relido).
- Consequência para nós: os rótulos de treino **não são um gabarito perfeito** (por isso o CODE-TEST, com vários médicos, é o gabarito).

## 3. As 6 alterações, em linguagem simples

Definições clássicas de livro-texto (critérios da AHA/ACC/HRS, Kligfield et al. 2007, citado por Paixão et al.; **os números de corte abaixo são os usuais na clínica, mas não foram conferidos no texto das diretrizes nesta sessão**):

| Rótulo | O que é | Como aparece no ECG |
|---|---|---|
| 1dAVb (BAV de 1º grau) | O impulso demora mais para passar do átrio para o ventrículo | Intervalo PR > 200 ms, toda onda P é seguida de QRS |
| RBBB (bloqueio de ramo direito) | O ramo direito do sistema de condução não conduz; o ventrículo direito é ativado "por tabela" | QRS ≥ 120 ms, padrão rSR' em V1 |
| LBBB (bloqueio de ramo esquerdo) | O mesmo no ramo esquerdo | QRS ≥ 120 ms, R larga em V6, S profunda em V1 |
| SB (bradicardia sinusal) | Ritmo normal, mas lento | FC < 60 bpm (corte clássico; ver nota abaixo sobre < 50) |
| ST (taquicardia sinusal) | Ritmo normal, mas rápido | FC > 100 bpm |
| AF (fibrilação atrial) | Os átrios "tremem" de forma desorganizada | Sem onda P, intervalos RR irregulares |

**Nota sobre SB (achado nosso, posterior ao teste, seção 0 do CLAUDE.md):** os SB da CODE-15% têm mediana de 47 bpm e 89% abaixo de 50; no PTB-XL, 30% dos `SBRAD` ficam entre 50 e 60. Isso sugere que o laudo da CODE marca SB com corte perto de 50 bpm, enquanto o PTB-XL usa 60. Hipótese apoiada pelos dados, **critério oficial não verificado**.

## 4. Por que essas alterações importam para o risco de morte

**Evidência na própria população de Minas (Paixão et al., Hearts 2021, lido na íntegra):** 1.558.421 pacientes adultos da TNMG (2010-2017), seguimento médio de 3,7 anos, mortalidade de 3,3%. Modelo de Cox ajustado por idade, sexo e comorbidades autorrelatadas (tabagismo, hipertensão, diabetes, dislipidemia, doença de Chagas, infarto prévio, DPOC); grupo de comparação = pacientes **sem** as alterações maiores. Mortalidade por todas as causas:
- **FA: HR 2,10 (2,03-2,17)**, a maior entre as alterações estudadas;
- **BRE: HR 1,69 (1,62-1,76)**;
- **BRD: HR 1,32 (1,27-1,36)**;
- **BAV de 1º grau: sobrevida relativa 0,76 (0,71-0,81)** (modelo log-normal, porque o de Cox não serviu; RS < 1 = pior sobrevida);
- pré-excitação ventricular: HR 1,41 (0,56-3,57), sem associação; BAV Mobitz I: sem associação.
- SB e ST **não foram estudadas** nesse artigo.

**Comparação com o nosso B4 (razões de chance, regressão logística no treino da Parte B, partes 0 a 7, óbito em 2 anos, ajustado só por idade e sexo; seção 10 do CLAUDE.md, 12/11):** FA 2,59; BRE 1,57; BAV 1º grau 1,49; BRD 1,33 (0,99-1,56); ST 2,59; SB 0,40. **A ordem e o tamanho batem com a coorte inteira** (FA > BRE > BRD; BAV associado a pior sobrevida), mesmo com outra medida (razão de chances em 2 anos × razão de riscos em todo o seguimento) e sem ajuste por comorbidades. Isso é uma boa checagem de sanidade da nossa Parte B.
- **ST (taquicardia) com OR 2,59 e SB com OR 0,40:** não estão no artigo acima. Hipótese (não verificada): a taquicardia em repouso acompanha doenças graves (infecção, insuficiência cardíaca, anemia, DPOC); a bradicardia com corte perto de 50 bpm pode marcar pessoas fisicamente ativas. Precisa de fonte antes de ir para o paper.

**FA fora do Brasil (Odutayo et al., BMJ 2016, meta-análise de 104 coortes; só resumo/busca):** a FA se associou a risco cerca de 2 vezes maior de morte cardiovascular, 2,3 vezes maior de AVC e 5 vezes maior de insuficiência cardíaca. O número "5 vezes mais AVC", comum em textos antigos, vem de estudos mais velhos (Framingham). **Conferir no artigo antes de citar.**

## 5. IA prevendo morte direto do ECG (literatura; o que já existe)

- **Raghunath et al., Nat. Med. 2020** ("Prediction of mortality from 12-lead electrocardiogram voltage data using a deep neural network"; Geisinger, EUA): rede neural prevendo morte em 1 ano direto do traçado, com bom desempenho **mesmo em ECGs laudados como normais** (como resumido por Paixão et al. 2021, que o citam). Isso confirma a lembrança registrada na seção 2 do CLAUDE.md ("trabalho americano"). **Números do artigo não verificados.**
- **Lima et al., Nat. Commun. 2021:** a "idade do ECG" (rede treinada nos outros 85% da CODE) prevê mortalidade; já descrito no CLAUDE.md.
- **O que é nosso:** pergunta feita **nesta base brasileira de atenção primária**, comparando lado a lado (i) idade + sexo, (ii) a idade do ECG, (iii) as 6 alterações do laudo, (iv) as 6 notas do nosso A3 e (v) uma CNN treinada para o óbito, com a regra do horizonte e IC pareado por paciente. Resultado com as 18 partes (validação, 18.665 pacientes, 450 óbitos; `resultados/parte_b_cnn_p0-17.csv`, `parte_b_alteracoes_p0-17.csv`, `parte_b_analises_p0-17.csv`): a CNN com idade e sexo (B3z) tem AUROC 0,831 contra 0,794 do B0 (+0,036; IC 0,024 a 0,049) e fica **acima** do modelo que só usa as 6 notas do A3 (B4p 0,818; +0,013; IC 0,003 a 0,023). Nos 16.832 pacientes sem nenhuma das 6 alterações no laudo, o B3z ainda ganha do B0 (+0,026; IC 0,013 a 0,040): **há informação de risco no ECG além das 6 alterações**. Nos pacientes com laudo "ECG normal" (6.594, 103 óbitos), o ganho não aparece (+0,001; IC −0,020 a 0,024). Com só 8 partes nada disso era mensurável (B3z − B4p +0,003, IC incluindo zero): mais dados mudaram a conclusão. **No teste (aberto uma vez em 15/11, 36.880 pacientes, 895 óbitos; `resultados/parte_b_teste_p0-17.csv`):** B3z 0,832 contra B0 0,793 (+0,039; IC 0,030 a 0,049) e contra B4p 0,819 (+0,013; IC 0,006 a 0,019); nos sem nenhuma das 6, +0,030 (0,020 a 0,039); no laudo "ECG normal", +0,023 (0,006 a 0,040), o que a validação não mostrava.

## 6. Uso prático possível (moldura para a apresentação, sem exagero)

- **Triagem da fila de laudos e de acompanhamento:** nos 10% de pacientes de validação com maior nota, o B3z (18 partes) reúne 47,8% dos óbitos em 2 anos, contra 37,1% do B0 (só idade e sexo; diferença +0,107, IC 0,059 a 0,154) e 42,2% do B4p (as 6 alterações vistas pelo A3); `resultados/parte_b_analises_p0-17.csv`. Ou seja, um ECG barato de atenção primária pode ajudar a decidir quem chamar primeiro; parte do ganho vem de alterações que o cardiologista já laudaria e parte vem de algo além delas.
- Paixão et al. (2021) defendem o mesmo papel do ECG: estratificar risco na atenção primária e priorizar encaminhamentos para centros secundários e terciários.
- **No teste** (15/11): os 10% de maior nota do B3z reúnem 48,8% dos óbitos em 2 anos, contra 40,1% do B0 (+0,087; IC 0,056 a 0,120) e 46,6% do B4p (`resultados/parte_b_teste_analises_p0-17.csv`).
- **Limites (para não vender demais):** o teste é da mesma população (sem teste externo da Parte B); o óbito vem de linkage probabilístico com o sistema de mortalidade (pode errar); não sabemos a causa da morte na CODE-15%; o modelo nunca foi testado fora de Minas para a Parte B.

## Fontes
- Paixão GMM, Lima EM, Gomes PR, et al. Electrocardiographic Predictors of Mortality: Data from a Primary Care Tele-Electrocardiography Cohort of Brazilian Patients. *Hearts* 2021;2(4):35. https://www.mdpi.com/2673-3846/2/4/35 (lido na íntegra em 14/11/2026).
- Alkmim MB, et al. Improving patient access to specialized health care: the Telehealth Network of Minas Gerais, Brazil. *Bull World Health Organ* 2012;90:373-378. https://pmc.ncbi.nlm.nih.gov/articles/PMC3341691 (só resumo/busca).
- Odutayo A, et al. Atrial fibrillation and risks of cardiovascular disease, renal disease, and death: systematic review and meta-analysis. *BMJ* 2016 (só resumo/busca).
- Raghunath S, et al. Prediction of mortality from 12-lead electrocardiogram voltage data using a deep neural network. *Nat Med* 2020;26:886-891 (citado por Paixão et al.; não lido).
- Ribeiro AH, et al. Automatic diagnosis of the 12-lead ECG using a deep neural network. *Nat Commun* 2020;11:1760.
- Lima EM, et al. Deep neural network-estimated electrocardiographic age as a mortality predictor. *Nat Commun* 2021;12:5117.
- Kligfield P, et al. Recommendations for the Standardization and Interpretation of the Electrocardiogram. *Circulation* 2007;115:1306-1324 (citado por Paixão et al.; não lido).
