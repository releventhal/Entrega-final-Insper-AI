# Roteiro da apresentação intermediária (11/11)

Rascunho do Claude do Tom (08/11), **para o Tom e a Renata revisarem**. Todos os números são da **validação** (o teste interno, o PTB-XL e o CODE-TEST ainda não foram abertos) e são **provisórios, a confirmar com a Renata**. Nenhum número foi inventado: cada um vem de `resultados/` ou do `CLAUDE.md`. Figuras em `figuras/`.

---

### Slide 1. Título
- **Ideia:** o que o ECG diz sobre o paciente, avaliado do jeito honesto.
- **Figura:** nenhuma (ou um traçado de ECG de 12 derivações).
- **Fala:** "Treinamos uma rede que lê eletrocardiogramas de 233 mil brasileiros, e a nossa obsessão foi medir o resultado sem nos enganar."

### Slide 2. Motivação: a fase 1 nos enganou
- **Ideia:** com poucos pacientes, o modelo decora as pessoas em vez de aprender a doença.
- **Figura:** duas barras, F1 macro 0,89 (split aleatório) vs 0,39 (split por paciente), MIT-BIH, 44 pacientes. (Fazer a partir de `arquivo/CLAUDE_fase1_arritmia.md` / `projeto_arritmia.ipynb`; não está em `figuras/`.)
- **Fala:** "Mesma rede, mesmos dados: só mudando a forma de separar treino e teste, o F1 caiu de 0,89 para 0,39. A sensibilidade de uma das classes foi de 0,75 para 0,05."

### Slide 3. A base nova: CODE-15%
- **Ideia:** 345.779 ECGs de 233.770 pacientes da atenção primária de Minas Gerais, com rótulos de 6 alterações e dados de óbito.
- **Figura:** nenhuma (números grandes na tela). Mencionar: usamos as partes 0 a 3 (80.000 exames) por limite de computador.
- **Fala:** "Saímos de 44 pacientes para 233 mil. E mantivemos a regra: sempre avaliar em pessoas que o modelo nunca viu."

### Slide 4. Duas perguntas
- **Ideia:** (1) Um modelo treinado em Minas funciona em pacientes novos, em outro país e contra especialistas? (2) O ECG diz algo sobre risco de morte que a idade e o sexo não dizem?
- **Figura:** nenhuma.
- **Fala:** "A primeira é sobre generalizar. A segunda é a pergunta que mais nos interessa, e a resposta 'quase nada' também vale, se for bem medida."

### Slide 5. Desenho
- **Ideia:** Parte A: CNN 1D, 12 derivações, 6 saídas sigmoid (um exame pode ter mais de uma alteração). Parte B: óbito em até 2 anos, comparando idade + sexo (B0) com idade + sexo + "idade do ECG" (B1); depois, a CNN (B2/B3, ainda não feitos).
- **Figura:** diagrama simples (sinal 4.096 × 12 → 4 camadas de convolução → 6 saídas). Não está em `figuras/`.
- **Fala:** "Sigmoid e não softmax, porque um paciente pode ter bloqueio de ramo e fibrilação ao mesmo tempo."

### Slide 6. Regras do jogo
- **Ideia:** split por paciente (70/10/20, nenhum paciente em dois grupos); nada é escolhido olhando o teste; todo modelo é comparado com um baseline simples; acurácia nunca sozinha.
- **Figura:** nenhuma.
- **Fala:** "Época, limiar e normalização foram escolhidos só na validação. O teste está trancado: o script se recusa a abri-lo até congelarmos o modelo."

### Slide 7. Acurácia engana
- **Ideia:** dizer "não" para todo mundo dá 98% de acurácia e não acha nenhuma doença.
- **Figura:** nenhuma; números do A0a na validação (8.180 exames): acurácia 0,980, sensibilidade 0, AUROC 0,500.
- **Fala:** "Por isso olhamos AUROC, AUPRC, sensibilidade e precisão de cada alteração."

### Slide 8. A CNN contra o baseline de idade + sexo
- **Ideia:** a CNN ganha com folga do modelo que só sabe idade e sexo.
- **Figura:** `figuras/b_por_alteracao_A0b_A3.png`.
- **Fala:** "Na validação de 8.158 exames, o AUROC médio foi 0,989 contra 0,705 da idade + sexo; a AUPRC, 0,691 contra 0,046. O 1dAVb é o mais difícil: AUPRC 0,44."
- **Acréscimo de 12/11 (Claude do Tom; degraus intermediários, fora das aulas; só validação, provisório):** entre A0b e a CNN entram a árvore de decisão (A0c) e a random forest (A0d) sobre 22 atributos calculados do sinal (FC, variabilidade do RR, amplitude, largura do QRS aproximada; **sem o PR**). Macro AUROC / AUPRC: A0b 0,705 / 0,046; A0c 0,911 / 0,402; A0d 0,943 / 0,531; A3 0,989 / 0,691 (`resultados/experimentos.csv`). Fala sugerida: "Atributos simples já levam o AUROC a 0,94; a CNN ainda fica bem acima em AUPRC. A floresta se aproxima da CNN em bradicardia e taquicardia, e falha no 1dAVb (AUROC 0,75 vs 0,97), justamente o que depende do PR, que não calculamos." Árvore de SB: raiz em FC ≈ 50 bpm (`figuras/arvore_SB_ecg.png`).

### Slide 9. Como chegamos ao A3 (uma mudança por vez)
- **Ideia:** A1 (sem normalização) → A2 (normaliza cada exame) → A3 (4 partes em vez de 1), nos mesmos 2.069 exames.
- **Figura:** `figuras/a_modelos_macro.png`.
- **Fala:** "AUROC 0,976 → 0,985 → 0,990 e AUPRC 0,656 → 0,682 → 0,721. Mas é uma seed só e os intervalos se sobrepõem: não dá para afirmar que cada passo ajudou." Completar com `figuras/h_seeds_A3.png`: em 3 seeds do A3, macro AUROC 0,9897 ± 0,0004, AUPRC 0,697 ± 0,007, F1 0,713 ± 0,004; nos 2.069 exames, o A2 (1 seed) fica abaixo das 3 seeds em AUROC e AUPRC e empata em F1.

### Slide 10. Curvas do A3
- **Ideia:** as 6 alterações têm curva ROC colada no canto; na precisão-recall dá para ver quem é difícil.
- **Figura:** `figuras/c_roc_pr_A3.png` (e, se couber, `figuras/e_loss_A3.png`: a perda de validação piora depois da época 4, por isso o early stopping).
- **Fala:** "A curva ROC é otimista quando a doença é rara; a precisão-recall mostra que no 1dAVb, para achar metade dos casos, mais da metade dos alarmes é falsa (precisão máxima de 0,45 com sensibilidade de pelo menos 0,5)."

### Slide 11. Onde o modelo erra (análise de erros na validação)
- **Ideia:** os erros não vêm de poucos pacientes (ao contrário da fase 1); exames com 2+ alterações erram muito mais; 1dAVb perde metade dos positivos.
- **Figura:** `figuras/d_calibracao_A3.png` (ou uma tabela curta).
- **Fala:** "Os 76 falsos negativos de 1dAVb vêm de 72 pacientes diferentes. Exames com duas ou mais alterações têm algum erro em 66% dos casos (85 exames), contra 4% nos exames sem alteração."

### Slide 12. Robustez já testada
- **Ideia:** sem vazamento achado; multiplicar o sinal por 0,45 ou 2 não muda as previsões (diferença máxima 3e-5), por causa da normalização por exame; tirar exames com derivação zerada ou duplicados não muda o resultado (≤ 0,002).
- **Figura:** nenhuma.
- **Fala:** "O PTB-XL alemão tem amplitudes cerca de metade das nossas; a normalização por exame torna o modelo indiferente a esse fator de escala. Outras diferenças entre aparelhos não estão cobertas."
- **Acréscimo de 12/11 (Fase 2 de robustez; só validação, provisório; registrado a partir dos CSVs):** teste de estresse em 8.158 exames (`figuras/robustez_02_estresse.png`): amplitude ×0,5 e ×2 não muda nada; ruído de 1,0 do desvio tira 0,0070 de AUROC macro; deriva de linha de base e derivação zerada tiram ≤ 0,0018. Ponto de atenção: mantendo só 25% do sinal (o resto vira zero), o AUROC cai só 0,0055, mas o F1 com os limiares fixos vai a 0,0043 (hipótese, não verificada: pooling diluído pelos zeros). IC do estresse com 200 reamostras. Ensemble das 3 seeds: AUROC 0,9904, AUPRC 0,7108 (A3 seed 42: 0,9892 / 0,6908). Calibração (`figuras/robustez_03_calibracao.png`) e Grad-CAM (`figuras/robustez_gradcam_logit_*.png`, 3 exemplos por alteração, sem teste de sanidade) só como diagnóstico.

### Slide 13. Parte B: o paradoxo de Simpson
- **Ideia:** sem controlar a idade, quem tem o "ECG mais velho" morre menos; dentro de cada faixa de idade, morre mais.
- **Figura:** `figuras/f_simpson.png`.
- **Fala:** "Sem controle: 3,0% de óbito no grupo com ECG 8+ anos mais velho contra 4,7% no 8+ mais jovem. Mas o primeiro grupo tem 42 anos de média e o segundo 63. Dentro de 60-69 anos: 3,2%, 3,7% e 6,4%. Toda comparação de risco tem que levar a idade em conta."

### Slide 14. Parte B: o ECG acrescenta à idade?
- **Ideia:** idade + sexo já dão AUROC 0,79 para óbito em 2 anos; a "idade do ECG" acrescenta cerca de 0,01.
- **Figura:** `figuras/g_B0_B1_faixas.png`.
- **Fala:** "Na validação (18.711 pacientes, 450 óbitos), B0 0,794 e B1 0,804; a diferença tem intervalo de 0,003 a 0,016. Por faixa de idade, o ganho só aparece com clareza entre 40 e 59 anos. A CNN direto no sinal (B2/B3) é o próximo passo."

### Slide 15. Limitações honestas
- **Ideia:** tudo é validação (que também escolheu época e limiar, então é otimista); uma seed por configuração (exceto o A3, com 3 seeds); só 4 das 18 partes; a origem dos rótulos da CODE-15% e a unidade de `timey` não estão documentadas; o CODE-TEST pode ter pacientes em comum com a CODE-15% (não dá para verificar); 1dAVb fraco.
- **Figura:** nenhuma.
- **Fala:** "Preferimos mostrar o que não sabemos a mostrar um número bonito que não se sustenta."

### Nota de 12/11 (Claude do Tom): este roteiro está anterior à intermediária
Os slides 5, 14 e 16 não mencionam a Parte B com CNN (B2z/B3/B3z, 9/11) nem B4/B4p (12/11), feitos pela Renata e registrados no `CLAUDE.md`; o texto acima ficou como estava. A versão de 14 slides da Renata (artifact) é a que vale para a apresentação.

### Slide 16. O que falta até 17/11
- **Ideia:** congelar o modelo; abrir uma vez o teste interno, o PTB-XL (principal) e o CODE-TEST (secundário); B2/B3 da Parte B; decisões em `decisoes_pendentes.md`.
- **Figura:** nenhuma.
- **Fala:** "O número que vale é o do teste, e ele ainda está trancado. Na final vamos mostrar se o modelo funciona na Alemanha e contra cardiologistas."
