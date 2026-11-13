# Decisões pendentes (Tom e Renata)

Organizado pelo Claude do Tom em 08/11/2026. **Nada aqui foi decidido.** Cada item traz a proposta, a evidência medida (só treino e validação), o risco e o que muda se a escolha for outra. Números de validação são otimistas, porque a própria validação escolheu épocas e limiares.

## 1. Congelar o A3 como modelo final da Parte A?
- **Proposta:** congelar o A3 (seed 42), como está em `congelado_candidato_A3.json`, e só então abrir o teste interno, o PTB-XL e o CODE-TEST, uma única vez.
- **Evidência:** validação de 8.158 exames: macro AUROC 0,989, AUPRC 0,691, F1 0,709 (A0b: 0,705 / 0,046 / 0,095). Três seeds (42, 43, 44): macro AUROC 0,9897 ± 0,0004, AUPRC 0,697 ± 0,007, F1 0,713 ± 0,004 (a seed 43 parou pelo limite de tempo depois de uma parada do computador; ver `RELATORIO_NOITE.md`, Bloco 3). O modelo da seed 42 não é o melhor nem o pior: está dentro da variação. A4: não treinado. Sem vazamento achado (seção 9 do `CLAUDE.md`). Teste de estresse de escala e análise de erros: Bloco 2.
- **Risco:** depois de abrir o teste, qualquer troca de modelo deixa o número do teste "usado" (regra 3). Se o A4 (ou outra mudança) for melhor, tem que ser escolhido **antes**, só com a validação.
- **Se escolher diferente:** congelar o A4 (se houver) exige gerar um JSON novo com o caminho, SHA256 e limiares do A4 (o JSON atual aponta para o A3). Treinar mais (ex.: mais partes) atrasa o teste externo e a intermediária de 11/11.

## 2. Alternativa A4 (perda ponderada, fora das aulas)
- **Proposta:** tratar como candidato; decidir entre A3 e A4 só pela validação, olhando também a variação entre seeds do A3.
- **Evidência:** nenhuma ainda: o A4 foi implementado e conferido, mas **não treinado** (o orçamento da sessão de 08/11 acabou). Comando no `RELATORIO_NOITE.md`, Bloco 4.
- **Risco:** com uma seed, uma diferença menor que o desvio entre seeds do A3 não é diferença demonstrada. A perda ponderada muda a escala das notas (calibração), mas os limiares são reescolhidos na validação.
- **Se escolher diferente:** ficar com o A3 mantém o modelo mais simples e dentro das aulas.

## 3. Limite de `likelihood` do PTB-XL
- **Proposta (só pela documentação):** positivo = o código aparece em `scp_codes` com qualquer valor (`presente`).
- **Evidência:** documentação do PhysioNet e de Wagner et al. 2020: para ritmo, `likelihood` 0 = "sem informação". Contagens (seção 4 do `CLAUDE.md`): SB 637 / 0 / 0 / 0 positivos com `presente` / `>0` / `>=50` / `=100`; ST 826 / 4 / 4 / 1; AF 1.514 / 48 / 48 / 48. 1dAVb, RBBB e LBBB quase não mudam (1dAVb 793 / 793 / 790 / 779).
- **Risco:** "código ausente" = negativo confirmado nos rótulos de ritmo é **não verificado**.
- **Se escolher diferente:** com `>0` ou mais, SB fica com 0 positivos e ST/AF quase sem positivos: essas três alterações não podem ser avaliadas no PTB-XL.

## 4. Exclusão de AFLT, IRBBB e ILBBB só do rótulo correspondente (PTB-XL)
- **Proposta:** registro com AFLT sai só da avaliação de AF; IRBBB só de RBBB; ILBBB só de LBBB (padrão provisório no JSON).
- **Evidência:** AFLT 73 registros, IRBBB 1.118, ILBBB 77; com a exclusão, avaliados: RBBB 20.681 (540 positivos), LBBB 21.722 (536), AF 21.726 (1.497).
- **Risco:** se forem contados como negativos, o modelo pode ser "punido" por dizer RBBB num bloqueio incompleto, que é parecido; se forem positivos, misturamos uma classe que a CODE não tem.
- **Se escolher diferente:** contar como negativos aumenta os falsos positivos aparentes de RBBB (1.118 registros com IRBBB).

## 5. Quais registros do PTB-XL avaliar
- **Proposta (padrão do JSON):** todos os 21.799.
- **Evidência:** os folds 9 e 10 passaram por revisão humana (rótulos melhores), segundo a documentação.
- **Risco:** todos os registros = mais positivos, rótulos menos revisados; só folds 9-10 = rótulos melhores, ~1/5 dos positivos (contagem por fold: **não medida**).
- **Se escolher diferente:** reportar os dois (todos e folds 9-10) é possível, mas precisa ser declarado antes de abrir.

## 6. Regra dos exames zerados nos testes
- **Proposta (padrão do JSON):** no teste interno, tirar só os exames com as 12 derivações zeradas (mesma regra do treino/validação do A3; 32 de 16.141); parciais ficam. PTB-XL: 0 todo-zero, 1 com uma derivação zerada (fica). CODE-TEST: 14 com alguma derivação zerada (ficam).
- **Evidência (Bloco 2, validação):** desempenho com e sem os exames de derivação parcialmente zerada: ver `RELATORIO_NOITE.md`.
- **Risco:** tirar exames do teste por um critério do sinal é aceitável se decidido antes; decidir depois de ver o teste não é.
- **Se escolher diferente:** manter os todo-zero no teste interno dá nota quase 0 nesses 32 exames (o modelo nunca viu exame todo-zero).

## 7. Horizonte da Parte B
- **Proposta:** 2 anos (declarado antes de rodar o Bloco 1), com 1 e 3 anos como sensibilidade.
- **Evidência (Bloco 1, validação):** 2 anos: 18.711 elegíveis, 450 óbitos; B0 AUROC 0,794, B1 0,804. 1 ano: 251 óbitos; 3 anos: 585 óbitos, mas 9.088 pacientes saem por censura na validação (39% dos que têm `death`).
- **Risco:** trocar o horizonte depois de ver qual dá o maior ganho seria escolher pelo resultado.
- **Se escolher diferente:** 3 anos tem mais óbitos e menos pacientes, com seleção mais forte (só quem foi acompanhado 3 anos); 1 ano tem poucos óbitos (IC largos).

## 8. Quantas partes no resultado final
- **Proposta:** partes 0 a 3 (80.000 exames), como o A3.
- **Evidência:** A3 (4 partes) vs A2 (1 parte), nos mesmos 2.069 exames: AUROC 0,990 vs 0,985, AUPRC 0,721 vs 0,682, F1 0,717 vs 0,717 (uma seed cada); entre as 3 seeds do A3, nos mesmos 2.069 exames, a faixa foi 0,0006 em AUROC, 0,009 em AUPRC e 0,019 em F1 (o A2 não tem várias seeds). Tempo: ~37 min por treino de 4 partes neste computador.
- **Risco:** mais partes = mais disco (3,9 GB cada), mais tempo, e a Renata precisaria das mesmas partes para reproduzir.
- **Se escolher diferente:** mais partes exigem retreinar e reescolher limiares antes de congelar.

## 9. Outros pontos ainda abertos (do `CLAUDE.md`, seção 5.6)
- Normalizar ou não entre bases (T8): o A2/A3 normaliza cada exame por si mesmo; o estresse de escala do Bloco 2 mostra quanto isso protege contra diferença de amplitude.
- Proporção do split (70/10/20, provisória): **não refazer** o split agora.

## 10. Como incluir árvore/floresta (A0c/A0d) no teste: estender a trava do `avaliar_externo.py` (acrescentado pelo Claude do Tom em 12/11)
**Nada decidido; `avaliar_externo.py` NÃO foi editado.** Só proposta.
- **Situação:** `congelado_candidato_arvore_rf.json` (status "candidato") guarda os 4 modelos `.joblib` com SHA256, hiperparâmetros, limiares (F1 máximo na validação) e versões dos pacotes. O `avaliar_externo.py` só sabe carregar CNN Keras (confere SHA256 do `.keras`, normaliza o sinal, chama `predict`), então **não leria esse JSON mesmo que o status fosse "congelado"**.
- **Proposta de extensão (texto, sem código aplicado):** (a) aceitar `modelo.tipo = "sklearn_joblib"` no JSON; (b) nesse tipo, conferir o SHA256 do `.joblib` como já faz com o Keras e **não normalizar** o sinal; (c) rodar `atributos_sinal.atributos` sobre o sinal **cru** e montar a tabela na ordem de `atributos` do JSON (NaN aceito); (d) aplicar os limiares do JSON e reaproveitar `bootstrap`, a exclusão dos exames todo-zero e a trava existente (status "congelado" **e** `--confirmo-congelado`). No PTB-XL, calcular os atributos depois de `preprocessar_ptbxl` (400 Hz, FS = 400); no CODE-TEST, depois de reordenar as derivações.
- **Evidência (só validação):** A0d floresta macro AUROC 0,943-0,950, AUPRC 0,531-0,537, F1 0,556-0,557 contra A3 0,989 / 0,691 / 0,709 (`resultados/experimentos.csv`). A floresta só supera o A3 em AUPRC na ST (0,712-0,722 vs 0,695).
- **Risco:** (1) abrir o teste com mais um modelo depois de o A3 já ter sido testado: o teste dessas versões novas não é "uma vez só" para a Parte A, e isso deve ser declarado (ver proposta do feedback de 11/11 no log do `CLAUDE.md`); (2) escolher entre A3 e floresta olhando o teste seria violar a regra 3; (3) o PR não é calculado, então o 1dAVb tende a ser pior; (4) os atributos precisam ser calculados igual nas 3 bases (largura do QRS é aproximação).
- **Se escolher diferente:** não testar a floresta no PTB-XL e usá-la só como degrau explicativo da escadinha na validação (sem custo de código).

## 11. Fase 2 de robustez: o que fazer com ensemble, calibração e o script de GPU (acrescentado em 12/11; registrado a partir dos scripts e CSVs da sessão paralela do Tom)
**Nada decidido.** Todos os números são de validação (otimistas).
- **Ensemble das 3 seeds como modelo final?** Evidência: macro AUROC 0,9904, AUPRC 0,7108, F1 0,7205 contra 0,9892 / 0,6908 / 0,7087 da seed 42; diferença de AUPRC +0,0200 (IC 0,0106 a 0,0298) e de F1 +0,0118 (IC −0,0004 a 0,0242) (`resultados/robustez_01_ensemble_diferencas.csv`). Risco: o A3 seed 42 já foi testado e as seeds 43 e 44 nunca; testar o ensemble reabre o teste para uma versão nova, o que deve ser declarado, e exigiria o mesmo tipo de extensão da trava do item 10 (vários modelos no JSON). Escolher entre A3 e ensemble olhando o teste violaria a regra 3. **Ressalva sobre o ganho:** o +0,0200 de AUPRC é sobre a seed 42, que é a pior das 3 (0,6908 contra 0,7040 e 0,6961); sobre a média das 3 seeds (0,6970) o ganho é de cerca de **+0,014** (0,7108 − 0,6970; conta simples, sem IC).
- **Aplicar calibração (Platt ou isotônica)?** Evidência: ganho de Brier com IC acima de zero só em 1dAVb e ST (Platt; no ST o limite inferior é 0,000007, praticamente zero), e pequeno (`resultados/robustez_03_calibracao.csv`). A isotônica reduz o AUROC (a causa "por empates" é hipótese não verificada; a Platt também reduz o AUROC em 1dAVb e RBBB). Mudaria as notas e obrigaria a reescolher os limiares antes de congelar. Proposta: manter como diagnóstico.
- **Recorte do sinal (estresse):** mantendo só 25% do sinal (o resto vira zero) o F1 com os limiares do JSON cai a 0,0043, e mantendo só 50% (o resto vira zero) a 0,3150, embora o AUROC quase não mude (`resultados/robustez_02_estresse.csv`). A explicação (pooling diluído pelos zeros) é hipótese não verificada. Decisão em aberto: se exames de duração curta importam (o CODE-TEST tem sinais de 7,3 s com zeros dos dois lados, como a CODE-15%), vale testar a hipótese só na validação antes de congelar qualquer coisa. Observação: o IC do estresse usou 200 reamostras, não 1.000.
- **Rodar o `robustez_06_treino_gpu.py`** (A4, seeds extras do A1/A2, augmentation, k-fold agrupado por paciente) em GPU externa, que o Tom está pesquisando? Custo: horas de treino por configuração, não medido. **Estimativa a partir da CPU, não medida em GPU, preço da GPU não pesquisado:** cerca de 58 mil lotes no conjunto todo (4 runs de A1/A2, o A4 e 5 folds, ~10 épocas cada), o que dá ~4 h com 1 worker de leitura (~0,26 s por lote, a leitura do HDF5 é o gargalo) ou ~1 a 1,5 h com `--workers 4`. O teste de fumaça (1 lote, 1 época, CPU) rodou sem erro em 3 configurações (A4 com augment, A1 seed 43, k-fold fold 0). Risco do k-fold: é extensão de protocolo (o split oficial 70/10/20 segue valendo para os números oficiais) e deve ser declarada. Qualquer modelo novo escolhido deve ser decidido **só pela validação**, antes de abrir o teste.
