# Relatório dos testes do A3 congelado (08/11/2026, Claude do Tom)

**O que é:** o modelo A3 (seed 42, `modelos/A3_melhor.keras`, SHA256 `bae64fa9…`), congelado em `congelado_candidato_A3.json` (status "congelado", `congelado_em` 2026-11-08T22:23:17-03:00, `congelado_commit` b80883a), aplicado **uma vez** em cada conjunto de teste, nesta ordem: teste interno, PTB-XL, CODE-TEST. Sem retreinar, com os limiares, o pré-processamento, as exclusões e o limite de `likelihood` do JSON. Nada disso foi mudado depois de ver resultado.

**Pré-condições conferidas antes de rodar:** status "congelado" com data, commit e SHA256; nenhum "a decidir" no JSON; SHA256 atual do modelo = o do JSON; árvore git limpa; commit do congelamento (82a8e3e) no `origin`. Observação: os campos de texto `aviso` e `provisorio` do JSON ainda dizem "NAO congelado" / "ainda candidato" (texto antigo, não atualizado pelo `congelar.py`); o campo que a trava lê (`status`) diz "congelado". O JSON não foi alterado.

**Mudança de código antes de rodar (só saída):** `avaliar_externo.py` (commit 466e7a3) passou a gravar a saída bruta (`resultados/bruto_<conjunto>_<data>_<commit>.npz`: notas do modelo, rótulos, máscara de avaliação, ids, pacientes, resumo, SHA256 do JSON e do modelo) logo depois do `predict` e antes do bootstrap, com escrita atômica; se esse arquivo já existir, o script não refaz o `predict`. Modelo, limiares, pré-processamento e exclusões não mudaram. Motivo: sem as notas salvas, a análise posterior pediria um segundo `predict` no teste.

**Saídas brutas (em `resultados/`):** `bruto_<conjunto>_20261108_466e7a3.npz` (notas), `bruto_<conjunto>_20261108_466e7a3.log` (saída do terminal) e `avaliacao_<conjunto>_20261108_466e7a3.csv` (métricas e IC). Também ficam `avaliacao_<conjunto>.csv`, que o script escreve. `avaliacao_validacao_A3.csv` não foi tocado.

**Métricas:** AUROC e AUPRC (não dependem de limiar); sensibilidade, precisão, F1 e especificidade com o limiar congelado de cada alteração. IC de 95% por bootstrap (1.000 reamostras, seed 42), por paciente no teste interno e no PTB-XL, por exame no CODE-TEST (sem identificador). As tabelas abaixo são geradas por `analise_testes.py tabela <conjunto>` a partir dos CSVs (4 casas, arredondadas para 3; pode haver diferença de 0,001 em relação ao log, que arredonda direto de float).

## Referência: validação do A3 (8.158 exames, a mesma que escolheu a época e os limiares; otimista)

| Alteração | Positivos | AUROC | AUPRC | Sensibilidade | Precisão | F1 | Especificidade |
|---|---|---|---|---|---|---|---|
| 1dAVb | 141 | 0,972 [0,961-0,981] | 0,436 [0,358-0,524] | 0,461 [0,372-0,546] | 0,512 [0,422-0,605] | 0,485 [0,409-0,557] | 0,992 [0,990-0,994] |
| RBBB | 227 | 0,995 [0,993-0,997] | 0,851 [0,802-0,893] | 0,952 [0,923-0,977] | 0,730 [0,679-0,782] | 0,826 [0,789-0,861] | 0,990 [0,988-0,992] |
| LBBB | 130 | 0,994 [0,990-0,997] | 0,844 [0,783-0,897] | 0,839 [0,770-0,895] | 0,762 [0,695-0,833] | 0,798 [0,742-0,851] | 0,996 [0,994-0,997] |
| SB | 118 | 0,993 [0,990-0,995] | 0,618 [0,531-0,709] | 0,678 [0,592-0,759] | 0,640 [0,557-0,721] | 0,658 [0,590-0,725] | 0,994 [0,993-0,996] |
| ST | 171 | 0,995 [0,994-0,996] | 0,695 [0,620-0,777] | 0,877 [0,832-0,923] | 0,708 [0,644-0,768] | 0,783 [0,737-0,825] | 0,992 [0,990-0,994] |
| AF | 175 | 0,986 [0,978-0,992] | 0,700 [0,624-0,776] | 0,669 [0,592-0,740] | 0,736 [0,662-0,802] | 0,701 [0,635-0,755] | 0,995 [0,993-0,996] |
| Média macro |  | 0,989 [0,987-0,991] | 0,691 [0,664-0,724] | 0,746 [0,718-0,773] | 0,681 [0,651-0,711] | 0,709 [0,685-0,733] | 0,993 [0,993-0,994] |

## 1. Teste interno (CODE-15%, grupo teste das partes 0 a 3)

- **Rodado em 08/11, uma vez, sem erro** (exit 0; 1.000 reamostras válidas). Nenhuma nova execução.
- **Exames:** 16.141 no grupo teste; **32 excluídos** (12 derivações zeradas, pelo sinal); avaliados **16.109 exames de 14.360 pacientes**.

| Alteração | Positivos | AUROC | AUPRC | Sensibilidade | Precisão | F1 | Especificidade |
|---|---|---|---|---|---|---|---|
| 1dAVb | 281 | 0,978 [0,972-0,983] | 0,477 [0,415-0,545] | 0,502 [0,438-0,561] | 0,551 [0,488-0,614] | 0,525 [0,469-0,580] | 0,993 [0,991-0,994] |
| RBBB | 442 | 0,992 [0,989-0,995] | 0,826 [0,788-0,871] | 0,919 [0,893-0,944] | 0,740 [0,704-0,776] | 0,819 [0,795-0,846] | 0,991 [0,989-0,992] |
| LBBB | 290 | 0,993 [0,990-0,996] | 0,784 [0,733-0,838] | 0,841 [0,798-0,880] | 0,728 [0,678-0,779] | 0,781 [0,744-0,817] | 0,994 [0,993-0,996] |
| SB | 308 | 0,992 [0,989-0,994] | 0,693 [0,631-0,752] | 0,692 [0,638-0,745] | 0,666 [0,613-0,717] | 0,678 [0,631-0,721] | 0,993 [0,992-0,995] |
| ST | 353 | 0,993 [0,990-0,995] | 0,718 [0,664-0,772] | 0,844 [0,807-0,881] | 0,659 [0,614-0,703] | 0,740 [0,706-0,772] | 0,990 [0,989-0,992] |
| AF | 310 | 0,984 [0,975-0,991] | 0,761 [0,715-0,802] | 0,661 [0,604-0,713] | 0,712 [0,661-0,758] | 0,686 [0,640-0,724] | 0,995 [0,994-0,996] |
| Média macro |  | 0,989 [0,987-0,991] | 0,710 [0,686-0,733] | 0,743 [0,724-0,761] | 0,676 [0,654-0,697] | 0,705 [0,687-0,722] | 0,993 [0,992-0,993] |
| Validação A3, média macro (8.158 exames) | | 0,989 [0,987-0,991] | 0,691 [0,664-0,724] | 0,746 [0,718-0,773] | 0,681 [0,651-0,711] | 0,709 [0,685-0,733] | 0,993 [0,993-0,994] |

**Leitura:** em pacientes novos da mesma base, o resultado da validação se repete (macro AUROC 0,989 nos dois; AUPRC 0,710 vs 0,691; F1 0,705 vs 0,709; ICs sobrepostos). Os limiares escolhidos na validação não "decoraram" a validação. 1dAVb continua o ponto fraco (sensibilidade 0,50).

## 2. PTB-XL (teste externo PRINCIPAL, Alemanha)

- **Rodado em 08/11, uma vez, sem erro** (exit 0; 1.000 reamostras válidas). Nenhuma nova execução. Era o primeiro uso do caminho `ptbxl` com dados reais: nenhum bug de integração apareceu.
- **Registros:** 21.799; **0 excluídos** por terem as 12 derivações zeradas (igual à medição anterior); avaliados **21.799 registros de 18.869 pacientes**. Limite de `likelihood` "presente"; exclusão por rótulo ligada: **1.118 registros (IRBBB) fora só de RBBB, 77 (ILBBB) fora só de LBBB, 73 (AFLT) fora só de AF**; 1.264 registros com algum desses três códigos. Positivos após a exclusão: RBBB 540 (541 sem ela), AF 1.497 (1.514 sem ela).

| Alteração | Positivos | AUROC | AUPRC | Sensibilidade | Precisão | F1 | Especificidade |
|---|---|---|---|---|---|---|---|
| 1dAVb | 793 | 0,924 [0,916-0,933] | 0,400 [0,363-0,439] | 0,503 [0,466-0,541] | 0,412 [0,381-0,445] | 0,453 [0,424-0,483] | 0,973 [0,971-0,975] |
| RBBB | 540 | 0,997 [0,993-0,999] | 0,951 [0,928-0,971] | 0,959 [0,941-0,977] | 0,892 [0,864-0,916] | 0,924 [0,907-0,940] | 0,997 [0,996-0,998] |
| LBBB | 536 | 0,998 [0,996-0,999] | 0,942 [0,926-0,958] | 0,965 [0,947-0,980] | 0,673 [0,636-0,710] | 0,793 [0,763-0,819] | 0,988 [0,986-0,990] |
| SB | 637 | 0,920 [0,907-0,933] | 0,671 [0,630-0,708] | 0,557 [0,517-0,596] | 0,833 [0,794-0,867] | 0,668 [0,631-0,699] | 0,997 [0,996-0,997] |
| ST | 826 | 0,993 [0,991-0,995] | 0,878 [0,853-0,900] | 0,881 [0,859-0,903] | 0,757 [0,727-0,785] | 0,814 [0,793-0,833] | 0,989 [0,987-0,990] |
| AF | 1497 | 0,990 [0,987-0,992] | 0,886 [0,866-0,905] | 0,810 [0,787-0,832] | 0,839 [0,817-0,861] | 0,824 [0,807-0,842] | 0,989 [0,987-0,990] |
| Média macro |  | 0,970 [0,968-0,973] | 0,788 [0,776-0,799] | 0,779 [0,767-0,790] | 0,734 [0,721-0,747] | 0,746 [0,735-0,756] | 0,989 [0,988-0,989] |
| Validação A3, média macro (8.158 exames) | | 0,989 [0,987-0,991] | 0,691 [0,664-0,724] | 0,746 [0,718-0,773] | 0,681 [0,651-0,711] | 0,709 [0,685-0,733] | 0,993 [0,993-0,994] |

**Leitura:** em outro país, o macro AUROC cai de 0,989 para 0,970 (ICs não se sobrepõem). A queda vem de **duas alterações: 1dAVb (0,972 → 0,924) e SB (0,993 → 0,920)**; RBBB, LBBB, ST e AF ficam iguais ou acima da validação. AUPRC e F1 macro ficam **acima** da validação, mas isso não é "melhor": AUPRC e F1 dependem da prevalência, e no PTB-XL cinco das seis alterações são mais frequentes que na validação (1,6 a 3,2 vezes; AF 6,9% vs 2,1%, 1dAVb 3,6% vs 1,7%; RBBB fica igual, 2,6% vs 2,8%). RBBB, que tem a mesma prevalência, também sobe em AUPRC (0,851 → 0,951), então a prevalência não explica tudo; o resto é **não verificado**. AUROC e especificidade não dependem da prevalência e são as comparações mais limpas entre bases.

## 3. CODE-TEST (teste SECUNDÁRIO, gabarito de especialistas)

- **Rodado em 08/11, uma vez, sem erro** (exit 0; 1.000 reamostras válidas). Nenhuma nova execução. Primeiro uso do caminho `code_test` com dados reais: nenhum bug de integração apareceu.
- **Exames:** 827; **0 excluídos** (nenhum com as 12 derivações zeradas); avaliados **827**. Bootstrap por exame (não há `patient_id`).
- **Limitações:** AF tem **só 13 positivos** e SB 16, então os ICs são largos (ex.: sensibilidade de AF 0,43 a 0,92). A **sobreposição de pacientes com a CODE-15% é não verificada** (o CODE-TEST não tem identificador); se houver pacientes do nosso treino aqui, o resultado fica otimista.

| Alteração | Positivos | AUROC | AUPRC | Sensibilidade | Precisão | F1 | Especificidade |
|---|---|---|---|---|---|---|---|
| 1dAVb | 28 | 0,997 [0,993-0,999] | 0,920 [0,829-0,981] | 0,500 [0,318-0,708] | 0,933 [0,778-1,000] | 0,651 [0,465-0,818] | 0,999 [0,996-1,000] |
| RBBB | 34 | 0,995 [0,987-1,000] | 0,952 [0,887-0,995] | 0,941 [0,844-1,000] | 0,889 [0,786-0,975] | 0,914 [0,841-0,973] | 0,995 [0,990-0,999] |
| LBBB | 30 | 1,000 [0,999-1,000] | 0,996 [0,982-1,000] | 0,967 [0,885-1,000] | 1,000 [1,000-1,000] | 0,983 [0,939-1,000] | 1,000 [1,000-1,000] |
| SB | 16 | 0,997 [0,994-1,000] | 0,864 [0,693-0,973] | 0,625 [0,375-0,867] | 0,769 [0,500-1,000] | 0,690 [0,444-0,857] | 0,996 [0,991-1,000] |
| ST | 37 | 0,999 [0,997-1,000] | 0,977 [0,939-0,999] | 0,838 [0,706-0,945] | 0,939 [0,840-1,000] | 0,886 [0,800-0,957] | 0,998 [0,994-1,000] |
| AF | 13 | 0,997 [0,993-1,000] | 0,789 [0,558-0,981] | 0,692 [0,429-0,923] | 0,818 [0,556-1,000] | 0,750 [0,522-0,919] | 0,998 [0,994-1,000] |
| Média macro |  | 0,998 [0,996-0,999] | 0,916 [0,870-0,960] | 0,760 [0,686-0,830] | 0,891 [0,821-0,954] | 0,812 [0,747-0,865] | 0,998 [0,996-0,999] |
| Validação A3, média macro (8.158 exames) | | 0,989 [0,987-0,991] | 0,691 [0,664-0,724] | 0,746 [0,718-0,773] | 0,681 [0,651-0,711] | 0,709 [0,685-0,733] | 0,993 [0,993-0,994] |

**Leitura:** contra o gabarito de especialistas, o A3 vai **melhor** que na validação em AUROC, AUPRC, precisão e F1 (macro F1 0,812 vs 0,709); a sensibilidade fica parecida (0,760 vs 0,746) e 1dAVb continua perdendo metade dos positivos (sensibilidade 0,50). Hipóteses (todas **não verificadas**): rótulos do gabarito mais limpos que os da CODE-15% (erros de rótulo no nosso teste interno "punem" o modelo por acertos), prevalência maior que na validação em 5 das 6 alterações (1dAVb, RBBB, LBBB e ST entre 3,4% e 4,5%; SB 1,9% vs 1,4%; AF é a exceção, 1,6% vs 2,1%), o que pesa em AUPRC, precisão e F1 mas não em AUROC, exames possivelmente escolhidos de forma diferente e possível sobreposição de pacientes com o treino.

## 4. Análise POSTERIOR ao teste (posterior, descritiva, não muda nada)

Feita depois de ver os resultados, só com as saídas brutas já salvas (`analise_testes.py posterior`; log em `logs/analise_testes_posterior_20261108.log`; CSVs `resultados/posterior_*.csv`). Nada aqui volta para o modelo, os limiares ou as exclusões.

### (a) Positivos previstos com o limiar congelado vs positivos reais

| Conjunto | Alteração | Avaliados | Positivos reais (prevalência) | Previstos | Previstos / reais |
|---|---|---|---|---|---|
| Teste interno | 1dAVb | 16109 | 281 (1,7%) | 256 | 0,91 |
| Teste interno | RBBB | 16109 | 442 (2,7%) | 549 | 1,24 |
| Teste interno | LBBB | 16109 | 290 (1,8%) | 335 | 1,16 |
| Teste interno | SB | 16109 | 308 (1,9%) | 320 | 1,04 |
| Teste interno | ST | 16109 | 353 (2,2%) | 452 | 1,28 |
| Teste interno | AF | 16109 | 310 (1,9%) | 288 | 0,93 |
| PTB-XL | 1dAVb | 21799 | 793 (3,6%) | 969 | 1,22 |
| PTB-XL | RBBB | 20681 | 540 (2,6%) | 581 | 1,08 |
| PTB-XL | LBBB | 21722 | 536 (2,5%) | 768 | 1,43 |
| PTB-XL | SB | 21799 | 637 (2,9%) | 426 | 0,67 |
| PTB-XL | ST | 21799 | 826 (3,8%) | 962 | 1,16 |
| PTB-XL | AF | 21726 | 1497 (6,9%) | 1445 | 0,97 |
| CODE-TEST | 1dAVb | 827 | 28 (3,4%) | 15 | 0,54 |
| CODE-TEST | RBBB | 827 | 34 (4,1%) | 36 | 1,06 |
| CODE-TEST | LBBB | 827 | 30 (3,6%) | 29 | 0,97 |
| CODE-TEST | SB | 827 | 16 (1,9%) | 13 | 0,81 |
| CODE-TEST | ST | 827 | 37 (4,5%) | 33 | 0,89 |
| CODE-TEST | AF | 827 | 13 (1,6%) | 11 | 0,85 |

No teste interno, o modelo prevê de 0,91 a 1,28 vez o número real (limiar de F1 máximo tende a aceitar alguns falsos positivos). No PTB-XL, **SB fica muito abaixo (0,67)** e **LBBB muito acima (1,43)**; no CODE-TEST, **1dAVb fica em 0,54** (o modelo quase só marca quando tem certeza: precisão 0,93, sensibilidade 0,50).

### (b) Teste interno com e sem os exames de sinal idêntico a outro

MD5 do sinal bruto de todos os exames das partes 0 a 3 (cache em `code15/md5_p0-3.csv`, fora do git). Dos 16.109 exames avaliados, **139 têm sinal idêntico a outro exame** das partes 0 a 3; **28** são idênticos a algum exame de treino/validação e **28** têm o mesmo sinal em outro paciente, e são **os mesmos 28 exames** (conferido): todo sinal do teste repetido em treino/validação pertence a outro paciente; os outros 111 repetem um exame do próprio paciente dentro do teste. Só **5** dos 139 têm alguma das 6 alterações.

| Alteração | AUROC com / sem | AUPRC com / sem | F1 com / sem |
|---|---|---|---|
| 1dAVb | 0,978 / 0,978 | 0,477 / 0,479 | 0,525 / 0,526 |
| RBBB | 0,992 / 0,992 | 0,826 / 0,826 | 0,819 / 0,823 |
| LBBB | 0,993 / 0,993 | 0,784 / 0,782 | 0,781 / 0,779 |
| SB | 0,992 / 0,992 | 0,693 / 0,695 | 0,678 / 0,679 |
| ST | 0,993 / 0,993 | 0,717 / 0,718 | 0,740 / 0,741 |
| AF | 0,984 / 0,985 | 0,761 / 0,764 | 0,686 / 0,689 |
| Média macro | 0,989 / 0,989 | 0,710 / 0,711 | 0,705 / 0,706 |

Maior diferença em qualquer métrica e alteração: **0,005** (tirando só os 28 idênticos a treino/validação: 0,002). Os sinais idênticos **não explicam** o resultado do teste interno. (Métricas pontuais, sem IC.)

### (c) Onde o PTB-XL e o CODE-TEST caem em relação à validação

AUROC (diferença em relação à validação entre parênteses; AUROC não depende da prevalência):

| Alteração | Validação A3 | Teste interno | PTB-XL | CODE-TEST |
|---|---|---|---|---|
| 1dAVb | 0,972 | 0,978 (+0,006) | 0,924 (-0,048) | 0,997 (+0,025) |
| RBBB | 0,995 | 0,992 (-0,003) | 0,997 (+0,001) | 0,995 (+0,000) |
| LBBB | 0,994 | 0,993 (-0,001) | 0,998 (+0,003) | 1,000 (+0,006) |
| SB | 0,993 | 0,992 (-0,001) | 0,920 (-0,072) | 0,997 (+0,004) |
| ST | 0,995 | 0,993 (-0,002) | 0,993 (-0,002) | 0,999 (+0,004) |
| AF | 0,986 | 0,984 (-0,002) | 0,990 (+0,003) | 0,997 (+0,011) |
| Média macro | 0,989 | 0,989 (-0,000) | 0,970 (-0,019) | 0,998 (+0,008) |

F1 com o limiar congelado:

| Alteração | Validação A3 | Teste interno | PTB-XL | CODE-TEST |
|---|---|---|---|---|
| 1dAVb | 0,485 | 0,525 (+0,040) | 0,453 (-0,032) | 0,651 (+0,166) |
| RBBB | 0,826 | 0,819 (-0,007) | 0,924 (+0,098) | 0,914 (+0,088) |
| LBBB | 0,798 | 0,781 (-0,018) | 0,793 (-0,006) | 0,983 (+0,185) |
| SB | 0,658 | 0,678 (+0,020) | 0,668 (+0,010) | 0,690 (+0,031) |
| ST | 0,783 | 0,740 (-0,043) | 0,814 (+0,031) | 0,886 (+0,102) |
| AF | 0,701 | 0,686 (-0,015) | 0,824 (+0,123) | 0,750 (+0,049) |
| Média macro | 0,709 | 0,705 (-0,004) | 0,746 (+0,037) | 0,812 (+0,104) |

(AUPRC por alteração em `figuras/j_testes_auprc.png` e na saída de `analise_testes.py comparacao`.)

**PTB-XL, rótulo a rótulo:**
- **SB (AUROC 0,993 → 0,920):** a maior queda. O modelo marca só 426 dos 637 casos esperados (sensibilidade 0,56) mas com precisão alta (0,83). **23,5% dos 637 SB do PTB-XL recebem nota < 0,01** (28,1% < 0,05; 44,3% abaixo do limiar congelado de 0,37): para um quarto deles o modelo "tem certeza" de que não é SB (ver o patamar na curva ROC de `figuras/k_roc_pr_ptbxl.png`). Hipóteses, **todas não verificadas:** (1) **definição do rótulo**: o corte de frequência cardíaca para "bradicardia sinusal" pode ser diferente entre o laudo da Telessaúde-MG e o código `SBRAD` do PTB-XL (o PTB-XL não tem coluna de frequência cardíaca para conferir); (2) **rótulos de ritmo do PTB-XL** vêm em parte de laudo automático e quase sempre com `likelihood` 0 (seção 4 do CLAUDE.md); (3) **população** (mais velha, mais homens; ex.: bradicardia com outros achados).
- **1dAVb (AUROC 0,972 → 0,924):** sensibilidade igual à da validação (~0,50), mas a especificidade cai de 0,992 para 0,973 e o modelo marca 1,22 vez o número real. Hipóteses, **não verificadas:** **definição** (corte do intervalo PR, PR limítrofe), **população** (PTB-XL mais velho, PR mais longo em média) e **escala/forma do sinal** (a normalização por exame anula fatores de escala, medido na validação; diferenças de filtro ou da reamostragem 500 → 400 Hz no intervalo PR não foram testadas).
- **LBBB:** AUROC alto (0,998), mas 1,43 vez mais positivos previstos que reais e especificidade 0,988 (validação 0,996). Hipótese, **não verificada:** os bloqueios incompletos (ILBBB) já saem da avaliação; outros achados de QRS largo do PTB-XL (ex.: distúrbio de condução intraventricular) podem ser marcados como LBBB.
- **RBBB, ST, AF:** iguais ou melhores que na validação. Para AF, AUPRC e F1 sobem com a prevalência (6,9% vs 2,1%).
- **Escala:** as amplitudes do PTB-XL são ~0,4-0,5× as da CODE-15% (seção 4 do CLAUDE.md), mas a normalização por exame anula fator de escala (teste de estresse na validação: notas mudam no máximo 3,3e-5). Escala de amplitude como causa é **improvável, mas não verificada no PTB-XL**.

**CODE-TEST:** nenhuma alteração cai em AUROC; F1 sobe em todas. A única fraqueza clara é a de sempre, 1dAVb (sensibilidade 0,50, previstos/reais 0,54).

### (d) CODE-TEST: limitações e comparação com os anotadores do próprio zip

- **AF tem só 13 positivos** (SB 16): ICs de sensibilidade de 0,43 a 0,92 (AF) e 0,38 a 0,87 (SB). Uma diferença de 1 ou 2 exames muda muito o número.
- **Sobreposição de pacientes com a CODE-15%: não verificada** (sem identificador). Se houver, o resultado é otimista.
- F1 contra o `gold_standard` (sem IC; contexto, não teste estatístico). As anotações `dnn` (rede de Ribeiro et al. 2020), cardiologistas, residentes e estudantes vêm no zip do CODE-TEST:

| Quem | 1dAVb | RBBB | LBBB | SB | ST | AF | Média macro |
|---|---|---|---|---|---|---|---|
| A3 (este projeto) | 0,651 | 0,914 | 0,983 | 0,690 | 0,886 | 0,750 | 0,812 |
| dnn | 0,897 | 0,944 | 1,000 | 0,882 | 0,960 | 0,870 | 0,925 |
| cardiologist1 | 0,828 | 0,957 | 0,966 | 0,897 | 0,896 | 0,870 | 0,902 |
| cardiologist2 | 0,926 | 0,971 | 1,000 | 0,897 | 0,930 | 0,889 | 0,935 |
| cardiology_residents | 0,776 | 0,917 | 0,947 | 0,882 | 0,882 | 0,769 | 0,862 |
| emergency_residents | 0,719 | 0,852 | 0,912 | 0,848 | 0,946 | 0,696 | 0,829 |
| medical_students | 0,732 | 0,928 | 0,915 | 0,750 | 0,873 | 0,706 | 0,817 |

**Correção (seção 7, D):** `cardiologist1` (F1 0,902) e `cardiologist2` (0,935) são os dois cardiologistas cuja concordância **é** o gold standard (nas divergências, um terceiro especialista que via as duas anotações decidiu; README do zip). Comparar os dois com o gold standard é **circular** e não deve ser usado. A comparação justa é com residentes e estudantes (seção 7, D).

O A3 (CNN pequena, ~406 mil parâmetros, 55 mil exames, CPU) fica abaixo da rede de Ribeiro et al. e perto dos estudantes de medicina e dos residentes de emergência em F1 macro. A maior distância está em 1dAVb e SB. Quantos exames a rede deles usou no treino e se ela viu pacientes do CODE-TEST: ver o artigo; **não verificado aqui**.

## 5. Figuras (geradas depois de salvar as saídas brutas)

- `figuras/i_testes_auroc.png`: AUROC por alteração, validação × teste interno × PTB-XL × CODE-TEST, com IC 95% (eixo começa em 0,5).
- `figuras/j_testes_auprc.png`: o mesmo para AUPRC.
- `figuras/k_roc_pr_ptbxl.png`: curvas ROC e Precisão-Recall do PTB-XL por alteração, com o ponto do limiar congelado.

## 6. O que não foi verificado

- Causa das quedas de SB e 1dAVb no PTB-XL (definição do rótulo, população, forma do sinal): **não verificado**.
- Sobreposição de pacientes entre CODE-TEST e CODE-15%: **não verificado**.
- Tudo foi rodado com uma seed (42); a variação entre seeds no teste não foi medida (por decisão, só o A3 seed 42 foi testado).

## 7. Análise exploratória posterior ao teste (madrugada de 08/11; posterior, descritiva, exploratória, não muda nada)

Feita com `analise_posterior_testes.py` (log em `logs/analise_posterior_testes_20261108.log`; CSVs em `resultados/posterior_testes/`). **Nenhum predict:** as notas vêm das saídas brutas já salvas (`resultados/bruto_*.npz`) e, para a validação, de `resultados/erros_A3/previsoes_e_rotulos.npy`. Sinal do PTB-XL e da validação só foi lido para estimar a frequência cardíaca. Modelo, limiares, normalização, exclusões, limite de `likelihood` e JSON não mudaram. **Nenhum número desta seção é resultado principal.**

### A) SB no PTB-XL: o rótulo quer dizer a mesma coisa nas duas bases? (hipótese)

**(i) Notas do modelo (limiar congelado 0,370).** PTB-XL, 637 positivos SB: mediana 0,456; **179 (28,1%) com nota < 0,05**; 355 (55,7%) acima do limiar. Negativos (21.162): mediana 0,000; 71 (0,3%) acima do limiar. Validação da CODE-15%, 118 positivos: mediana 0,526; **5 (4,2%) com nota < 0,05**; 80 (67,8%) acima do limiar. Figura `l_sb_notas_ptbxl_validacao.png`.

**(ii)/(iii) Frequência cardíaca (FC) estimada pelo sinal.**
- **Método:** derivação II do sinal bruto (PTB-XL a 500 Hz; CODE-15% a 400 Hz, só a parte com sinal, sem os zeros das pontas). Filtro passa-banda de 5 a 15 Hz (realça o QRS e atenua onda T e linha de base), derivada ao quadrado, média móvel de 150 ms, picos com no mínimo 0,25 s entre si e altura de pelo menos 30% do percentil 99. FC = 60 / mediana dos intervalos entre picos. Com menos de 3 picos, sem estimativa.
- **Amostras (seed 42):** PTB-XL: **todos os 637 SB positivos** (não há 1.000), 1.000 negativos sorteados e 200 positivos de taquicardia (ST) como controle. Validação: todos os 118 SB positivos, 1.000 negativos sorteados e os 171 ST positivos como controle.
- **Incerteza:** (1) em sinais sintéticos de FC conhecida (40 a 150 bpm, 400 e 500 Hz), erro < 1,5 bpm (`python analise_posterior_testes.py checar`); (2) a FC da derivação II e a da V5 (calculadas separadamente) ficam a até 5 bpm uma da outra em 96,9% a 98,9% dos exames, conforme o grupo; (3) controle de taquicardia: FC > 100 em 187 de 195 estimados no PTB-XL (95,9%) e 161 de 165 na validação (97,6%); (4) sem estimativa em 2 de 637 SB positivos do PTB-XL e 2 de 118 na validação. **É uma estimativa, não a FC do laudo**; perto do corte de 50 bpm, um erro de 1 a 2 bpm muda a faixa de um exame. Exames da CODE-15% de 7,3 s têm poucos batimentos (cerca de 6 a 50 bpm).
- **Resultados (derivação II):**
  - **PTB-XL, SB positivos (635 com estimativa):** mediana 48,6 bpm (p10 43,2; p90 57,3). < 50 bpm: 429 (67,6%); **50 a 59,9: 190 (29,9%)**; < 60: 619 (97,5%).
  - **Validação CODE-15%, SB positivos (116):** mediana 47,3 bpm (p10 42,1; **p90 50,1**). < 50 bpm: 103 (88,8%); 50 a 59,9: 13 (11,2%); < 60: 116 (100%).
  - **Validação CODE-15%, SB negativos (976 com estimativa):** **159 (16,3%) com FC estimada entre 50 e 59,9** e mesmo assim sem o rótulo SB; < 50: 10 (1,0%). PTB-XL, negativos: 50 a 59,9 em 143 de 987 (14,5%); < 50: 11 (1,1%).
  - **Nota do modelo por faixa de FC, SB positivos do PTB-XL:** FC < 50: 429, nota < 0,05 em 11 (2,6%), acima do limiar 347 (80,9%). FC 50 a 54,9: 90, nota < 0,05 em 54 (60,0%), acima do limiar 5 (5,6%). **FC 55 a 59,9: 100, nota < 0,05 em 100 (100%)**, acima do limiar 0. FC ≥ 60: 16, nota < 0,05 em 14. **Dos 179 SB do PTB-XL com nota < 0,05, 168 têm FC estimada ≥ 50.**
  - **Exploratório:** AUROC de SB no PTB-XL usando só os positivos com FC estimada < 50 (429) contra todos os negativos: **0,997**; só os positivos com FC ≥ 50 (206): 0,761. (Para comparar: 0,920 com todos, 0,993 na validação.) Figura `m_fc_sb_ptbxl_validacao.png`.
- **Leitura (hipótese, agora com evidência descritiva):** na CODE-15%, o rótulo SB se comporta como **"FC abaixo de ~50 bpm"** (90% dos positivos abaixo de 50,1 bpm, e 16% dos negativos entre 50 e 60 sem o rótulo). No PTB-XL, `SBRAD` inclui 50 a 60 bpm (30% dos positivos). O modelo aprendeu o corte da CODE-15% e dá nota quase zero aos SB do PTB-XL entre 55 e 60 bpm. **A queda de SB no PTB-XL parece ser uma diferença de definição do rótulo, não uma falha do modelo em ler o sinal de outra base.** O critério oficial de SB da CODE-15% (laudo da Telessaúde-MG) é **não verificado** (a documentação não diz); a conclusão depende de uma FC estimada.

### B) 1dAVb: notas e erros por idade e sexo

Notas nos positivos (mediana; quantis 5% e 95%): validação 0,431 (0,028 a 0,899); interno 0,489 (0,032 a 0,880); PTB-XL 0,498 (0,001 a 0,958); CODE-TEST 0,532 (0,175 a 0,852). Nos negativos, a mediana é 0,000 nos quatro, mas o quantil 95% vai a 0,243 no PTB-XL (0,076 a 0,082 na CODE-15%, 0,043 no CODE-TEST). Em todos os conjuntos, **a mediana dos positivos fica perto do limiar (0,489)**, então cerca de metade dos 1dAVb fica abaixo dele: o problema é o mesmo nas quatro bases. No PTB-XL há também mais negativos com nota intermediária (mais falsos positivos).

Erros por subgrupo (sensibilidade, com o número de positivos; taxa de falso positivo):
- **Validação:** < 40 anos 0,31 (13 positivos); 40-59 0,59 (17); 60-69 0,48 (29); 70+ 0,45 (82). Homens 0,38 (69), mulheres 0,54 (72). Falso positivo: de 0,26% (< 40) a 1,54% (70+).
- **Teste interno:** < 40 0,45 (29); 40-59 0,49 (39); 60-69 0,56 (32); 70+ 0,50 (181). Homens 0,53 (163), mulheres 0,47 (118). Falso positivo: de 0,21% (< 40) a 1,54% (70+).
- **PTB-XL:** < 40 0,52 (29); 40-59 0,60 (100); 60-69 0,51 (195); 70+ 0,48 (469). Homens 0,53 (481), mulheres 0,46 (312). **Falso positivo de 0,92% (< 40) a 4,06% (70+)**; homens 3,57%, mulheres 1,80%.
- **CODE-TEST:** 2 a 17 positivos por subgrupo (instável): homens 0,41 (17), mulheres 0,64 (11); 1 falso positivo em 799 negativos.
- **Leitura:** a sensibilidade fica perto de 0,5 em quase todas as faixas e nos dois sexos. A diferença entre homens e mulheres **muda de lado** entre a validação (mulheres melhor) e o teste interno/PTB-XL (homens melhor), então não há padrão estável. O que aumenta com a idade, de forma consistente, é a **taxa de falso positivo**; no PTB-XL ela é 2,6 a 4,4 vezes a do teste interno na mesma faixa de idade. Causa: **não verificada** (sem medida do intervalo PR, as hipóteses de definição, população e forma do sinal não foram testadas).

### C) LBBB no PTB-XL: quem são os falsos positivos

- **251 falsos positivos** em 21.186 negativos avaliados (os 77 registros com ILBBB já estavam fora da avaliação de LBBB; o modelo marcou 33 deles como LBBB).
- **203 dos 251 (80,9%) têm algum código de distúrbio de condução** (classe "CD" do `scp_statements.csv`), contra **19,5%** entre os verdadeiros negativos. Por código (um registro pode ter vários): **LAFB 141**, **IVCD 71** (distúrbio de condução intraventricular inespecífico), 1AVB 28, WPW 5, IRBBB 1, ILBBB 0 (excluídos), CRBBB 0, LPFB 0.
- Os 48 sem código CD: **marca-passo (PACE) 28**, hipertrofia ventricular esquerda (LVH) 12, isquemia (ISC_) 11, entre outros.
- **Leitura:** os falsos positivos de LBBB são, na maioria, **outros distúrbios de condução** (LAFB, IVCD) e **ritmos de marca-passo**, que no ECG podem se parecer com LBBB. Se esses achados existem na CODE-15% e como foram rotulados lá: **não verificado**.

### D) CODE-TEST: comparação só com os grupos justos

- **Quem é quem (README do zip):** `cardiologist1` e `cardiologist2` são dois cardiologistas; o **gold standard é a concordância deles**, e nas divergências um terceiro especialista, que via as duas anotações, decidiu. `cardiology_residents` = dois residentes de cardiologia do 4º ano (cada um anotou metade); `emergency_residents` = dois residentes de emergência do 3º ano (metade cada); `medical_students` = dois estudantes do 5º ano de medicina (metade cada); `dnn` = rede de Ribeiro et al., com limiar escolhido para maximizar o F1 (em que conjunto: **não verificado** pelo README).
- **Os F1 reportados antes: 0,902 = cardiologist1; 0,935 = cardiologist2. Os dois são circulares** (são a origem do gold standard) e saem da comparação.
- **F1 com IC 95% (bootstrap por exame, 1.000 reamostras, seed 42, as mesmas para todos) e diferença A3 − grupo:**
  - **1dAVb (n=28; A3 AUROC 0,997 [0,993-0,999]):** A3 0,651 [0,465-0,818]; res. cardiologia 0,776 [0,622-0,893] (dif. −0,124 [−0,302 a 0,055]); res. emergência 0,719 [0,580-0,833] (−0,068 [−0,274 a 0,140]); estudantes 0,732 [0,600-0,839] (−0,081 [−0,284 a 0,133]).
  - **RBBB (n=34; AUROC 0,995 [0,987-1,000]):** A3 0,914 [0,841-0,973]; res. cardiologia 0,917 [0,846-0,973] (−0,002 [−0,072 a 0,073]); res. emergência 0,852 [0,750-0,938] (+0,062 [−0,051 a 0,176]); estudantes 0,928 [0,857-0,986] (−0,013 [−0,077 a 0,056]).
  - **LBBB (n=30; AUROC 1,000 [0,999-1,000]):** A3 0,983 [0,939-1,000]; res. cardiologia 0,947 [0,868-1,000] (+0,036 [0,000 a 0,101]); res. emergência 0,912 [0,824-0,982] (+0,071 [−0,003 a 0,166]); estudantes 0,915 [0,828-0,980] (+0,068 [−0,001 a 0,162]).
  - **SB (n=16; AUROC 0,997 [0,994-1,000]):** A3 0,690 [0,444-0,857]; **res. cardiologia 0,882 [0,727-0,977] (−0,193 [−0,412 a −0,019])**; res. emergência 0,848 [0,692-0,968] (−0,159 [−0,380 a 0,022]); estudantes 0,750 [0,533-0,900] (−0,060 [−0,225 a 0,078]).
  - **ST (n=37; AUROC 0,999 [0,997-1,000]):** A3 0,886 [0,800-0,957]; res. cardiologia 0,882 [0,788-0,952] (+0,003 [−0,110 a 0,115]); res. emergência 0,946 [0,882-0,988] (−0,060 [−0,158 a 0,026]); estudantes 0,873 [0,775-0,947] (+0,012 [−0,104 a 0,136]).
  - **AF (n=13; AUROC 0,997 [0,993-1,000]):** A3 0,750 [0,522-0,919]; res. cardiologia 0,769 [0,545-0,929] (−0,019 [−0,274 a 0,233]); res. emergência 0,696 [0,429-0,875] (+0,054 [−0,292 a 0,373]); estudantes 0,706 [0,476-0,851] (+0,044 [−0,215 a 0,307]).
  - **Média macro:** A3 0,812 [0,747-0,865]; res. cardiologia 0,862 [0,805-0,906] (−0,050 [−0,122 a 0,012]); res. emergência 0,829 [0,766-0,882] (−0,017 [−0,099 a 0,058]); estudantes 0,817 [0,754-0,864] (−0,005 [−0,072 a 0,059]).
- **Leitura:** com 827 exames e 13 a 37 positivos por alteração, **não dá para separar o A3 dos residentes e dos estudantes** em F1 macro (todas as diferenças têm IC que inclui zero). A única diferença com IC fora de zero é **SB contra os residentes de cardiologia (A3 pior)**; em LBBB o A3 fica no limite de ser melhor que os três grupos (limite inferior do IC entre −0,003 e 0,000). Sobreposição de pacientes com a CODE-15%: continua **não verificada**.

### E) Figuras (todas marcadas "posterior ao teste")

- `figuras/i_testes_auroc.png` e `figuras/j_testes_auprc.png` (validação × interno × PTB-XL × CODE-TEST, com IC) e `figuras/k_roc_pr_ptbxl.png`: regeneradas com a marca "posterior ao teste: modelo congelado, sem ajuste depois do teste" (os números não mudaram).
- `figuras/l_sb_notas_ptbxl_validacao.png`: distribuição acumulada das notas de SB, positivos e negativos, PTB-XL × validação.
- `figuras/m_fc_sb_ptbxl_validacao.png`: FC estimada nos SB positivos das duas bases e, no PTB-XL, nota do modelo × FC.

### O que se confirmou, o que não e o que continua não verificado

- **Confirmado (descritivo):** a queda de SB no PTB-XL se concentra nos positivos com FC estimada entre 50 e 60 bpm, faixa que a CODE-15% praticamente não rotula como SB. Os falsos positivos de LBBB no PTB-XL são, na maioria, outros distúrbios de condução e marca-passo. 1dAVb perde cerca de metade dos positivos em todas as bases, e o PTB-XL tem mais falsos positivos de 1dAVb, crescendo com a idade. No CODE-TEST, o A3 fica no nível dos residentes e estudantes em F1 macro.
- **Não confirmado:** padrão estável de erro de 1dAVb por sexo; A3 "melhor" que residentes ou estudantes.
- **Continua não verificado:** critério oficial de SB e de 1dAVb da CODE-15%; FC e intervalo PR "verdadeiros" (só estimativa de FC, nenhuma medida de PR); sobreposição de pacientes CODE-TEST × CODE-15%; em que conjunto foi escolhido o limiar da `dnn`.

