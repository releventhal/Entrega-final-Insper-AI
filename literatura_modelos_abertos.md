# Modelos abertos de ECG (levantamento de 14/11, sessão sem GPU do Claude do Tom)

> **Como foi feito:** só buscas na web (resumos de model cards, papers e benchmarks). **Nenhum peso foi baixado** e nenhuma página foi lida inteira nesta sessão. Tudo marcado "não verificado" precisa ser conferido na fonte original antes de ir para slides ou paper. Quando duas fontes discordam, as duas aparecem.

## Por que isto importa para o projeto
- **Risco de vazamento:** se um modelo pré-treinado viu a **CODE-15%** no pré-treino, usá-lo (fine-tuning ou extrator) e avaliar na nossa validação/teste interno da CODE-15% **não é avaliação em pacientes nunca vistos** pelo pré-treino. Ele só seria comparável de forma limpa no PTB-XL e no CODE-TEST (e, no PTB-XL, só se o PTB-XL também não estiver no pré-treino).
- **Escala:** os modelos abaixo têm de ~34 a ~312 milhões de parâmetros; o A3 tem ~406 mil. Nenhum foi testado aqui (sem GPU, regra da sessão).

## Tabela

| Modelo | Parâmetros | Arquitetura / tarefa de pré-treino | Dados de pré-treino | CODE-15% no pré-treino? | Licença dos pesos | Métricas reportadas | Link |
|---|---|---|---|---|---|---|---|
| **HuBERT-ECG** (Coppola et al., medRxiv 2024) | base: 93,1 M (página do HF); 97,2 M segundo o benchmark de Oldenburg; 3 tamanhos (small/base/large) | Transformer, autossupervisionado estilo HuBERT (predição mascarada) | 9,1 milhões de ECGs de 12 derivações de 4 países (lista por base **não verificada**) | **Não verificado** no pré-treino. O conjunto de **fine-tuning** "Cardio-Learning" inclui "CODE" (e PTB-XL, SaMi-Trop, Chapman, Ningbo etc.), segundo o model card | CC BY-NC 4.0 (model card e GitHub; uso não comercial) | Avaliado em fine-tuning em 21 bases (valores **não verificados** aqui) | [HF base](https://huggingface.co/Edoardo-BS/hubert-ecg-base) · [preprint](https://www.medrxiv.org/content/10.1101/2024.11.14.24317328.full.pdf) |
| **ECG-FM** (McKeen et al., JAMIA Open 2025; Wang Lab, Toronto) | 311.940.352 (README) vs 90.883.072 no checkpoint "base" público; 93,9 M segundo o benchmark de Oldenburg (fontes conflitam) | Transformer (wav2vec 2.0/BERT), máscara + contrastivo | Paper: UHN-ECG + PhysioNet 2021 + MIMIC-IV-ECG (~1,5 M ECGs). Checkpoint público: só MIMIC-IV-ECG v1.0 + PhysioNet 2021 v1.0.3 | **Não listado** nas fontes lidas (não verificado na lista completa da PhysioNet 2021; ela inclui PTB-XL, o que **contamina o PTB-XL** como teste) | Repositório MIT; licença específica dos pesos no HF **não verificada** | **Não verificado** aqui | [GitHub](https://github.com/bowang-lab/ECG-FM) · [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12530324/) |
| **ECG-JEPA** (Kim, 2024) | 87,2 M (benchmark de Oldenburg) | Transformer, JEPA (predição no espaço latente) | ~174-180 mil ECGs de bases abertas; benchmark lista **Chapman, Ningbo e CODE-15%** (outro benchmark, FOUND-AF, descreve diferente) | **Sim, segundo o benchmark de Oldenburg** (confirmar no paper original) | **Não verificado** (código aberto no GitHub segundo o paper) | **Não verificado** aqui | [arXiv 2410.08559](https://arxiv.org/html/2410.08559v4) |
| **ST-MEM** | 90,3 M (Oldenburg) ou 85,2 M (FOUND-AF) | Transformer, autoencoder mascarado espaço-temporal | Chapman, Ningbo e **CODE-15%** (~174 mil; Oldenburg) | **Sim, segundo o benchmark de Oldenburg** (confirmar no paper original) | **Não verificado** | **Não verificado** | [benchmark](https://arxiv.org/html/2509.25095v1) |
| **ECGFounder** (Li et al., 2024-2025) | 33,8 M (Oldenburg) | CNN RegNet, **supervisionado** com 150 rótulos | Harvard-Emory ECG Database (10.771.552 ECGs) | Não listado nas fontes lidas | **Não verificado** (o paper cita CC BY-NC-SA 4.0, que é a licença do texto; pesos: não verificado) | AUROC > 0,95 em 80 diagnósticos na validação interna (dos autores) | [GitHub](https://github.com/PKUDigitalHealth/ECGFounder) · [arXiv 2410.04133](https://arxiv.org/pdf/2410.04133) |
| **ECG-CPC**, **MERL**, **ECGFM-KED** | ECG-CPC 3,8 M; os outros **não verificados** | ECG-CPC: contrastivo preditivo; MERL/KED: ECG + texto | ECG-CPC ~10,7 M amostras; MERL/KED ~800 mil (Oldenburg) | **Não verificado** | **Não verificado** | **Não verificado** | [benchmark](https://arxiv.org/html/2509.25095v1) |
| **Ribeiro et al. 2020** (referência do projeto) | **não verificado** | ResNet 1D supervisionada, 6 alterações | CODE (tamanho **não verificado nesta sessão**; a CODE-15% é uma amostra de 15% da CODE) | Sim (mesma coorte, seção 2 do CLAUDE.md) | código no GitHub; licença dos pesos **não verificada** | F1 no CODE-TEST: rede 0,925 macro (anotação `dnn` do zip, já medido em 08/11) | [GitHub](https://github.com/antonior92/automatic-ecg-diagnosis) |

## Leitura (o que dá para dizer sem rodar nada)
1. **ECG-JEPA e ST-MEM** (segundo o benchmark de Oldenburg) têm a **CODE-15%** no pré-treino: usar um deles e avaliar na nossa validação ou no teste interno seria vazamento de pré-treino. Hipótese, não testada: no PTB-XL/CODE-TEST a comparação seria mais limpa.
2. **ECG-FM** usa a PhysioNet 2021, que (pela composição conhecida do desafio; **não verificado nesta sessão**) inclui o PTB-XL: nesse caso o PTB-XL deixa de ser externo para ele.
3. **HuBERT-ECG** tem pesos CC BY-NC 4.0 (uso acadêmico ok; comercial não). A lista exata do pré-treino precisa do paper completo.
4. Benchmarks independentes (Oldenburg, arXiv 2509.25095; FOUND-AF, arXiv 2608.03597; OpenECG, arXiv 2503.00711) existem e são a fonte certa para comparar números; eles próprios discordam em parâmetros e dados de pré-treino.

## Fontes
- https://huggingface.co/Edoardo-BS/hubert-ecg-base
- https://www.medrxiv.org/content/10.1101/2024.11.14.24317328.full.pdf
- https://github.com/bowang-lab/ECG-FM · https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12530324/ · https://arxiv.org/abs/2408.05178v2
- https://arxiv.org/html/2410.08559v4 (ECG-JEPA)
- https://arxiv.org/html/2509.25095v1 e https://arxiv.org/pdf/2509.25095 (Benchmarking ECG FMs: A Reality Check Across Clinical Tasks)
- https://arxiv.org/pdf/2608.03597 (FOUND-AF)
- https://arxiv.org/pdf/2503.00711 (OpenECG)
- https://arxiv.org/pdf/2410.04133 · https://github.com/PKUDigitalHealth/ECGFounder (ECGFounder)
