#!/bin/bash
# Parte B com as 18 partes no Colab (GPU). SÓ TREINO -> VALIDAÇÃO; o teste da Parte B nunca é lido
# (parte_b_cnn.carregar corta o grupo teste na carga).
# Pré-requisito: code15/ com exams.csv e exams_part0..17.hdf5 (python colab_baixar.py --partes 0-17).
# Uso (raiz do repositório): bash colab_parte_b.sh > /content/parte_b.log 2>&1
# Ao fim, empacota resultados + modelo B3z em /content/parte_b_p0-17.zip.
set -euo pipefail
P=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17
T=p0-17
cd "$(dirname "$0")"
pip install -q wfdb==4.3.0                                           # ecg_dados.py importa wfdb (não vem no Colab)
date; nvidia-smi --query-gpu=name --format=csv,noheader; git log -1 --oneline
python -u parte_b_cnn.py preparar --partes $P                        # elegíveis, todo-zero pelo sinal
python -u parte_b_cnn.py treinar_zero_idade --partes $P --max-epocas 15 --max-min 300   # B3z (prioridade)
python -u parte_b_cnn.py caracteristicas --partes $P                 # 128 características do A3 congelado (B2t, B3, B4p)
python -u parte_b_cnn.py avaliar --partes $P                         # B0, B1, B2t, B3, B3z + IC
python -u parte_b_alteracoes.py --partes $P                          # B4, B4n, B4p
python -u parte_b_analises.py --partes $P                            # risco escondido e triagem
python -u parte_b_km.py --partes $P                                  # Kaplan-Meier por quartil (inclui censurados)
date
cd /content && zip -j parte_b_$T.zip repo/resultados/parte_b_*${T}* repo/modelos/B3z_${T}_melhor.keras repo/figuras/parte_b_km_${T}.png parte_b.log
echo FIM
