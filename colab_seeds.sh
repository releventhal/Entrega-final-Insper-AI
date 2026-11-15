#!/bin/bash
# Parte B: seeds 43 e 44 do B3z com as 18 partes no Colab (GPU), SÓ TREINO -> VALIDAÇÃO, só para relatar estabilidade
# (não escolhem nem trocam o modelo final, que é o B3z p0-17 seed 42). O teste da Parte B nunca é lido
# (parte_b_cnn.carregar corta o grupo teste na carga).
# Pré-requisito: code15/ com exams.csv e exams_part0..17.hdf5 (python colab_baixar.py --partes 0-17).
# As duas seeds rodam AO MESMO TEMPO na mesma GPU (TF_FORCE_GPU_ALLOW_GROWTH, para uma não reservar a memória toda).
# Uso (raiz do repositório): bash colab_seeds.sh > /content/seeds.log 2>&1
set -uo pipefail
P=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17
T=p0-17
cd "$(dirname "$0")"
pip install -q wfdb==4.3.0 keras==3.15.1
export TF_FORCE_GPU_ALLOW_GROWTH=true
date; nvidia-smi --query-gpu=name --format=csv,noheader; git log -1 --oneline
for S in 43 44; do
  python -u parte_b_cnn.py treinar_zero_idade --partes $P --seed $S --max-epocas 15 --max-min 300 \
    > /content/seed_$S.log 2>&1 &
done
wait
date
tail -3 /content/seed_43.log /content/seed_44.log
python -u parte_b_seeds.py --partes $P --seeds 43,44
cd /content && zip -j parte_b_seeds_$T.zip repo/resultados/parte_b_B3z_${T}_s4*  repo/resultados/parte_b_seeds_${T}.csv \
  seed_43.log seed_44.log seeds.log
echo FIM
