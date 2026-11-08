#!/bin/bash
# Fila do Bloco 3/4: um treino por vez; não começa treino novo depois de 14:47 (4h30 desde 10:17).
cd /c/ecg
PY=.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8
pode() { [ "$(date +%H%M)" -lt 1447 ]; }
until grep -q "seed 43" resultados/experimentos.csv || grep -q Traceback logs/A3_seed43.log; do sleep 20; done; sleep 15
echo "$(date) seed43 terminou" >> logs/fila.log
for s in 44 45; do
  pode || { echo "$(date) corte de horario: seed $s nao iniciada" >> logs/fila.log; exit 0; }
  echo "$(date) inicia seed $s" >> logs/fila.log
  $PY a1_cnn.py treinar A3_seed$s --normalizar --partes 0,1,2,3 --max-min 90 --max-epocas 12 --seed $s > logs/A3_seed$s.log 2>&1
  echo "$(date) fim seed $s (codigo $?)" >> logs/fila.log
done
pode || { echo "$(date) corte de horario: A4 nao iniciado" >> logs/fila.log; exit 0; }
echo "$(date) inicia A4" >> logs/fila.log
$PY a1_cnn.py treinar A4 --normalizar --partes 0,1,2,3 --max-min 90 --max-epocas 12 --peso-por-classe > logs/A4.log 2>&1
echo "$(date) fim A4 (codigo $?)" >> logs/fila.log
