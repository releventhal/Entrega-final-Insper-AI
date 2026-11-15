#!/bin/bash
# Parte B: TESTE do B3z p0-17 congelado (congelado_candidato_parte_b.json, status "congelado" em 15/11), UMA vez, no Colab.
# Ordem combinada com o Tom: as seeds 43/44 (colab_seeds.sh) terminam ANTES; este script espera o "FIM" do seeds.log.
# Passos: (1) características do A3 para treino e validação (o .npz de 77 MB não vai pelo git; necessárias para ajustar
# o B4p no treino); (2) conferência no conjunto livre "validacao" (tem de reproduzir as notas salvas; se não reproduzir,
# o script PARA antes do teste); (3) o teste, uma vez; (4) empacota tudo em /content/parte_b_teste_p0-17.zip.
# Uso (raiz do repositório): nohup bash colab_teste_b.sh > /content/teste_b.log 2>&1 &
set -euo pipefail
P=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17
T=p0-17
cd "$(dirname "$0")"
until grep -q "^FIM" /content/seeds.log 2>/dev/null; do sleep 60; done
echo "seeds terminaram: $(date)"
git pull -q
git log -1 --oneline
python -c "import json; c=json.load(open('congelado_candidato_parte_b.json')); print('status:', c['status']); assert c['status']=='congelado'"
python -u parte_b_cnn.py caracteristicas --partes $P
python -u parte_b_teste.py --conjunto validacao --partes $P
echo "=== TESTE (uma vez) $(date) ==="
python -u parte_b_teste.py --conjunto teste --partes $P --confirmo-congelado
date
cd /content && zip -j parte_b_teste_$T.zip repo/resultados/parte_b_teste_* repo/resultados/parte_b_validacao_*${T}* \
  repo/resultados/parte_b_todo_zero_teste_${T}.npy teste_b.log
echo FIM
