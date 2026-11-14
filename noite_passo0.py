"""Treino da noite, PASSO 0 (13/11): mede máquina, tempo por lote/época (A3, A6, A5) e memória por lote, em CPU. SÓ treino/validação (a1_cnn.carregar_indices corta o teste).
Nada é treinado de verdade (alguns passos de gradiente num modelo recém-criado, descartado). Não escreve em arquivo rastreado: acrescenta ao novo resultados/noite_status.txt.
Uso (raiz, .venv): python noite_passo0.py          # roda os 3 modelos, cada um num processo separado (para o pico de RAM não se misturar)
                  python noite_passo0.py medir A6  # um modelo
"""
import os
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
STATUS = RAIZ / "resultados" / "noite_status.txt"
LOTES = {"A3": 15, "A6": 15, "A5": 4}


def medir(nome):
    import keras
    import psutil

    import a1_cnn
    import arquiteturas
    from ecg_dados import GeradorECG
    if nome != "A3":
        a1_cnn.construir_modelo = arquiteturas.CONSTRUTORES[nome]
    tr, va, _ = a1_cnn.carregar_indices(partes=[0, 1, 2, 3])
    g = GeradorECG(tr, embaralhar=True, normalizar=True)
    proc = psutil.Process()
    m = a1_cnn.construir_modelo()
    X, y = g[0]
    m.train_on_batch(X, y)  # aquecimento
    rss0 = proc.memory_info().rss / 2**30
    n = LOTES[nome]
    t_ler = t_cal = 0.0
    for i in range(1, n + 1):
        t = time.perf_counter(); X, y = g[i]; t_ler += time.perf_counter() - t
        t = time.perf_counter(); m.train_on_batch(X, y); t_cal += time.perf_counter() - t
    gv = GeradorECG(va, normalizar=True)
    t = time.perf_counter()
    for i in range(3):
        Xv, _ = gv[i]; m.predict_on_batch(Xv)
    t_val = (time.perf_counter() - t) / 3
    lote = (t_ler + t_cal) / n
    ep = (len(g) * lote + len(gv) * t_val) / 60
    pico = getattr(proc.memory_info(), "peak_wset", proc.memory_info().rss) / 2**30
    print(f"RESULTADO {nome} | parâmetros {m.count_params():,} | {n} lotes: leitura {t_ler / n * 1000:.0f} ms + treino {t_cal / n * 1000:.0f} ms = {lote * 1000:.0f} ms por lote de 64 | "
          f"validação {t_val * 1000:.0f} ms por lote | {len(g)} lotes de treino + {len(gv)} de validação => ESTIMATIVA {ep:.1f} min por época | "
          f"RAM do processo depois do aquecimento {rss0:.2f} GiB, pico {pico:.2f} GiB (inclui TensorFlow e o modelo)", flush=True)


def main():
    import psutil
    linhas = [f"=== PASSO 0 ({time.strftime('%Y-%m-%d %H:%M')}) ==="]
    vm = psutil.virtual_memory()
    bat = psutil.sensors_battery()
    linhas.append(f"máquina: {psutil.cpu_count(logical=False)} núcleos físicos / {psutil.cpu_count()} lógicos; RAM total {vm.total / 2**30:.1f} GiB, livre agora {vm.available / 2**30:.1f} GiB; "
                  f"bateria: {'na tomada' if bat is None or bat.power_plugged else 'NA BATERIA (' + str(int(bat.percent)) + '%)'}")
    for nome in ("A3", "A6", "A5"):
        r = subprocess.run([sys.executable, str(Path(__file__)), "medir", nome], capture_output=True, text=True, cwd=RAIZ,
                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"})
        ok = [l for l in r.stdout.splitlines() if l.startswith("RESULTADO")]
        linhas.append(ok[0] if ok else f"FALHOU {nome}: {r.stderr[-400:]}")
        print(linhas[-1], flush=True)
    with open(STATUS, "a", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "medir":
        medir(sys.argv[2])
    else:
        assert not STATUS.exists() or "PASSO 0" not in STATUS.read_text(encoding="utf-8"), "PASSO 0 já registrado"
        main()
