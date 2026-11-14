"""Baixa o exams.csv e as partes da CODE-15% para code15/ (pensado para o Colab, mas roda em qualquer lugar).

Cada zip é baixado em pedaços paralelos (HTTP Range, mesma ideia da função `baixar` do projeto_ecg.ipynb),
conferido contra o tamanho e o md5 publicados pela API do Zenodo, extraído e apagado. Partes já extraídas são puladas.
Várias partes ao mesmo tempo (--simultaneas). Tudo é registrado em code15/download.log.

Uso: python colab_baixar.py --partes 0-17 --simultaneas 3
"""
import argparse
import hashlib
import json
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RAIZ = Path(__file__).parent
DADOS = RAIZ / "code15"
API = "https://zenodo.org/api/records/4916206"
ZENODO = "https://zenodo.org/records/4916206/files"


def log(msg):
    linha = f"{time.strftime('%H:%M:%S')} {msg}"
    print(linha, flush=True)
    with open(DADOS / "download.log", "a", encoding="utf-8") as f:
        f.write(linha + "\n")


def baixar(url, destino, conexoes=16):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"}), timeout=60) as r:
        total = int(r.headers["Content-Range"].split("/")[1])
    with open(destino, "wb") as f:
        f.truncate(total)
    passo = -(-total // conexoes)
    baixado = [0] * conexoes

    def pedaco(i):
        ini, fim = i * passo, min(total, (i + 1) * passo) - 1
        for _ in range(200):
            if baixado[i] >= fim - ini + 1:
                return
            try:
                pos = ini + baixado[i]
                req = urllib.request.Request(url, headers={"Range": f"bytes={pos}-{fim}"})
                with urllib.request.urlopen(req, timeout=60) as r, open(destino, "r+b") as f:
                    f.seek(pos)
                    while bloco := r.read(1 << 20):
                        f.write(bloco)
                        baixado[i] += len(bloco)
            except Exception:
                time.sleep(3)
        raise RuntimeError(f"pedaço {i} incompleto")

    with ThreadPoolExecutor(conexoes) as ex:
        list(ex.map(pedaco, range(conexoes)))
    assert sum(baixado) == total, "download incompleto"
    return total


def md5(caminho):
    h = hashlib.md5()
    with open(caminho, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def uma_parte(p, meta):
    h5 = DADOS / f"exams_part{p}.hdf5"
    if h5.exists():
        log(f"parte {p}: já existe, pulando")
        return
    nome = f"exams_part{p}.zip"
    z = DADOS / nome
    t0 = time.time()
    for tentativa in range(3):
        tam = baixar(f"{ZENODO}/{nome}?download=1", z)
        ok_tam, soma = tam == meta[nome]["size"], md5(z)
        if ok_tam and soma == meta[nome]["md5"]:
            break
        log(f"parte {p}: tamanho/md5 não conferem (tentativa {tentativa + 1}), baixando de novo")
    else:
        raise RuntimeError(f"parte {p}: md5 não confere após 3 tentativas")
    zipfile.ZipFile(z).extractall(DADOS)
    z.unlink()
    log(f"parte {p}: OK (md5 confere) em {(time.time() - t0) / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partes", default="0-17")
    ap.add_argument("--simultaneas", type=int, default=3)
    a = ap.parse_args()
    ini, fim = (int(x) for x in a.partes.split("-")) if "-" in a.partes else (int(a.partes),) * 2
    DADOS.mkdir(exist_ok=True)
    with urllib.request.urlopen(API, timeout=60) as r:
        meta = {f["key"]: dict(size=f["size"], md5=f["checksum"].split(":")[1]) for f in json.load(r)["files"]}
    if not (DADOS / "exams.csv").exists():
        baixar(f"{ZENODO}/exams.csv?download=1", DADOS / "exams.csv", conexoes=1)
        assert md5(DADOS / "exams.csv") == meta["exams.csv"]["md5"]
        log("exams.csv: OK (md5 confere)")
    t0 = time.time()
    with ThreadPoolExecutor(a.simultaneas) as ex:
        for f in [ex.submit(uma_parte, p, meta) for p in range(ini, fim + 1)]:
            f.result()
    log(f"TUDO PRONTO: partes {ini}-{fim} em {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
