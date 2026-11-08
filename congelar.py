"""Congela o candidato: muda o status do JSON para "congelado", grava data, hash do commit e SHA256 do modelo, e faz um commit LOCAL.

SÓ RODAR MANUALMENTE, DEPOIS que o Tom e a Renata decidirem os itens abertos (limite de likelihood, exclusão AFLT/IRBBB/ILBBB, folds,
normalização, regra de exclusão, número de seeds...). Nenhuma IA deve rodar este script: ele pede que você digite CONGELAR.
Depois de congelar, o ÚNICO jeito de avaliar o teste é: python avaliar_externo.py --json <json> --conjunto <...> --confirmo-congelado
O script NÃO faz push.

Uso (raiz do repositório, Python do .venv):  python congelar.py [--json congelado_candidato_A3.json]
PROVISÓRIO: este script nunca foi executado (código não rodado).
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).parent


def git(*args):
    r = subprocess.run(["git", *args], capture_output=True, text=True, cwd=RAIZ)
    if r.returncode != 0:
        sys.exit(f"git {' '.join(args)} falhou: {r.stderr.strip()}")
    return r.stdout.strip()


def sha256(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="congelado_candidato_A3.json")
    a = ap.parse_args()
    caminho_json = RAIZ / a.json
    cfg = json.loads(caminho_json.read_text(encoding="utf-8"))

    if cfg.get("status") != "candidato":
        sys.exit(f"RECUSADO: o status do JSON é '{cfg.get('status')}', só um 'candidato' pode ser congelado.")
    pendentes = [k for k, v in cfg.get("ptbxl", {}).items() if isinstance(v, str) and v.lower().startswith("a decidir")]
    if pendentes:
        sys.exit(f"RECUSADO: ainda há itens 'a decidir' em ptbxl: {pendentes}.")
    modelo = RAIZ / cfg["modelo"]["caminho"]
    if not modelo.exists():
        sys.exit(f"RECUSADO: modelo não encontrado: {modelo}")
    sha = sha256(modelo)
    if sha != cfg["modelo"]["sha256"]:
        sys.exit("RECUSADO: o SHA256 do modelo atual não bate com o do JSON candidato. Gere o candidato de novo e revise.")
    sujo = [l for l in git("status", "--porcelain").splitlines() if l[3:].strip() != a.json]
    if sujo:
        sys.exit("RECUSADO: há alterações não commitadas (o hash do commit precisa descrever o código exato):\n" + "\n".join(sujo))

    print(f"Vai congelar '{a.json}': modelo {cfg['modelo']['caminho']} (SHA256 {sha[:16]}...), commit {git('rev-parse', 'HEAD')[:10]}.")
    print("Depois disso o teste interno, o PTB-XL e o CODE-TEST poderão ser avaliados (uma vez). Isto NÃO faz push.")
    if input("Digite CONGELAR para confirmar: ").strip() != "CONGELAR":
        sys.exit("Cancelado. Nada foi alterado.")

    cfg["status"] = "congelado"
    cfg["congelado_em"] = datetime.now().astimezone().isoformat(timespec="seconds")
    cfg["congelado_commit"] = git("rev-parse", "HEAD")  # commit do código no momento do congelamento (o commit do JSON vem logo depois)
    cfg["modelo"]["sha256"] = sha
    caminho_json.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    git("add", a.json)
    git("commit", "-m", f"Congela modelo A3 (status congelado, SHA256 {sha[:12]})")
    print("Congelado e commitado localmente. Faça o push quando decidirem. Commit:", git("rev-parse", "--short", "HEAD"))


if __name__ == "__main__":
    main()
