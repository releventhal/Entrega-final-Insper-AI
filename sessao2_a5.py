"""Sessão 2 (14/11, tarde): depois da retomada do A5_completo. Só validação; teste nunca lido.

  python sessao2_a5.py log           # resultados/A5_completo_log_por_epoca_consolidado.csv (épocas 1-3 do backup + 4-10 + novas), conferido contra o estado
  python sessao2_a5.py experimentos  # acrescenta as linhas do A5_completo em resultados/experimentos.csv (linhas antigas iguais; aborta se o nome já existe)
"""
import json
import sys

import numpy as np
import pandas as pd

RES = __import__("pathlib").Path(__file__).parent / "resultados"
ESTADO = RES.parent / "logs" / "noite_A5_completo_estado.json"


def log():
    partes = [RES / "A5_completo_log_por_epoca_backup_epocas1a3_20261114_sessao2.csv",
              RES / "A5_completo_log_por_epoca_epocas4a10_20261114.csv",
              RES / "A5_completo_log_por_epoca_epocas11a12_20261114.csv"]  # cópia do log reescrito pela retomada (só as épocas novas); o original voltou ao do git
    d = pd.concat([pd.read_csv(p).assign(origem=p.name) for p in partes], ignore_index=True).drop_duplicates(["epoch", "loss", "val_loss"])
    d["epoca"] = d.epoch + 1
    h = pd.DataFrame(json.loads(ESTADO.read_text(encoding="utf-8"))["historia"])
    assert list(d.epoca) == list(h.epoca) == list(range(1, len(h) + 1)), (list(d.epoca), list(h.epoca))
    assert np.allclose(d.val_loss, h.val_loss, atol=1e-9) and np.allclose(d.loss, h.loss, atol=1e-9)
    saida = RES / "A5_completo_log_por_epoca_consolidado.csv"
    assert not saida.exists()
    d.merge(h[["epoca", "segundos", "hora"]], on="epoca")[["epoca", "loss", "val_loss", "segundos", "hora", "origem"]].to_csv(saida, index=False)
    print(pd.read_csv(saida).to_string(index=False))


def experimentos():
    from a0_baselines import melhor_limiar_f1, metricas
    from ecg_dados import ROTULOS
    from robustez_comum import carregar_val
    arq = RES / "experimentos.csv"
    antigo = pd.read_csv(arq)
    _, Y, _, _, _ = carregar_val()
    S = np.load(RES / "A5_completo_previsoes_validacao.npy").astype(np.float64)
    e = json.loads(ESTADO.read_text(encoding="utf-8"))
    h, termino = e["historia"], e["termino"]
    rot = "" if termino.startswith("early stopping") else " | PARCIAL, NÃO CONCLUSIVO (sem early stopping)"
    nome = f"A5_completo ResNet pequena (partes 0 a 3; sessão 2 de 14/11; terminou por {termino}){rot}"
    assert nome not in set(antigo.experimento)
    df = pd.DataFrame([dict(experimento=nome, rotulo=r, n_val=len(Y), positivos_val=int(Y[:, k].sum()),
                            **metricas(Y[:, k], S[:, k], melhor_limiar_f1(Y[:, k], S[:, k]))) for k, r in enumerate(ROTULOS)])
    mac = df[["auroc", "auprc", "sensibilidade", "precisao", "f1", "especificidade", "acuracia"]].mean()
    df = pd.concat([df, pd.DataFrame([dict(experimento=nome, rotulo="MACRO", n_val=len(Y), **mac)])], ignore_index=True)
    df = df.assign(dados="partes 0 a 3", n_treino=55561, epocas=len(h), melhor_epoca=1 + int(np.argmin([x["val_loss"] for x in h])),
                   minutos_treino=sum(x["segundos"] for x in h) / 60)
    novo = pd.concat([antigo, df], ignore_index=True)
    assert novo.iloc[:len(antigo)].equals(antigo)
    novo.to_csv(arq, index=False, float_format="%.4f")
    print(df[["rotulo", "auroc", "auprc", "f1", "epocas", "melhor_epoca"]].round(4).to_string(index=False))


if __name__ == "__main__":
    {"log": log, "experimentos": experimentos}[sys.argv[1]]()
