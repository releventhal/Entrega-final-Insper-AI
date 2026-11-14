"""Treino da noite (13/11): fila A6 -> A5 na CPU, SÓ treino -> validação. Não abre nenhum teste (a1_cnn.carregar_indices corta o grupo teste na carga).
NÃO altera a1_cnn.py, arquiteturas.py, o A3, o JSON congelado nem arquivo rastreado (não escreve em experimentos.csv, não faz commit nem push).
O treino é `a1_cnn.treinar` (seed 42, partes 0 a 3, normalização A2, Adam, lote 64, early stopping na perda de validação com paciência 3, melhor época = menor
perda de validação); aqui só se troca o construtor do modelo (como em arquiteturas.py) e se ACRESCENTA, no fit, o BackupAndRestore (retomar) e um callback que grava
o histórico e o estado do early stopping a cada época. Nomes novos: A6_completo / A5_completo (nada de A6_*.csv da Renata é tocado).

  python noite_fila.py fila                      # supervisor: A6, depois A5 se sobrar tempo (orçamento total de 8 h de relógio)
  python noite_fila.py treinar --nome X --modelo A6 --max-min M --max-epocas E [--lote 64] [--n-treino N]   # um treino (usado pelo supervisor)
  python noite_fila.py avaliar --nome A6_completo   # avaliação na validação (IC por bootstrap de pacientes) contra o A3
"""
import argparse
import functools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
RES, LOGS, MODELOS = RAIZ / "resultados", RAIZ / "logs", RAIZ / "modelos"
STATUS = Path(os.environ.get("NOITE_STATUS", RES / "noite_status.txt"))   # NOITE_STATUS só para o teste de fumaça
ORCAMENTO_MIN = 480          # 8 h de relógio, contadas desde o início da fila
MARGEM_FIM_MIN = 20          # sobra para a avaliação final
EST_EPOCA_MIN = {"A6": 3.9, "A5": 18.6}   # PASSO 0 (resultados/noite_status.txt)
EPOCAS_A5_PLANEJADAS = 10    # estimativa do A5 = 10 épocas x 18,6 min (A3: parou na 7ª; B2z: 10)
ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}


def status(msg):
    with open(STATUS, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%d/%m %H:%M:%S')}] {msg}\n")


# ----------------------------------------------------------------------------------------------------------- treino (processo filho)
def treinar_filho(a):
    import keras

    import a1_cnn
    import arquiteturas
    from a1_cnn import LimiteDeTempo

    nome, modelo = a.nome, a.modelo
    backup = MODELOS / f"noite_backup_{nome}"
    arq_estado, arq_epocas = LOGS / f"noite_{nome}_estado.json", LOGS / f"noite_{nome}_epocas.jsonl"
    caminho_melhor = MODELOS / f"{nome}_melhor.keras"
    construtor = arquiteturas.CONSTRUTORES[modelo]
    a1_cnn.ROTULO_EXPERIMENTO[nome] = f"{arquiteturas.DESCRICAO[modelo]} [treino completo da noite de 13/11]"
    if a.lote != 64:
        a1_cnn.GeradorECG = functools.partial(a1_cnn.GeradorECG, lote=a.lote)
        status(f"ATENÇÃO >>> {nome}: lote {a.lote} em vez de 64. O PROTOCOLO DEIXOU DE SER IGUAL AO DO A3 (a partir deste ponto do treino).")

    class EstadoNoite(keras.callbacks.Callback):
        """Grava a cada época: histórico (jsonl), estado do early stopping (json) e uma linha no noite_status.txt; na retomada, restaura o estado."""

        def __init__(self, es, mc, lim):
            super().__init__()
            self.es, self.mc, self.lim = es, mc, lim
            self.hist, self.anterior, self.t_ep = [], [], time.time()

        def on_train_begin(self, logs=None):
            if arq_estado.exists():
                s = json.loads(arq_estado.read_text(encoding="utf-8"))
                self.hist = self.anterior = list(s["historia"])
                self.es.best, self.es.wait, self.es.best_epoch = s["es_best"], s["es_wait"], s["es_best_epoch"]
                self.mc.best = s["mc_best"]
                if caminho_melhor.exists():
                    self.es.best_weights = keras.models.load_model(caminho_melhor, compile=False).get_weights()
                status(f"RETOMADO {nome} depois da época {len(self.hist)} (estado do early stopping restaurado: melhor val_loss {self.es.best:.4f}, espera {self.es.wait}/{self.es.patience})")
            self.t_ep = time.time()

        def on_epoch_begin(self, epoch, logs=None):
            self.t_ep = time.time()

        def on_epoch_end(self, epoch, logs=None):
            seg = time.time() - self.t_ep
            self.hist = self.hist + [dict(epoca=epoch + 1, loss=float(logs["loss"]), val_loss=float(logs["val_loss"]), segundos=round(seg, 1), hora=time.strftime("%d/%m %H:%M:%S"))]
            if len(self.hist) != epoch + 1:
                status(f"AVISO {nome}: histórico com {len(self.hist)} épocas, mas o Keras está na época {epoch + 1}")
            melhor = min(range(len(self.hist)), key=lambda i: self.hist[i]["val_loss"])
            with open(arq_epocas, "a", encoding="utf-8") as f:
                f.write(json.dumps(self.hist[-1], ensure_ascii=False) + "\n")
            arq_estado.write_text(json.dumps(dict(historia=self.hist, es_best=float(self.es.best), es_wait=int(self.es.wait), es_best_epoch=int(self.es.best_epoch),
                                                  mc_best=float(self.mc.best), termino=None), ensure_ascii=False), encoding="utf-8")
            status(f"{nome} época {epoch + 1}/{a.max_epocas}: perda de treino {logs['loss']:.4f} | perda de validação {logs['val_loss']:.4f} "
                   f"(melhor {self.hist[melhor]['val_loss']:.4f} na época {melhor + 1}; paciência {self.es.wait}/{self.es.patience}) | {seg / 60:.1f} min")

        def on_train_end(self, logs=None):
            if self.es.stopped_epoch > 0:
                termino = "early stopping (paciência 3)"
            elif self.lim.estourou:
                termino = "LIMITE DE TEMPO"
            elif len(self.hist) >= a.max_epocas:
                termino = "LIMITE DE ÉPOCAS"
            else:
                termino = "interrompido por outro motivo"
            s = json.loads(arq_estado.read_text(encoding="utf-8")) if arq_estado.exists() else {}
            s["termino"] = termino
            arq_estado.write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
            status(f"FIM_TREINO {nome}: {len(self.hist)} épocas; terminou por {termino}")

    def construir(seed=a1_cnn.SEED, pesos=None):
        m = construtor(seed, pesos)
        fit_original = m.fit

        def fit(*args, **kw):
            cbs = list(kw["callbacks"])
            es = next(c for c in cbs if isinstance(c, keras.callbacks.EarlyStopping))
            mc = next(c for c in cbs if isinstance(c, keras.callbacks.ModelCheckpoint))
            lim = next(c for c in cbs if isinstance(c, LimiteDeTempo))
            est = EstadoNoite(es, mc, lim)
            kw["callbacks"] = cbs + [keras.callbacks.BackupAndRestore(str(backup), save_freq="epoch", delete_checkpoint=False), est]
            h = fit_original(*args, **kw)
            if est.anterior:  # depois de uma retomada o History só tem as épocas novas: junta as anteriores (treinar usa len() e argmin)
                n = len(est.anterior)
                for k in ("loss", "val_loss"):
                    h.history[k] = [e[k] for e in est.anterior] + h.history[k]
                assert len(h.history["val_loss"]) >= n
            return h

        m.fit = fit
        return m

    a1_cnn.construir_modelo = construir
    status(f"INICIO_TREINO {nome} ({modelo}): partes 0-3, seed 42, lote {a.lote}, máximo {a.max_epocas} épocas e {a.max_min:.0f} min"
           + (f", SÓ {a.n_treino} exames de treino (TESTE DE FUMAÇA)" if a.n_treino else ""))
    df, _, _ = a1_cnn.treinar(nome, normalizar=True, n_treino=a.n_treino, max_min=a.max_min, max_epocas=a.max_epocas, partes=[0, 1, 2, 3], seed=a1_cnn.SEED)
    print("treino e avaliação interna do treinar concluídos", flush=True)


# ----------------------------------------------------------------------------------------------------------- avaliação (processo filho)
def avaliar(nome):
    import numpy as np
    import pandas as pd

    import avaliar_externo as ae
    from a0_baselines import melhor_limiar_f1
    from ecg_dados import ROTULOS
    from robustez_comum import carregar_val, limiares_json, reamostras
    from sklearn.metrics import average_precision_score, roc_auc_score

    S3, Y, _, _, pid = carregar_val()
    S = np.load(RES / f"{nome}_previsoes_validacao.npy").astype(float)
    assert S.shape == S3.shape == Y.shape, (S.shape, S3.shape)
    Yf, mask = Y.astype(float), np.ones(Y.shape, bool)
    lim = {r: melhor_limiar_f1(Y[:, k], S[:, k]) for k, r in enumerate(ROTULOS)}   # limiar do novo modelo: F1 máximo na validação (como o A3 original)
    lim3 = limiares_json()                                                          # A3: limiares do JSON
    pt = ae.com_macro(ae.metricas_exame(Yf, S, lim, mask)); pt3 = ae.com_macro(ae.metricas_exame(Yf, S3.astype(float), lim3, mask))
    t0 = time.time()
    ic, n_ok = ae.bootstrap(Yf, S, lim, mask, pid)
    ic3, n_ok3 = ae.bootstrap(Yf, S3.astype(float), lim3, mask, pid)
    status(f"{nome}: bootstrap por paciente (avaliar_externo.bootstrap, 1.000 reamostras): {n_ok} válidas (novo), {n_ok3} (A3); {time.time() - t0:.0f} s")
    # diferença pareada (mesmas reamostras de pacientes) em AUROC e AUPRC
    idxs = reamostras(pid, 1000, 42)
    d = {"auroc": [], "auprc": []}
    for ix in idxs:
        a_, b_ = [], []
        for k in range(6):
            y = Y[ix, k]
            if y.min() == y.max():
                a_.append((np.nan, np.nan)); b_.append((np.nan, np.nan)); continue
            a_.append((roc_auc_score(y, S[ix, k]), average_precision_score(y, S[ix, k]))); b_.append((roc_auc_score(y, S3[ix, k]), average_precision_score(y, S3[ix, k])))
        a_, b_ = np.array(a_), np.array(b_)
        d["auroc"].append(np.append(a_[:, 0] - b_[:, 0], np.nanmean(a_[:, 0]) - np.nanmean(b_[:, 0])))
        d["auprc"].append(np.append(a_[:, 1] - b_[:, 1], np.nanmean(a_[:, 1]) - np.nanmean(b_[:, 1])))
    linhas = []
    for j, r in enumerate(ROTULOS + ["MACRO"]):
        for c in ae.METRICAS:
            l = dict(rotulo=r, metrica=c, novo=pt[r][c], novo_ic_inf=ic[(r, c)][0], novo_ic_sup=ic[(r, c)][1], A3=pt3[r][c], A3_ic_inf=ic3[(r, c)][0], A3_ic_sup=ic3[(r, c)][1])
            if c in d:
                v = np.array(d[c])[:, j]
                l |= dict(dif_novo_menos_A3=pt[r][c] - pt3[r][c], dif_ic_inf=np.nanpercentile(v, 2.5), dif_ic_sup=np.nanpercentile(v, 97.5))
            linhas.append(l)
    tab = pd.DataFrame(linhas)
    saida = RES / f"noite_{nome}_avaliacao.csv"
    assert not saida.exists(), f"já existe, não sobrescrevo: {saida}"
    tab.to_csv(saida, index=False, float_format="%.4f")
    estado = json.loads((LOGS / f"noite_{nome}_estado.json").read_text(encoding="utf-8")) if (LOGS / f"noite_{nome}_estado.json").exists() else {}
    termino = estado.get("termino") or "desconhecido"
    es_ok = termino.startswith("early stopping")
    hist = estado.get("historia", [])
    melhor_ep = min(range(len(hist)), key=lambda i: hist[i]["val_loss"]) + 1 if hist else None
    linhas_s = [f"=== AVALIAÇÃO {nome} na VALIDAÇÃO (8.158 exames; limiar = F1 máximo na validação; A3 com os limiares do JSON; IC 95% por bootstrap de pacientes) ===",
                f"Treino: {len(hist)} épocas, melhor época {melhor_ep}, terminou por {termino}."]
    for j, r in enumerate(ROTULOS + ["MACRO"]):
        g = lambda c: tab[(tab.rotulo == r) & (tab.metrica == c)].iloc[0]
        au, ap = g("auroc"), g("auprc")
        sinal = lambda x: "IC acima de zero" if x.dif_ic_inf > 0 else ("IC abaixo de zero" if x.dif_ic_sup < 0 else "IC inclui zero")
        linhas_s.append(f"  {r:6s} AUROC {au.novo:.3f} ({au.novo_ic_inf:.3f}-{au.novo_ic_sup:.3f}) vs A3 {au.A3:.3f} ({au.A3_ic_inf:.3f}-{au.A3_ic_sup:.3f}) | dif {au.dif_novo_menos_A3:+.3f} ({au.dif_ic_inf:+.3f} a {au.dif_ic_sup:+.3f}; {sinal(au)})"
                        f" || AUPRC {ap.novo:.3f} ({ap.novo_ic_inf:.3f}-{ap.novo_ic_sup:.3f}) vs A3 {ap.A3:.3f} ({ap.A3_ic_inf:.3f}-{ap.A3_ic_sup:.3f}) | dif {ap.dif_novo_menos_A3:+.3f} ({ap.dif_ic_inf:+.3f} a {ap.dif_ic_sup:+.3f}; {sinal(ap)})")
    if es_ok:
        linhas_s.append("  LEITURA (mecânica, sem causa): o treino terminou por early stopping, então a comparação com o A3 usa um modelo no ponto de parada do mesmo critério; "
                        "as linhas acima com 'IC acima/abaixo de zero' são diferenças medidas (validação, otimista; uma seed).")
    else:
        linhas_s.append(f"  NÃO CONCLUSIVO: o treino terminou por {termino}, NÃO por early stopping; não declarar o modelo 'pior' nem 'melhor' que o A3 (pode estar subtreinado).")
    status("\n".join(linhas_s))
    print("\n".join(linhas_s))
    status(f"FIM_AVALIACAO {nome}: resultados/noite_{nome}_avaliacao.csv")


# ----------------------------------------------------------------------------------------------------------- supervisor
def impedir_suspensao():
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: o Windows não suspende enquanto o supervisor viver
        status("suspensão automática bloqueada pelo supervisor (SetThreadExecutionState); fechar a tampa do notebook ou acabar a bateria AINDA para o treino")
    except Exception as e:
        status(f"AVISO: não consegui bloquear a suspensão ({e})")


def rodar(cmd, log, deadline):
    import psutil
    with open(log, "a", encoding="utf-8") as lg:
        p = subprocess.Popen(cmd, stdout=lg, stderr=subprocess.STDOUT, cwd=RAIZ, env=ENV)
        ultimo_aviso = 0
        while p.poll() is None:
            time.sleep(30)
            livre = psutil.virtual_memory().available / 2**30
            if livre < 1.0 and time.time() - ultimo_aviso > 600:
                status(f"AVISO: só {livre:.1f} GiB de RAM livre")
                ultimo_aviso = time.time()
        return p.returncode


def treinar_com_retry(nome, modelo, max_epocas, teto_min, deadline):
    log = LOGS / f"noite_{nome}.log"
    for tentativa, lote in enumerate((64, 32)):
        restante = (deadline - time.time()) / 60 - MARGEM_FIM_MIN
        max_min = min(restante, teto_min)
        if max_min < 10:
            status(f"{nome}: sem tempo útil no orçamento (restam {restante:.0f} min); parei.")
            return False
        cmd = [sys.executable, str(Path(__file__)), "treinar", "--nome", nome, "--modelo", modelo, "--max-min", f"{max_min:.1f}", "--max-epocas", str(max_epocas), "--lote", str(lote)]
        rc = rodar(cmd, log, deadline)
        if rc == 0:
            return True
        cauda = " | ".join(l.strip() for l in log.read_text(encoding="utf-8", errors="replace").splitlines()[-4:])
        status(f"ERRO {nome}: o processo de treino saiu com código {rc} (tentativa {tentativa + 1}/2). Últimas linhas do log: {cauda}")
        if tentativa == 0:
            status(f"ATENÇÃO >>> {nome}: vou tentar UMA vez com lote 32 e retomar do último checkpoint; SE DER CERTO, O PROTOCOLO NÃO É MAIS IGUAL AO DO A3.")
    status(f"PARADO: {nome} falhou duas vezes; motivo no log logs/noite_{nome}.log. Fila interrompida.")
    return False


def fila():
    t0 = time.time()
    deadline = t0 + ORCAMENTO_MIN * 60
    (LOGS / "noite_fila.pid").write_text(str(os.getpid()))
    status(f"=== FILA INICIADA (pid {os.getpid()}); orçamento de {ORCAMENTO_MIN / 60:.0f} h: termina até {time.strftime('%d/%m %H:%M', time.localtime(deadline))} ===")
    impedir_suspensao()
    feitos = []
    for modelo, max_epocas, teto in (("A6", 30, 300), ("A5", 20, 10 ** 6)):
        nome = f"{modelo}_completo"
        restante = (deadline - time.time()) / 60
        if modelo == "A5":
            est = EST_EPOCA_MIN["A5"] * EPOCAS_A5_PLANEJADAS + MARGEM_FIM_MIN
            if restante <= est:
                status(f"A5 NÃO lançado: restam {restante:.0f} min <= estimativa {est:.0f} min ({EPOCAS_A5_PLANEJADAS} épocas x {EST_EPOCA_MIN['A5']} min + {MARGEM_FIM_MIN} de margem).")
                break
            status(f"A5 será lançado: restam {restante:.0f} min > estimativa {est:.0f} min.")
        if not (RES / f"{nome}_previsoes_validacao.npy").exists():
            if not treinar_com_retry(nome, modelo, max_epocas, teto, deadline):
                break
        if not (RES / f"noite_{nome}_avaliacao.csv").exists():
            rc = rodar([sys.executable, str(Path(__file__)), "avaliar", "--nome", nome], LOGS / f"noite_{nome}_avaliacao.log", deadline)
            if rc != 0:
                status(f"ERRO na avaliação de {nome} (código {rc}); ver logs/noite_{nome}_avaliacao.log. Fila interrompida.")
                break
        feitos.append(nome)
    status(f"=== RESUMO FINAL ({time.strftime('%d/%m %H:%M')}; {(time.time() - t0) / 3600:.1f} h de relógio) ===")
    for nome in feitos:
        e = json.loads((LOGS / f"noite_{nome}_estado.json").read_text(encoding="utf-8")) if (LOGS / f"noite_{nome}_estado.json").exists() else {}
        status(f"  {nome}: {len(e.get('historia', []))} épocas, terminou por {e.get('termino')}; avaliação em resultados/noite_{nome}_avaliacao.csv")
    status("FIM_TUDO")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=["fila", "treinar", "avaliar"])
    ap.add_argument("--nome")
    ap.add_argument("--modelo", choices=["A5", "A6"])
    ap.add_argument("--max-min", type=float, default=300)
    ap.add_argument("--max-epocas", type=int, default=30)
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--n-treino", type=int, default=None)
    a = ap.parse_args()
    {"fila": lambda: fila(), "treinar": lambda: treinar_filho(a), "avaliar": lambda: avaliar(a.nome)}[a.modo]()
