"""Robustez, tarefa 6: script ÚNICO de treino para rodar na GPU depois (NÃO foi rodado de verdade; só teste de fumaça de 1 lote e 1 época na CPU).
Reaproveita ecg_dados.py (GeradorECG, normalizar_exame) e a CNN/perda/avaliação de a1_cnn.py; não altera nenhum deles. Todas as saídas levam
o prefixo robustez_06_ (resultados/ e modelos/) e o script ABORTA se o arquivo já existe. Não toca em experimentos.csv, no JSON, no split nem no A3 congelado.
Só treino e validação (o grupo teste é cortado em a1_cnn.carregar_indices e nunca chega aqui).

Escolha da configuração (--arq): A1 = parte 0, sem normalização; A2 = parte 0, normalização por exame e derivação; A3 = partes 0-3, normalização.
  (a) seeds extras do A1/A2:        --arq A1 --seed 43     |   --arq A2 --seed 44        (seed 42 já existe; só a seed muda)
  (b) A4 = A3 + perda ponderada:     --arq A3 --peso-por-classe   (peso = negativos/positivos de cada alteração no treino, teto 10; FORA DAS AULAS; nunca treinado)
  (c) augmentation (FORA DAS AULAS): --augment  só no treino, no sinal CRU e ANTES da normalização oficial: escala x U(0,7; 1,4), rotação circular da região com
      sinal de até ±1 s, ruído gaussiano com desvio U(0; 0,2) do desvio da derivação. Com normalização (A2/A3) a escala não tem efeito (a normalização a anula);
      ela só importa no A1.
  (d) validação cruzada agrupada por paciente: --kfold 5 [--fold i]   GroupKFold(k=5, shuffle, seed 42) por patient_id SÓ sobre treino + validação (o teste fica fora).
      EXTENSÃO DE PROTOCOLO: o split oficial 70/10/20 segue valendo para os números oficiais; o k-fold só mede a variação entre divisões.
      Em cada fold: 4/5 treinam (10% desses, sorteados POR PACIENTE, viram validação interna para early stopping e limiar de F1 máximo);
      o 1/5 de fora só é avaliado no fim, com o limiar da validação interna. A tabela de folds é gravada em resultados/robustez_06_folds_p<partes>.csv.
Fumaça: --fumaca --saida <pasta>  (1 lote de treino, 1 época, saídas só na pasta dada, sem métricas).
Só a tabela de folds, sem treinar: --so-folds
"""
import argparse
import time
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from keras.utils import PyDataset
from sklearn.model_selection import GroupKFold

from a0_baselines import melhor_limiar_f1
from a1_cnn import LimiteDeTempo, bce_ponderada, carregar_indices, construir_modelo, tabela_metricas
from ecg_dados import ROTULOS, SEED, GeradorECG, normalizar_exame
from robustez_02_estresse import ctx, deslocar, ruido
from robustez_comum import MODELOS, RES, salvar_novo

ARQS = {"A1": dict(normalizar=False, partes=[0]), "A2": dict(normalizar=True, partes=[0]), "A3": dict(normalizar=True, partes=[0, 1, 2, 3])}
PROTOCOLO_KFOLD = "extensão de protocolo: k-fold agrupado por paciente sobre treino+validação (teste fora); o split oficial 70/10/20 segue valendo"


class GeradorAug(PyDataset):
    """Envolve um GeradorECG(normalizar=False): augmentation no sinal cru (escala, rotação, ruído) e depois, se `normalizar`, a normalização oficial."""

    def __init__(self, base, normalizar, seed=SEED, **kw):
        super().__init__(**kw)
        self.base, self.normalizar, self.seed = base, normalizar, seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        X, y = self.base[i]
        out = []
        for k, x in enumerate(X):
            rng = np.random.default_rng([self.seed, self.base.epoca, i, k])
            ix, ok, sd = ctx(x)
            if len(ix):
                x = x * np.float32(rng.uniform(0.7, 1.4))
                x = deslocar(rng.uniform(-1, 1))(x, ix, ok, sd, rng)
                x = ruido(rng.uniform(0, 0.2))(x, ix, ok, sd * 1.0, rng)
            out.append(normalizar_exame(x) if self.normalizar else x)
        return np.stack(out), y

    def on_epoch_end(self):
        self.base.on_epoch_end()


def tabela_folds(partes, k=5):
    """Treino + validação oficiais (já sem os exames todo-zero) com o fold de cada exame. GroupKFold por patient_id."""
    tr, va, _ = carregar_indices(partes=partes)
    d = pd.concat([tr.assign(grupo_original="treino"), va.assign(grupo_original="validação")], ignore_index=True)
    assert set(d.grupo).issubset({"treino", "validação"}), "grupo teste no índice!"
    fold = np.full(len(d), -1)
    for f, (_, te) in enumerate(GroupKFold(n_splits=k, shuffle=True, random_state=SEED).split(d, groups=d.patient_id)):
        fold[te] = f
    d["fold"] = fold
    assert (fold >= 0).all() and d.groupby("patient_id").fold.nunique().max() == 1, "paciente em mais de um fold"
    return d


def gravar_folds(partes, k, pasta):
    d = tabela_folds(partes, k)
    arq = Path(pasta) / f"robustez_06_folds_p{''.join(map(str, partes))}_k{k}.csv"
    if not arq.exists():
        salvar_novo(d[["exam_id", "patient_id", "parte", "linha", "grupo_original", "fold"]], arq)
    return d, arq


def gerador(info, cfg, treino, augment, seed, lote=64, **kw):
    if treino and augment:
        return GeradorAug(GeradorECG(info, lote, embaralhar=True, normalizar=False, seed=seed), cfg["normalizar"], seed, **kw)
    return GeradorECG(info, lote, embaralhar=treino, normalizar=cfg["normalizar"], seed=seed, **kw)


def treinar_um(nome, tr, va, cfg, a, pasta_mod, fumaca=False):
    """Treina em `tr`, early stopping em `va`; devolve (modelo carregado do melhor checkpoint, histórico, minutos)."""
    kw = dict(workers=a.workers, use_multiprocessing=a.workers > 1)
    pesos = None
    if a.peso_por_classe:  # A4: só do treino
        pos = tr[ROTULOS].sum().to_numpy()
        pesos = np.minimum((len(tr) - pos) / pos, 10.0)
        print("pesos por alteração:", dict(zip(ROTULOS, pesos.round(2))), flush=True)
    m = construir_modelo(a.seed, pesos)
    g_tr, g_va = gerador(tr, cfg, True, a.augment, a.seed, **kw), gerador(va, cfg, False, False, a.seed, **kw)
    caminho = pasta_mod / f"{nome}_melhor.keras"
    assert not caminho.exists(), f"já existe, não sobrescrevo: {caminho}"
    cb = [keras.callbacks.EarlyStopping("val_loss", patience=a.paciencia, restore_best_weights=True),
          keras.callbacks.ModelCheckpoint(caminho, monitor="val_loss", save_best_only=True), LimiteDeTempo(a.max_min)]
    if not fumaca:
        log = Path(a.saida) / f"{nome}_log_por_epoca.csv"
        assert not log.exists(), f"já existe, não sobrescrevo: {log}"
        cb.append(keras.callbacks.CSVLogger(log))
    t0 = time.perf_counter()
    h = m.fit(g_tr, validation_data=g_va, epochs=1 if fumaca else a.max_epocas, steps_per_epoch=1 if fumaca else None,
              validation_steps=1 if fumaca else None, verbose=2, callbacks=cb).history
    minutos = (time.perf_counter() - t0) / 60
    return keras.models.load_model(caminho, compile=False), h, minutos, g_va


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arq", choices=ARQS, default="A3")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--peso-por-classe", action="store_true")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--kfold", type=int, default=0)
    ap.add_argument("--fold", type=int, default=None, help="só este fold (0 a k-1); sem isto roda todos")
    ap.add_argument("--so-folds", action="store_true")
    ap.add_argument("--max-epocas", type=int, default=15)
    ap.add_argument("--max-min", type=float, default=240)
    ap.add_argument("--paciencia", type=int, default=3)
    ap.add_argument("--workers", type=int, default=1, help="leitura/augmentation em paralelo (útil na GPU, onde a leitura vira o gargalo)")
    ap.add_argument("--fumaca", action="store_true")
    ap.add_argument("--saida", default=str(RES))
    a = ap.parse_args()
    cfg = ARQS[a.arq]
    a.saida = Path(a.saida)
    a.saida.mkdir(parents=True, exist_ok=True)
    pasta_mod = (a.saida / "modelos") if a.fumaca else MODELOS
    pasta_mod.mkdir(exist_ok=True, parents=True)
    nome = f"robustez_06_{'A4' if a.peso_por_classe and a.arq == 'A3' else a.arq}{'+peso' if a.peso_por_classe and a.arq != 'A3' else ''}" \
           f"{'+aug' if a.augment else ''}_seed{a.seed}{f'_k{a.kfold}' if a.kfold else ''}"
    print("configuração:", cfg, "| nome:", nome, flush=True)

    if a.so_folds or a.kfold:
        d, arq = gravar_folds(cfg["partes"], a.kfold or 5, RES if not a.fumaca else a.saida)
        print(f"tabela de folds: {arq} | exames por fold: {d.fold.value_counts().sort_index().tolist()} | pacientes por fold: {d.groupby('fold').patient_id.nunique().tolist()}")
        if a.so_folds:
            return
    if not a.kfold:
        tr, va, rel = carregar_indices(partes=cfg["partes"])
        print("descartes (só todo-zero):", rel, flush=True)
        if a.fumaca:
            tr, va = tr.iloc[:64], va.iloc[:64]
        m, h, minutos, g_va = treinar_um(nome, tr, va, cfg, a, pasta_mod, a.fumaca)
        S = m.predict(g_va, verbose=0)
        assert S.shape == (len(va), 6)
        if a.fumaca:
            print(f"FUMAÇA OK: 1 lote + 1 época em {minutos:.2f} min; previsão {S.shape}, perda de validação {h['val_loss'][0]:.4f}")
            return
        df = tabela_metricas(nome, g_va.Y, S)
        df["minutos_treino"], df["epocas"], df["melhor_epoca"] = round(minutos, 1), len(h["loss"]), int(np.argmin(h["val_loss"])) + 1
        salvar_novo(df, a.saida / f"{nome}_metricas.csv")
        np.save(a.saida / f"{nome}_previsoes_validacao.npy", S.astype(np.float32))
        print(df.round(3).to_string(index=False))
        return

    # --- k-fold agrupado por paciente (extensão de protocolo)
    macros = []
    for f in range(a.kfold) if a.fold is None else [a.fold]:
        fora = d[d.fold == f].reset_index(drop=True)
        dentro = d[d.fold != f]
        rng = np.random.default_rng(SEED + f)  # validação interna: 10% dos PACIENTES de dentro
        pac = np.sort(dentro.patient_id.unique())
        interna = set(rng.choice(pac, max(1, len(pac) // 10), replace=False))
        tr = dentro[~dentro.patient_id.isin(interna)].reset_index(drop=True)
        vi = dentro[dentro.patient_id.isin(interna)].reset_index(drop=True)
        assert not set(tr.patient_id) & (set(vi.patient_id) | set(fora.patient_id)) and not set(vi.patient_id) & set(fora.patient_id)
        if a.fumaca:
            tr, vi, fora = tr.iloc[:64], vi.iloc[:64], fora.iloc[:64]
        print(f"fold {f}: treino {len(tr)} | validação interna {len(vi)} | fora {len(fora)} exames", flush=True)
        nf = f"{nome}_fold{f}"
        m, h, minutos, g_vi = treinar_um(nf, tr, vi, cfg, a, pasta_mod, a.fumaca)
        g_fora = gerador(fora, cfg, False, False, a.seed)
        S_fora = m.predict(g_fora, verbose=0)
        if a.fumaca:
            print(f"FUMAÇA OK (k-fold, fold {f}): 1 lote + 1 época em {minutos:.2f} min; previsão do fold de fora {S_fora.shape}")
            return
        S_vi = m.predict(g_vi, verbose=0)
        lim = {r: melhor_limiar_f1(g_vi.Y[:, k].astype(int), S_vi[:, k]) for k, r in enumerate(ROTULOS)}  # limiar da validação interna
        df = tabela_metricas(nf, g_fora.Y, S_fora, lim)
        df["fold"], df["minutos_treino"], df["protocolo"] = f, round(minutos, 1), PROTOCOLO_KFOLD
        salvar_novo(df, a.saida / f"{nf}_metricas.csv")
        np.save(a.saida / f"{nf}_previsoes_fora.npy", S_fora.astype(np.float32))
        macros.append(df[df.rotulo == "MACRO"])
        print(df.round(3).to_string(index=False), flush=True)
    if len(macros) > 1:
        mm = pd.concat(macros)
        resumo = pd.DataFrame({c: [mm[c].mean(), mm[c].std(ddof=1)] for c in ["auroc", "auprc", "f1", "sensibilidade", "precisao"]}, index=["media", "desvio"]).reset_index()
        salvar_novo(resumo, a.saida / f"{nome}_resumo_macro.csv")
        print(resumo.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
