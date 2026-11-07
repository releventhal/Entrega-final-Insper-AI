"""Índice dos exames, lista de exames zerados e gerador em lotes (T9).

Nada aqui treina modelo. Os sinais só são lidos pelo gerador e por `listar_zerados`.
"""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from keras.utils import Sequence

DADOS = Path(__file__).parent / "code15"
ROTULOS = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF"]
SEED = 42
PARTES = [0, 1, 2, 3]


def tabela_com_grupo():
    """exams.csv inteiro + coluna `grupo`. Mesma fórmula do notebook (seção 5):
    cada paciente da tabela INTEIRA é sorteado uma vez, então adicionar partes não muda o grupo de ninguém."""
    tabela = pd.read_csv(DADOS / "exams.csv")
    pacientes = np.sort(tabela["patient_id"].unique())
    sorteio = np.random.default_rng(SEED).random(len(pacientes))
    grupo = pd.Series(np.select([sorteio < 0.7, sorteio < 0.8], ["treino", "validação"], "teste"), index=pacientes)
    tabela["grupo"] = tabela["patient_id"].map(grupo).values
    return tabela


def construir_indice(partes=PARTES, tabela=None):
    """Uma linha por exame: (parte, linha no HDF5) + idade, sexo, rótulos, grupo. Só lê `exam_id`, não os sinais.
    A linha de preenchimento (exam_id = 0) de cada parte fica de fora."""
    tabela = tabela_com_grupo() if tabela is None else tabela
    blocos = []
    for p in partes:
        with h5py.File(DADOS / f"exams_part{p}.hdf5", "r") as f:
            ids = f["exam_id"][:]
        blocos.append(pd.DataFrame({"exam_id": ids, "parte": p, "linha": np.arange(len(ids))}))
    idx = pd.concat(blocos, ignore_index=True)
    idx = idx[idx["exam_id"].isin(tabela["exam_id"])]  # tira o exam_id = 0
    idx = idx.merge(tabela, on="exam_id", how="left", validate="one_to_one")
    return idx.sort_values(["parte", "linha"], ignore_index=True)


def listar_zerados(partes=PARTES, passo=500):
    """Passada única pelos sinais. Devolve array int32 (exam_id, parte, linha, n_derivações_zeradas),
    só dos exames com pelo menos 1 derivação zerada. NÃO descarta nada."""
    achados = []
    for p in partes:
        with h5py.File(DADOS / f"exams_part{p}.hdf5", "r") as f:
            ds, ids = f["tracings"], f["exam_id"][:]
            for ini in range(0, len(ids), passo):
                x = ds[ini:ini + passo]                                  # float32: o max evita o estouro da soma em float16
                n_zer = (np.abs(x).max(axis=1) == 0).sum(axis=1)         # (exames,) derivações com o sinal todo zero
                for k in np.where((n_zer > 0) & (ids[ini:ini + passo] != 0))[0]:
                    achados.append((ids[ini + k], p, ini + k, n_zer[k]))
    return np.array(achados, dtype=np.int32).reshape(-1, 4)


class GeradorECG(Sequence):
    """Lotes de `lote` exames lidos direto do HDF5 (float32), sem carregar parte nenhuma inteira.

    - Treino: embaralha a cada época com seed fixa (SEED + nº da época), então dá para reproduzir.
    - Validação/teste (`embaralhar=False`): ordem fixa = ordem de `self.info`, para as previsões
      baterem linha a linha com os rótulos.
    - Dentro de cada lote os exames são lidos ordenados por (parte, linha); X e y saem nessa mesma ordem.
    """

    def __init__(self, info, lote=64, embaralhar=False, seed=SEED, **kw):
        super().__init__(**kw)
        self.info = info.sort_values(["parte", "linha"], ignore_index=True)
        self.lote, self.embaralhar, self.seed, self.epoca = lote, embaralhar, seed, 0
        self.Y = self.info[ROTULOS].to_numpy(np.float32)
        self._arquivos = {}
        self._ordem = np.arange(len(self.info))
        self._sortear()

    def _sortear(self):
        if self.embaralhar:
            self._ordem = np.random.default_rng(self.seed + self.epoca).permutation(len(self.info))

    def __len__(self):
        return int(np.ceil(len(self.info) / self.lote))

    def posicoes(self, i):
        """Posições (em self.info) dos exames do lote i, na ordem em que são lidos."""
        pos = self._ordem[i * self.lote:(i + 1) * self.lote]
        return np.sort(pos) if self.embaralhar else pos  # self.info já está ordenado por (parte, linha)

    def __getitem__(self, i):
        pos = self.posicoes(i)
        X = np.empty((len(pos), 4096, 12), np.float32)
        for k, (p, linha) in enumerate(self.info.loc[pos, ["parte", "linha"]].to_numpy()):
            X[k] = self._abrir(p)["tracings"][linha]
        return X, self.Y[pos]

    def _abrir(self, p):
        if p not in self._arquivos:
            self._arquivos[p] = h5py.File(DADOS / f"exams_part{p}.hdf5", "r")
        return self._arquivos[p]

    def on_epoch_end(self):
        self.epoca += 1
        self._sortear()
