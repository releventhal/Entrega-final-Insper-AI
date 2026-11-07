"""Índice dos exames, lista de exames zerados e gerador em lotes (T9).

Nada aqui treina modelo. Os sinais só são lidos pelo gerador e por `listar_zerados`.
"""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import wfdb
from keras.utils import Sequence
from scipy.signal import resample_poly

DADOS = Path(__file__).parent / "code15"
PTBXL = Path(__file__).parent / "ptbxl"  # PTB-XL 1.0.3 (PhysioNet), fora do git
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


# ---------------------------------------------------------------------------------------------
# PTB-XL (teste externo). Só sinal e metadados: nenhum rótulo é usado aqui.
# ---------------------------------------------------------------------------------------------
# Ordem das colunas do PTB-XL medida com Einthoven (07/11, 1.000 registros): I, II, III, AVR, AVL, AVF, V1-V6,
# a mesma da CODE-15%. Então a permutação é a identidade (o CODE-TEST é que precisa reordenar).
ORDEM_PTBXL = list(range(12))


def ptbxl_indice():
    """Uma linha por registro do PTB-XL (ordem de ecg_id): patient_id, idade, sexo, fold e caminho do registro de 500 Hz."""
    db = pd.read_csv(PTBXL / "ptbxl_database.csv", index_col="ecg_id").sort_index()
    return db[["patient_id", "age", "sex", "strat_fold", "scp_codes", "filename_hr"]]


def preprocessar_ptbxl(x500):
    """(5000, 12) float64 em 500 Hz -> (4096, 12) float32 em 400 Hz, no formato da CODE-15%.
    500 -> 400 Hz = resample_poly(up=4, down=5) -> 4000 amostras (10 s); 48 zeros de cada lado -> 4096.
    Sem nenhuma normalização de amplitude (decisão da dupla, ainda em aberto)."""
    x = resample_poly(x500[:, ORDEM_PTBXL], up=4, down=5, axis=0)
    assert x.shape == (4000, 12), x.shape
    return np.pad(x, ((48, 48), (0, 0))).astype(np.float32)


class GeradorPTBXL(Sequence):
    """Lotes de `lote` registros do PTB-XL já no formato da CODE-15% (lote, 4096, 12) float32, lidos do disco.
    Ordem fixa = ordem de `self.info` (a de `ptbxl_indice()` ou do subconjunto passado). Devolve só X (sem rótulos)."""

    def __init__(self, info=None, lote=64, **kw):
        super().__init__(**kw)
        self.info = ptbxl_indice() if info is None else info
        self.lote = lote

    def __len__(self):
        return int(np.ceil(len(self.info) / self.lote))

    def __getitem__(self, i):
        caminhos = self.info["filename_hr"].iloc[i * self.lote:(i + 1) * self.lote]
        return np.stack([preprocessar_ptbxl(wfdb.rdrecord(str(PTBXL / c)).p_signal) for c in caminhos])
