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
CODE_TEST = Path(__file__).parent / "code_test" / "extraido" / "data"  # CODE-TEST (Zenodo), fora do git
ROTULOS = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF"]
SEED = 42
PARTES = [0, 1, 2, 3]
PARTES_DEV = [0]  # [PROV] dados de desenvolvimento do A1: só a parte 0 (a confirmar com a Renata)


def tabela_com_grupo():
    """exams.csv inteiro + coluna `grupo`. Mesma fórmula do notebook (seção 5):
    cada paciente da tabela INTEIRA é sorteado uma vez, então adicionar partes não muda o grupo de ninguém."""
    tabela = pd.read_csv(DADOS / "exams.csv")
    pacientes = np.sort(tabela["patient_id"].unique())
    sorteio = np.random.default_rng(SEED).random(len(pacientes))
    grupo = pd.Series(np.select([sorteio < 0.7, sorteio < 0.8], ["treino", "validação"], "teste"), index=pacientes)
    tabela["grupo"] = tabela["patient_id"].map(grupo).values
    return tabela


def exames_todo_zero(partes):
    """exam_id dos exames com as 12 derivações zeradas, a partir da lista code15/exames_zerados_p0-3.npy (gerada só pelo sinal).
    Regra de exclusão provisória (a confirmar com a Renata): só estes saem; os parciais ficam. A lista NÃO é alterada."""
    z = np.load(DADOS / "exames_zerados_p0-3.npy")  # (exam_id, parte, linha, n_derivações_zeradas)
    return set(z[(z[:, 3] == 12) & np.isin(z[:, 1], partes), 0])


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


def n_deriv_zeradas(x):
    """Nº de derivações com o sinal todo zero. x: (T, 12) ou (n, T, 12) -> escalar ou (n,). Só usa o sinal, nunca rótulos.
    Usa .max (não soma): a soma em float16 estoura (seção 9)."""
    return (np.abs(x).max(axis=-2) == 0).sum(axis=-1)


def todo_zero(x):
    """True nos exames com as 12 derivações zeradas (única regra de exclusão decidida em 08/11; os parciais ficam). Mesma definição de `listar_zerados`."""
    return n_deriv_zeradas(x) == 12


def listar_zerados(partes=PARTES, passo=500):
    """Passada única pelos sinais. Devolve array int32 (exam_id, parte, linha, n_derivações_zeradas),
    só dos exames com pelo menos 1 derivação zerada. NÃO descarta nada."""
    achados = []
    for p in partes:
        with h5py.File(DADOS / f"exams_part{p}.hdf5", "r") as f:
            ds, ids = f["tracings"], f["exam_id"][:]
            for ini in range(0, len(ids), passo):
                x = ds[ini:ini + passo]                                  # float32: o max evita o estouro da soma em float16
                n_zer = n_deriv_zeradas(x)                               # (exames,) derivações com o sinal todo zero
                for k in np.where((n_zer > 0) & (ids[ini:ini + passo] != 0))[0]:
                    achados.append((ids[ini + k], p, ini + k, n_zer[k]))
    return np.array(achados, dtype=np.int32).reshape(-1, 4)


def normalizar_exame(x):
    """A2 (provisório, a confirmar com a Renata): normalização POR EXAME e POR DERIVAÇÃO.
    Média e desvio são calculados só na região com sinal (fora do preenchimento de zeros das pontas), sem usar rótulos.
    O preenchimento continua zero. Derivação totalmente zerada (ou constante) fica zero."""
    ativo = np.abs(x).max(axis=1) > 0
    if not ativo.any():
        return x
    media, desvio = x[ativo].mean(axis=0), x[ativo].std(axis=0)
    y = np.zeros_like(x)
    y[ativo] = (x[ativo] - media) / np.where(desvio > 1e-6, desvio, np.inf)
    return y


class GeradorECG(Sequence):
    """Lotes de `lote` exames lidos direto do HDF5 (float32), sem carregar parte nenhuma inteira.

    - Treino: embaralha a cada época com seed fixa (SEED + nº da época), então dá para reproduzir.
    - Validação/teste (`embaralhar=False`): ordem fixa = ordem de `self.info`, para as previsões
      baterem linha a linha com os rótulos.
    - Dentro de cada lote os exames são lidos ordenados por (parte, linha); X e y saem nessa mesma ordem.
    """

    def __init__(self, info, lote=64, embaralhar=False, seed=SEED, normalizar=False, **kw):
        super().__init__(**kw)
        self.normalizar = normalizar  # False = nenhuma normalização (A1); True = por exame e derivação (A2 provisório)
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
            if self.normalizar:
                X[k] = normalizar_exame(X[k])
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

    def __init__(self, info=None, lote=64, normalizar=False, **kw):
        super().__init__(**kw)
        self.info = ptbxl_indice() if info is None else info
        self.lote = lote
        self.normalizar = normalizar  # True = normalizar_exame (A2) depois do pré-processamento; a região ativa exclui os 48 zeros de cada lado

    def __len__(self):
        return int(np.ceil(len(self.info) / self.lote))

    def __getitem__(self, i):
        caminhos = self.info["filename_hr"].iloc[i * self.lote:(i + 1) * self.lote]
        X = np.stack([preprocessar_ptbxl(wfdb.rdrecord(str(PTBXL / c)).p_signal) for c in caminhos])
        return np.stack([normalizar_exame(x) for x in X]) if self.normalizar else X


# ---------------------------------------------------------------------------------------------
# CODE-TEST (teste externo secundário, 827 exames, SEM patient_id). gold_standard.csv só como rótulo.
# ---------------------------------------------------------------------------------------------
# No CODE-TEST a ordem das colunas é DI, DII, DIII, AVL, AVF, AVR; [0,1,2,5,3,4,...] leva a DI, DII, DIII, AVR, AVL, AVF (medido em 07/11).
ORDEM_CODETEST = [0, 1, 2, 5, 3, 4, 6, 7, 8, 9, 10, 11]


def codetest_rotulos():
    """(827, 6) float32 na ordem de ROTULOS, lido de annotations/gold_standard.csv (colunas 1dAVb, RBBB, LBBB, SB, AF, ST)."""
    return pd.read_csv(CODE_TEST / "annotations" / "gold_standard.csv")[ROTULOS].to_numpy(np.float32)


class GeradorCodeTest(Sequence):
    """Lotes do CODE-TEST: (lote, 4096, 12) float32 (o arquivo é float64), derivações reordenadas, ordem fixa do arquivo.
    Devolve só X. Rótulos em `self.Y` (gold_standard). Sem patient_id: `self.info` só tem o número da linha."""

    def __init__(self, lote=64, normalizar=True, linhas=None, **kw):
        super().__init__(**kw)
        self.lote, self.normalizar = lote, normalizar
        with h5py.File(CODE_TEST / "ecg_tracings.hdf5", "r") as f:
            n_arquivo = f["tracings"].shape[0]
        Y = codetest_rotulos()
        assert len(Y) == n_arquivo, (len(Y), n_arquivo)  # rótulos alinhados linha a linha com os sinais
        self.linhas = np.arange(n_arquivo) if linhas is None else np.sort(np.asarray(linhas))  # `linhas`: subconjunto (ex.: sem exames todo-zero)
        self.n, self.Y = len(self.linhas), Y[self.linhas]
        self.info = pd.DataFrame({"linha": self.linhas})

    def __len__(self):
        return int(np.ceil(self.n / self.lote))

    def __getitem__(self, i):
        with h5py.File(CODE_TEST / "ecg_tracings.hdf5", "r") as f:
            x = f["tracings"][self.linhas[i * self.lote:(i + 1) * self.lote]]  # índices crescentes (h5py)
        x = x[:, :, ORDEM_CODETEST].astype(np.float32)
        return np.stack([normalizar_exame(e) for e in x]) if self.normalizar else x
