"""Robustez, tarefa 1: ensemble das 3 seeds do A3 (média das probabilidades) na VALIDAÇÃO. Só lê previsões salvas; não carrega modelo.
Limiar do ensemble por alteração = F1 máximo na validação (a0_baselines.melhor_limiar_f1), como no A3. Isso é OTIMISTA
(limiar e métrica na mesma validação), igual para as seeds sozinhas. O A3 oficial (seed 42) usa os limiares do JSON.
Saída nova: resultados/robustez_01_ensemble.csv  (e _diferencas.csv: ensemble - seed 42, bootstrap pareado por paciente).
"""
import numpy as np
import pandas as pd

from a0_baselines import melhor_limiar_f1, metricas
from ecg_dados import ROTULOS
from robustez_comum import RES, carregar_val, ic, limiares_json, macro_auc_ap, reamostras, salvar_novo

N_BOOT = 1000


def f1_vet(Y, S, lim, idx=None):
    if idx is not None:
        Y, S = Y[idx], S[idx]
    P = S >= lim
    tp, fp, fn = (P & (Y == 1)).sum(0), (P & (Y == 0)).sum(0), (~P & (Y == 1)).sum(0)
    return np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)


def main():
    S42, Y, age, male, pid = carregar_val()
    S = {"A3 oficial (seed 42, limiares do JSON)": S42,
         "seed 43": np.load(RES / "A3_seed43_previsoes_validacao.npy").astype(np.float64),
         "seed 44": np.load(RES / "A3_seed44_previsoes_validacao.npy").astype(np.float64)}
    assert all(v.shape == S42.shape == Y.shape for v in S.values()), "formas diferentes"
    S["ensemble 3 seeds (média das probabilidades)"] = np.mean(list(S.values()), axis=0)
    lim_json = np.array([limiares_json()[r] for r in ROTULOS])
    lims = {n: lim_json if n.startswith("A3 oficial") else np.array([melhor_limiar_f1(Y[:, k], s[:, k]) for k in range(6)]) for n, s in S.items()}
    # checagem de alinhamento: AUROC macro recalculado aqui tem que bater com experimentos.csv/seeds_A3.csv (0,9892 / 0,9899 / 0,9899)
    for n, s in S.items():
        print(f"{n}: AUROC macro {macro_auc_ap(Y, s)[0].mean():.4f}")

    boots = reamostras(pid, N_BOOT)
    linhas, boot_res = [], {}
    for n, s in S.items():
        au, ap = macro_auc_ap(Y, s)
        f1 = f1_vet(Y, s, lims[n])
        ba = np.array([macro_auc_ap(Y, s, i) for i in boots])  # (n_boot, 2, 6)
        bf = np.array([f1_vet(Y, s, lims[n], i) for i in boots])
        boot_res[n] = (ba[:, 0].mean(1), ba[:, 1].mean(1), bf.mean(1))
        for k, r in enumerate(ROTULOS):
            m = metricas(Y[:, k], s[:, k], lims[n][k])
            linhas.append(dict(modelo=n, rotulo=r, positivos=int(Y[:, k].sum()), limiar=lims[n][k], auroc=au[k], auprc=ap[k], f1=f1[k],
                               sensibilidade=m["sensibilidade"], precisao=m["precisao"], especificidade=m["especificidade"],
                               auroc_ic_inf=ic(ba[:, 0, k])[0], auroc_ic_sup=ic(ba[:, 0, k])[1], auprc_ic_inf=ic(ba[:, 1, k])[0],
                               auprc_ic_sup=ic(ba[:, 1, k])[1], f1_ic_inf=ic(bf[:, k])[0], f1_ic_sup=ic(bf[:, k])[1]))
        b = boot_res[n]
        linhas.append(dict(modelo=n, rotulo="MACRO", positivos=np.nan, limiar=np.nan, auroc=au.mean(), auprc=ap.mean(), f1=f1.mean(),
                           auroc_ic_inf=ic(b[0])[0], auroc_ic_sup=ic(b[0])[1], auprc_ic_inf=ic(b[1])[0], auprc_ic_sup=ic(b[1])[1],
                           f1_ic_inf=ic(b[2])[0], f1_ic_sup=ic(b[2])[1]))
    t = pd.DataFrame(linhas)
    salvar_novo(t, RES / "robustez_01_ensemble.csv")
    print(t[t.rotulo == "MACRO"].round(4).to_string(index=False))
    print(t[t.modelo.str.startswith("ensemble")].round(4)[["rotulo", "auroc", "auprc", "f1"]].to_string(index=False))

    # diferenças pareadas (mesmas reamostras de pacientes): ensemble - cada seed
    ens = "ensemble 3 seeds (média das probabilidades)"
    d = []
    for n in list(S)[:3]:
        for j, met in enumerate(("auroc", "auprc", "f1")):
            v = boot_res[ens][j] - boot_res[n][j]
            obs = t[(t.modelo == ens) & (t.rotulo == "MACRO")][met].item() - t[(t.modelo == n) & (t.rotulo == "MACRO")][met].item()
            d.append(dict(comparacao=f"ensemble - {n}", metrica_macro=met, diferenca=obs, ic_inf=ic(v)[0], ic_sup=ic(v)[1]))
    d = pd.DataFrame(d)
    salvar_novo(d, RES / "robustez_01_ensemble_diferencas.csv")
    print(d.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
