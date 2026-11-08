"""Bloco 3 (08/11): média e desvio do A3 entre seeds, na VALIDAÇÃO completa (8.158 exames). Só lê experimentos.csv.
Provisório, a confirmar com a Renata. Uso: .venv\\Scripts\\python seeds_A3.py  ->  resultados/seeds_A3.csv
"""
import pandas as pd

from a0_baselines import PASTA_RESULTADOS

BASE = "A3 provisório (partes 0 a 3; normalização por exame e derivação)"


def main():
    exp = pd.read_csv(PASTA_RESULTADOS / "experimentos.csv")
    nomes = {42: BASE} | {s: f"{BASE} | seed {s}" for s in range(43, 50)}
    d = pd.concat([exp[exp.experimento == n].assign(seed=s) for s, n in nomes.items()])
    seeds = sorted(d.seed.unique())
    linhas = []
    for (r, m), v in d.melt(["seed", "rotulo"], ["auroc", "auprc", "f1"], var_name="metrica").groupby(["rotulo", "metrica"]):
        v = v.sort_values("seed")
        linhas.append(dict(rotulo=r, metrica=m, n_seeds=len(v), media=v.value.mean(), desvio=v.value.std(ddof=1),
                           minimo=v.value.min(), maximo=v.value.max(), valores=[round(x, 4) for x in v.value]))
    t = pd.DataFrame(linhas)
    t.to_csv(PASTA_RESULTADOS / "seeds_A3.csv", index=False, float_format="%.4f")
    print(f"seeds: {seeds}  (desvio padrão amostral, ddof=1)")
    for _, l in t.iterrows():
        print(f"  {l.rotulo:6s} {l.metrica:5s}: média {l.media:.4f}  desvio {l.desvio:.4f}  ({l.minimo:.4f} a {l.maximo:.4f})  {l.valores}")
    print("\népocas / melhor época / minutos por seed:")
    print(d[d.rotulo == "MACRO"][["seed", "epocas", "melhor_epoca", "minutos_treino"]].to_string(index=False))


if __name__ == "__main__":
    main()
