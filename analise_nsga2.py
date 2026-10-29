#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análise consolidada dos resultados do NSGA-II (dieta PHDI x aceitabilidade cultural).

Entradas (geradas por NSGA2.py)
  --final     stats/pareto_final.csv     (obrigatório) front final de cada execução
  --evolucao  stats/pareto_evolucao.csv  (opcional)    front a cada geração (lido em blocos)
  --limites   limites.csv                (opcional)    colunas: grupo,min_kcal,max_kcal
                                         (limites [p5,p95] dos genes; habilita a auditoria
                                          de genes fora do intervalo)

Saídas (pasta --saida): tabelas/*.csv, graficos/*.png e relatorio.html (autocontido)

Métrica principal: médias de fo_phdi e fo_aceitabilidade do front em cada execução.
  score = (média_phdi / MAX_PHDI + média_ace / MAX_ACE) / 2
  A normalização usa os máximos teóricos (--max-phdi, --max-ace), e não min-max dos dados,
  para que o score seja comparável entre runs diferentes.
Métrica secundária: hipervolume 2D (ref. (0,0)) normalizado por MAX_PHDI x MAX_ACE.
O critério do ranking é escolhido por --criterio (score | hv); o outro é reportado
com a concordância (Spearman) entre os dois rankings.

Uso
  python analise_nsga2_v2.py --final stats/pareto_final.csv \
      --evolucao stats/pareto_evolucao.csv --saida analise_run3 --rotulo "Run 3"
"""
import argparse
import base64
import itertools
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- constantes
GRUPOS = [
    "nuts_and_peanuts", "legumes", "fruits", "vegetables", "whole_cereals",
    "eggs", "fish_and_seafood", "tubers_and_potatoes", "dairy", "vegetable_oils",
    "dgv", "rev", "red_meat", "chicken_and_substitutes", "animal_fats",
    "added_sugars", "refined_cereals",
]
GRUPOS_FIXOS = {"nuts_and_peanuts", "whole_cereals", "fish_and_seafood", "dgv", "refined_cereals"}

PARAMS = {
    "Numero de Individuos": "N",
    "Numero de Geracoes": "G",
    "Taxa de Cruzamento": "C",
    "Taxa de Mutacao": "M",
    "Crossover Index": "CI",
    "Distribution Index": "DI",
}
OBJ_PHDI, OBJ_ACE = "fo_phdi", "fo_aceitabilidade"
CHAVE = ["Conjunto", "Execucao"]
COLUNAS_OBRIGATORIAS = ["Conjunto", "Execucao", OBJ_PHDI, OBJ_ACE]
METRICAS = [("media_phdi", "Média fo_phdi"), ("media_ace", "Média fo_aceitabilidade"),
            ("score", "Score"), ("hv_norm", "Hipervolume norm.")]
COR = {"phdi": "#4C72B0", "ace": "#DD8452", "ok": "#55A868", "dest": "#C44E52"}

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
})


# ---------------------------------------------------------------- utilidades
def log(msg):
    print(msg, flush=True)


def ic95(s):
    s = pd.Series(s).dropna()
    n = len(s)
    if n < 2:
        return np.nan
    return stats.t.ppf(0.975, n - 1) * s.std(ddof=1) / np.sqrt(n)


def holm(p):
    p = np.nan_to_num(np.asarray(p, float), nan=1.0)
    n = len(p)
    ordem = np.argsort(p)
    aj = np.empty(n)
    corrente = 0.0
    for r, i in enumerate(ordem):
        corrente = max(corrente, (n - r) * p[i])
        aj[i] = min(corrente, 1.0)
    return aj


def mascara_nao_dominada(x, y):
    """Não dominados (maximizar x e y); duplicatas exatas viram um único ponto."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ordem = np.lexsort((-y, -x))
    melhor = -np.inf
    mask = np.zeros(len(x), bool)
    for i in ordem:
        if y[i] > melhor:
            mask[i] = True
            melhor = y[i]
    return mask


def hipervolume_2d(x, y, ref=(0.0, 0.0)):
    """Hipervolume 2D exato (maximização) em relação a `ref`."""
    pts = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    pts = pts[(pts[:, 0] > ref[0]) & (pts[:, 1] > ref[1])]
    if len(pts) == 0:
        return 0.0
    pts = pts[np.lexsort((-pts[:, 1], -pts[:, 0]))]
    hv, max_y = 0.0, ref[1]
    for px, py in pts:
        if py > max_y:
            hv += (px - ref[0]) * (py - max_y)
            max_y = py
    return float(hv)


def salvar_fig(fig, caminho):
    fig.tight_layout()
    fig.savefig(caminho, bbox_inches="tight")
    plt.close(fig)
    return caminho


def rotulo_conj(linha, fatores):
    partes = ", ".join(f"{PARAMS[f]}={linha[f]:g}" for f in fatores)
    return f"#{int(linha['Conjunto'])}" + (f" ({partes})" if partes else "")


# ---------------------------------------------------------------- carga e auditoria
def carregar_final(caminho, limites=None):
    df = pd.read_csv(caminho)
    faltando = [c for c in COLUNAS_OBRIGATORIAS if c not in df.columns]
    if faltando:
        sys.exit(f"ERRO: colunas ausentes em {caminho}: {faltando}")
    for c in df.columns:
        if c not in ("UF", "Faixa de Renda"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    def col(nome):
        return df[nome] if nome in df.columns else pd.Series(np.nan, index=df.index)

    # penalidades recalculadas a partir dos alvos (fórmula do NSGA2.py)
    df["erro_cal"] = (col("Calorias Totais") - col("Alvo Calorico")).abs() / col("Alvo Calorico")
    df["erro_custo"] = (col("Custo Total") - col("Custo Alvo")).abs() / col("Custo Alvo")
    df["pen_cal"] = np.maximum(1.0 - df["erro_cal"], 0.0)
    df["pen_custo"] = np.maximum(1.0 - df["erro_custo"], 0.0)
    df["pen_calc"] = df["pen_cal"] * df["pen_custo"]

    # penalidade implícita nos dados: fo / pontuação bruta (deve coincidir com pen_calc)
    pb_p, pb_a = col("Pontuacao PHDI"), col("Pontuacao Aceitabilidade")
    df["razao_phdi"] = df[OBJ_PHDI] / pb_p.where(pb_p > 0)
    df["razao_ace"] = df[OBJ_ACE] / pb_a.where(pb_a > 0)
    df["dif_pen"] = (df["razao_phdi"] - df["pen_calc"]).abs()

    # auditoria física
    cols_kcal = [f"{g}_kcal" for g in GRUPOS if f"{g}_kcal" in df.columns]
    if not cols_kcal:
        cols_kcal = [c for c in df.columns if c.endswith("_kcal")]
    df["tem_kcal_negativa"] = (df[cols_kcal] < 0).any(axis=1) if cols_kcal else False
    if {"vegetables_kcal", "dgv_kcal", "rev_kcal"} <= set(df.columns):
        df["viol_veg"] = (df["vegetables_kcal"] - df["dgv_kcal"] - df["rev_kcal"]) < -1e-9
    else:
        df["viol_veg"] = False

    df["n_fora_limite"] = 0
    if limites is not None:
        for _, r in limites.iterrows():
            c = f"{r['grupo']}_kcal"
            if c in df.columns:
                df["n_fora_limite"] += ((df[c] < r["min_kcal"] - 1e-6) | (df[c] > r["max_kcal"] + 1e-6)).astype(int)
    df["valida"] = ~(df["tem_kcal_negativa"] | df["viol_veg"] | (df["n_fora_limite"] > 0))
    return df


def carregar_limites(caminho):
    if not caminho:
        return None
    lim = pd.read_csv(caminho)
    falt = [c for c in ("grupo", "min_kcal", "max_kcal") if c not in lim.columns]
    if falt:
        sys.exit(f"ERRO: --limites precisa das colunas grupo,min_kcal,max_kcal (faltam {falt})")
    return lim


# ---------------------------------------------------------------- métricas
def metricas_por_execucao(df, max_phdi, max_ace, criterio):
    m = df.groupby(CHAVE).agg(
        media_phdi=(OBJ_PHDI, "mean"), media_ace=(OBJ_ACE, "mean"),
        max_phdi=(OBJ_PHDI, "max"), max_ace=(OBJ_ACE, "max"),
        n_pareto=(OBJ_PHDI, "size"),
        tempo=("Tempo de Execucao (s)", "first") if "Tempo de Execucao (s)" in df else (OBJ_PHDI, "size"),
        erro_cal=("erro_cal", "mean"), erro_cal_max=("erro_cal", "max"),
        erro_custo=("erro_custo", "mean"), erro_custo_max=("erro_custo", "max"),
        pen=("pen_calc", "mean"),
        pct_validas=("valida", "mean"),
        n_kcal_neg=("tem_kcal_negativa", "sum"),
        n_viol_veg=("viol_veg", "sum"),
        n_fora_limite=("n_fora_limite", lambda s: int((s > 0).sum())),
    ).reset_index()
    m["pct_validas"] *= 100.0

    hv, uni = {}, {}
    for k, g in df.groupby(CHAVE):
        x, y = g[OBJ_PHDI].values, g[OBJ_ACE].values
        hv[k] = hipervolume_2d(x, y)
        uni[k] = len(set(zip(np.round(x, 9), np.round(y, 9))))
    ks = list(zip(m["Conjunto"], m["Execucao"]))
    m["hipervolume"] = [hv[k] for k in ks]
    m["n_unicos"] = [uni[k] for k in ks]
    m["hv_norm"] = m["hipervolume"] / (max_phdi * max_ace)
    m["score"] = 0.5 * (m["media_phdi"] / max_phdi + m["media_ace"] / max_ace)
    m["crit"] = m["score"] if criterio == "score" else m["hv_norm"]
    return m


def tabela_configs(m, df):
    param_cols = [c for c in PARAMS if c in df.columns]
    params = df.groupby("Conjunto")[param_cols].first().reset_index()
    cfg = m.groupby("Conjunto").agg(
        n_execucoes=("Execucao", "nunique"),
        phdi_media=("media_phdi", "mean"), phdi_dp=("media_phdi", "std"),
        ace_media=("media_ace", "mean"), ace_dp=("media_ace", "std"),
        score_media=("score", "mean"), score_dp=("score", "std"), score_ic95=("score", ic95),
        hv_norm_media=("hv_norm", "mean"), hv_norm_dp=("hv_norm", "std"),
        crit_media=("crit", "mean"), crit_ic95=("crit", ic95),
        n_pareto_medio=("n_pareto", "mean"), n_unicos_medio=("n_unicos", "mean"),
        tempo_medio_s=("tempo", "mean"),
        erro_cal_max=("erro_cal_max", "max"), erro_custo_max=("erro_custo_max", "max"),
        pen_media=("pen", "mean"), pct_validas=("pct_validas", "mean"),
    ).reset_index()
    cfg = params.merge(cfg, on="Conjunto")
    cfg["ranking"] = cfg["crit_media"].rank(ascending=False, method="min").astype(int)
    cfg["rank_score"] = cfg["score_media"].rank(ascending=False, method="min").astype(int)
    cfg["rank_hv"] = cfg["hv_norm_media"].rank(ascending=False, method="min").astype(int)
    cfg["nao_dominada"] = mascara_nao_dominada(cfg["phdi_media"], cfg["ace_media"])
    return cfg.sort_values("ranking").reset_index(drop=True)


# ---------------------------------------------------------------- estatística
def testes_estatisticos(m, cfg):
    """Friedman (blocos = seeds pareadas) + Wilcoxon do melhor contra os demais (Holm)."""
    piv = m.pivot(index="Execucao", columns="Conjunto", values="crit").dropna(axis=0)
    info = {"n_blocos": len(piv), "k": piv.shape[1]}
    if piv.shape[1] < 3 or len(piv) < 3:
        return info, pd.DataFrame()
    chi2, p = stats.friedmanchisquare(*[piv[c].values for c in piv.columns])
    info.update(friedman_chi2=chi2, friedman_p=p, kendall_w=chi2 / (len(piv) * (piv.shape[1] - 1)))
    melhor = int(cfg.loc[cfg["ranking"] == 1, "Conjunto"].iloc[0])
    if melhor not in piv.columns:
        melhor = int(piv.mean().idxmax())
    linhas = []
    for c in piv.columns:
        if c == melhor:
            continue
        d = piv[melhor] - piv[c]
        try:
            pv = stats.wilcoxon(piv[melhor], piv[c]).pvalue
        except ValueError:
            pv = 1.0
        linhas.append({"Conjunto": c, "melhor": melhor, "dif_media_crit": d.mean(), "p_wilcoxon": pv})
    t = pd.DataFrame(linhas)
    t["p_holm"] = holm(t["p_wilcoxon"].values)
    t["melhor_sig_superior"] = t["p_holm"] < 0.05
    info.update(melhor=melhor, n_indistinguiveis=int((~t["melhor_sig_superior"]).sum()))
    return info, t.sort_values("p_holm").reset_index(drop=True)


def efeitos_fatores(m, df, fatores):
    if not fatores:
        return pd.DataFrame(), {}
    d = m.merge(df.groupby("Conjunto")[fatores].first().reset_index(), on="Conjunto")
    marg = []
    for f in fatores:
        for met, _ in METRICAS:
            for nivel, s in d.groupby(f)[met]:
                marg.append({"fator": PARAMS[f], "nivel": nivel, "metrica": met,
                             "media": s.mean(), "ic95": ic95(s), "n": len(s)})
    marg = pd.DataFrame(marg)

    anovas = {}
    ren = {f: f"f_{PARAMS[f]}" for f in fatores}
    dd = d.rename(columns=ren)
    termos = [f"C({v})" for v in ren.values()]
    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm
        for met, _ in METRICAS:
            formula = f"{met} ~ ({' + '.join(termos)})" + ("**2" if len(termos) > 1 else "")
            an = anova_lm(smf.ols(formula, data=dd).fit(), typ=2).reset_index().rename(columns={"index": "termo"})
            ss_res = an.loc[an["termo"] == "Residual", "sum_sq"].iloc[0]
            an["pct_variancia"] = 100 * an["sum_sq"] / an["sum_sq"].sum()
            an["eta2_parcial"] = an["sum_sq"] / (an["sum_sq"] + ss_res)
            an["termo"] = (an["termo"].str.replace("C(f_", "", regex=False)
                           .str.replace(")", "", regex=False).str.replace(":", " x "))
            anovas[met] = an
    except Exception as e:
        log(f"  [aviso] ANOVA completa indisponível ({e}); usando só efeitos principais.")
        for met, _ in METRICAS:
            ss_tot = ((dd[met] - dd[met].mean()) ** 2).sum()
            linhas = []
            for f in ren.values():
                grupos = [g[met].values for _, g in dd.groupby(f)]
                ss_b = sum(len(g) * (g.mean() - dd[met].mean()) ** 2 for g in grupos)
                p = stats.f_oneway(*grupos).pvalue if len(grupos) > 1 else np.nan
                linhas.append({"termo": f[2:], "sum_sq": ss_b, "PR(>F)": p, "pct_variancia": 100 * ss_b / ss_tot})
            anovas[met] = pd.DataFrame(linhas)
    return marg, anovas


# ---------------------------------------------------------------- evolução (em blocos)
def agregar_evolucao(caminho, max_phdi, max_ace, chunksize=1_000_000):
    """Lê pareto_evolucao.csv em blocos e agrega por (Conjunto, Execucao, Geracao):
    tamanho do front, médias/máximos das duas fo e penalidade média (se houver colunas)."""
    cab = pd.read_csv(caminho, nrows=0).columns
    base = ["Conjunto", "Execucao", "Geracao", OBJ_PHDI, OBJ_ACE]
    extra = ["Calorias Totais", "Alvo Calorico", "Custo Total", "Custo Alvo"]
    tem_pen = all(c in cab for c in extra)
    cols = base + (extra if tem_pen else [])
    chaves = ["Conjunto", "Execucao", "Geracao"]
    regras = {"s_phdi": "sum", "s_ace": "sum", "s_pen": "sum", "n": "sum", "max_phdi": "max", "max_ace": "max"}

    def combinar(partes):
        return pd.concat(partes).groupby(level=chaves).agg(regras)

    partes, acum = [], None
    for i, ch in enumerate(pd.read_csv(caminho, usecols=cols, chunksize=chunksize)):
        if tem_pen:
            pc = np.maximum(1 - (ch["Calorias Totais"] - ch["Alvo Calorico"]).abs() / ch["Alvo Calorico"], 0)
            pu = np.maximum(1 - (ch["Custo Total"] - ch["Custo Alvo"]).abs() / ch["Custo Alvo"], 0)
            ch["pen"] = pc * pu
        else:
            ch["pen"] = np.nan
        ch["n"] = 1
        g = ch.groupby(chaves).agg(s_phdi=(OBJ_PHDI, "sum"), s_ace=(OBJ_ACE, "sum"),
                                   s_pen=("pen", "sum"), n=("n", "sum"),
                                   max_phdi=(OBJ_PHDI, "max"), max_ace=(OBJ_ACE, "max"))
        partes.append(g)
        if len(partes) >= 20:
            acum = combinar(([acum] if acum is not None else []) + partes)
            partes = []
        if (i + 1) % 10 == 0:
            log(f"  ... {(i + 1) * chunksize:,} linhas lidas")
    acum = combinar(([acum] if acum is not None else []) + partes)
    acum["media_phdi"] = acum["s_phdi"] / acum["n"]
    acum["media_ace"] = acum["s_ace"] / acum["n"]
    acum["pen_media"] = acum["s_pen"] / acum["n"] if tem_pen else np.nan
    acum = acum.rename(columns={"n": "n_front"}).reset_index()
    acum["score"] = 0.5 * (acum["media_phdi"] / max_phdi + acum["media_ace"] / max_ace)
    return acum[chaves + ["n_front", "media_phdi", "media_ace", "max_phdi", "max_ace", "pen_media", "score"]]


def estabilizacao(ev, frac=0.95):
    """Primeira geração em que a execução atinge `frac` do ganho total do score."""
    linhas = []
    for (c, e), g in ev.sort_values("Geracao").groupby(["Conjunto", "Execucao"]):
        v, ger = g["score"].values, g["Geracao"].values
        idx = int(np.argmax(v >= v[0] + frac * (v[-1] - v[0]))) if v[-1] > v[0] else len(v) - 1
        linhas.append({"Conjunto": c, "Execucao": e, "ger_estab": ger[idx], "n_ger": len(v)})
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------- gráficos
def graf_ranking(cfg, fatores, pasta, nome_crit, top=15):
    t = cfg.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(t) + 1.5))
    y = np.arange(len(t))
    ax.barh(y, t["crit_media"], xerr=t["crit_ic95"].fillna(0), color=COR["phdi"], alpha=0.85, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels([rotulo_conj(r, fatores) for _, r in t.iterrows()], fontsize=8)
    ax.set_xlabel(f"{nome_crit} (média ± IC95% entre seeds)")
    ax.set_title(f"Ranking das {len(t)} melhores configurações")
    lo = (t["crit_media"] - t["crit_ic95"].fillna(0)).min()
    ax.set_xlim(max(0, lo * 0.98), None)
    return salvar_fig(fig, os.path.join(pasta, "ranking_configs.png"))


def graf_boxplot_top(m, cfg, pasta, nome_crit, top=10):
    ids = cfg.head(top)["Conjunto"].tolist()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot([m.loc[m["Conjunto"] == c, "crit"].values for c in ids], showmeans=True)
    ax.set_xticklabels([f"#{c}" for c in ids])
    ax.set_xlabel("Conjunto (ordenado por ranking)")
    ax.set_ylabel(f"{nome_crit} por execução")
    ax.set_title(f"Variabilidade entre seeds — top {len(ids)}")
    return salvar_fig(fig, os.path.join(pasta, "boxplot_top.png"))


def graf_dispersao_configs(cfg, pasta):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    nd, d = cfg[cfg["nao_dominada"]], cfg[~cfg["nao_dominada"]]
    ax.scatter(d["phdi_media"], d["ace_media"], c="#999999", s=28, label="Dominadas")
    ax.scatter(nd["phdi_media"], nd["ace_media"], c=COR["dest"], s=50, label="Não dominadas")
    for _, r in cfg.head(5).iterrows():
        ax.annotate(f"#{int(r['Conjunto'])}", (r["phdi_media"], r["ace_media"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xlabel("Média de fo_phdi (média entre seeds)")
    ax.set_ylabel("Média de fo_aceitabilidade (média entre seeds)")
    ax.set_title("Configurações no espaço das duas métricas")
    ax.legend()
    return salvar_fig(fig, os.path.join(pasta, "dispersao_configs.png"))


def graf_efeitos(marg, pasta):
    fatores = list(marg["fator"].unique())
    cores = [COR["phdi"], COR["ace"], COR["ok"], COR["dest"]]
    fig, axs = plt.subplots(len(fatores), 4, figsize=(15, 2.5 * len(fatores)), squeeze=False)
    for i, f in enumerate(fatores):
        for j, (met, nome) in enumerate(METRICAS):
            s = marg[(marg["fator"] == f) & (marg["metrica"] == met)].sort_values("nivel")
            ax = axs[i][j]
            x = np.arange(len(s))
            ax.errorbar(x, s["media"], yerr=s["ic95"].fillna(0), marker="o", capsize=3, color=cores[j])
            ax.set_xticks(x)
            ax.set_xticklabels([f"{v:g}" for v in s["nivel"]])
            ax.set_xlabel(f)
            if i == 0:
                ax.set_title(nome, fontsize=10)
    fig.suptitle("Efeito marginal de cada parâmetro (média ± IC95%)", fontweight="bold")
    return salvar_fig(fig, os.path.join(pasta, "efeitos_principais.png"))


def graf_interacoes(m, df, fatores, pasta):
    saidas = []
    if len(fatores) < 2:
        return saidas
    d = m.merge(df.groupby("Conjunto")[fatores].first().reset_index(), on="Conjunto")
    for a, b in itertools.combinations(fatores, 2):
        pv = d.pivot_table(index=a, columns=b, values="crit", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(4.8, 4))
        im = ax.imshow(pv.values, cmap="YlGnBu", aspect="auto", origin="lower")
        ax.set_xticks(range(pv.shape[1]))
        ax.set_xticklabels([f"{v:g}" for v in pv.columns])
        ax.set_yticks(range(pv.shape[0]))
        ax.set_yticklabels([f"{v:g}" for v in pv.index])
        ax.set_xlabel(PARAMS[b])
        ax.set_ylabel(PARAMS[a])
        ax.grid(False)
        for (i, j), v in np.ndenumerate(pv.values):
            ax.text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=7)
        ax.set_title(f"Interação {PARAMS[a]} x {PARAMS[b]}")
        fig.colorbar(im, ax=ax, fraction=0.046)
        saidas.append((f"Interação {PARAMS[a]} x {PARAMS[b]}",
                       salvar_fig(fig, os.path.join(pasta, f"interacao_{PARAMS[a]}_{PARAMS[b]}.png"))))
    return saidas


def graf_convergencia(ev, cfg, fatores, pasta, top=5):
    ids = cfg.head(top)["Conjunto"].tolist()
    pior = cfg.iloc[-1]["Conjunto"] if len(cfg) > top else None
    if pior is not None:
        ids.append(pior)
    media_geral = ev.groupby("Geracao")[["media_phdi", "media_ace", "score"]].mean()
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, col, tit in zip(axs, ("media_phdi", "media_ace", "score"),
                            ("Média fo_phdi do front", "Média fo_aceitabilidade do front", "Score (máx. teórico = 1)")):
        if len(cfg) > 1:
            ax.plot(media_geral.index, media_geral[col], color="#888888", lw=2.5, ls="--", label="Média de todas")
        for c in ids:
            sub = ev[ev["Conjunto"] == c].groupby("Geracao")[col].mean()
            linha = cfg.loc[cfg["Conjunto"] == c].iloc[0]
            ax.plot(sub.index, sub.values, lw=1.4,
                    label=(rotulo_conj(linha, fatores) + (" [pior]" if c == pior else "")) if col == "score" else None)
        ax.set_xlabel("Geração")
        ax.set_title(tit, fontsize=10)
    axs[2].legend(fontsize=7, loc="lower right")
    fig.suptitle("Convergência (média entre seeds)", fontweight="bold")
    return salvar_fig(fig, os.path.join(pasta, "convergencia.png"))


def graf_detalhe_melhor(ev, conj, nome, pasta):
    sub = ev[ev["Conjunto"] == conj].groupby("Geracao").agg(
        n_front=("n_front", "mean"), pen=("pen_media", "mean"),
        mp=("media_phdi", "mean"), xp=("max_phdi", "mean"), ma=("media_ace", "mean"), xa=("max_ace", "mean"))
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.3))
    axs[0].plot(sub.index, sub["mp"], c=COR["phdi"], lw=2, label="Média fo_phdi")
    axs[0].plot(sub.index, sub["xp"], c=COR["phdi"], ls="--", alpha=.7, label="Máx fo_phdi")
    axs[0].plot(sub.index, sub["ma"], c=COR["ok"], lw=2, label="Média fo_aceitabilidade")
    axs[0].plot(sub.index, sub["xa"], c=COR["ok"], ls="--", alpha=.7, label="Máx fo_aceitabilidade")
    axs[0].set_title(f"Front ao longo das gerações — {nome}", fontsize=10)
    axs[0].set_xlabel("Geração")
    axs[0].legend(fontsize=8, ncol=2)
    axs[1].plot(sub.index, sub["n_front"], c="#9467bd", lw=2, label="|F1| (tamanho do front)")
    axs[1].set_xlabel("Geração")
    axs[1].set_ylabel("|F1|", color="#9467bd")
    if sub["pen"].notna().any():
        tw = axs[1].twinx()
        tw.plot(sub.index, sub["pen"], c=COR["dest"], lw=1.6, ls=":", label="Penalidade média")
        tw.set_ylabel("Penalidade média (1 = sem desvio)", color=COR["dest"])
        tw.grid(False)
    axs[1].set_title("Tamanho do front e penalidade", fontsize=10)
    return salvar_fig(fig, os.path.join(pasta, "convergencia_detalhe_melhor.png"))


def graf_estab(est, cfg, pasta):
    t = est.groupby("Conjunto")["ger_estab"].mean().reset_index().merge(cfg[["Conjunto", "crit_media"]], on="Conjunto")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(t["ger_estab"], t["crit_media"], c=COR["phdi"], s=30)
    ax.set_xlabel("Geração média de estabilização (95% do ganho do score)")
    ax.set_ylabel("Critério final médio")
    ax.set_title("Velocidade de convergência x qualidade final")
    return salvar_fig(fig, os.path.join(pasta, "estabilizacao_vs_qualidade.png"))


# ---------------------------------------------------------------- dietas finais
def escolher_dietas(df, conjunto, tol):
    sub = df[df["Conjunto"] == conjunto].copy()
    sub["viavel"] = sub["valida"] & (sub["erro_cal"] <= tol) & (sub["erro_custo"] <= tol)
    base = sub[sub["viavel"]]
    aviso = None
    if base.empty:
        aviso = (f"Nenhuma dieta válida com erro de calorias e custo ≤ {tol:.0%}; "
                 "usando todas as dietas do conjunto.")
        base = sub
    front = base[mascara_nao_dominada(base[OBJ_PHDI].values, base[OBJ_ACE].values)]
    front = front.sort_values(OBJ_PHDI, ascending=False).reset_index(drop=True)

    reps = {}
    if len(front):
        reps["Máx. fo_phdi"] = 0
        reps["Máx. fo_aceitabilidade"] = len(front) - 1
        px = (front[OBJ_PHDI] - front[OBJ_PHDI].min()) / max(np.ptp(front[OBJ_PHDI]), 1e-12)
        py = (front[OBJ_ACE] - front[OBJ_ACE].min()) / max(np.ptp(front[OBJ_ACE]), 1e-12)
        reps["Equilibrada (knee: mais próxima do ponto ideal)"] = int(np.argmin(np.hypot(1 - px, 1 - py)))
        reps["Menor penalização (calorias/custo)"] = int(front["pen_calc"].idxmax())
    sel = front.loc[list(reps.values())].copy() if reps else front.iloc[0:0].copy()
    sel.insert(0, "perfil", list(reps.keys()))
    return sub, base, front, sel, aviso


def graf_pareto_final(sub, base, front, sel, pasta):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(sub[OBJ_PHDI], sub[OBJ_ACE], c="#cccccc", s=8, label="Todas as dietas do conjunto")
    ax.scatter(base[OBJ_PHDI], base[OBJ_ACE], c=COR["phdi"], s=10, alpha=0.5, label="Dietas viáveis")
    f = front.sort_values(OBJ_PHDI)
    ax.plot(f[OBJ_PHDI], f[OBJ_ACE], c=COR["dest"], lw=1.8, label="Front combinado")
    marc = ["^", "s", "*", "D"]
    for k, (_, r) in enumerate(sel.iterrows()):
        ax.scatter(r[OBJ_PHDI], r[OBJ_ACE], s=130, marker=marc[k % 4], edgecolor="k", zorder=5, label=r["perfil"])
    ax.set_xlabel("fo_phdi")
    ax.set_ylabel("fo_aceitabilidade")
    ax.set_title("Front combinado do conjunto escolhido")
    ax.legend(fontsize=7)
    return salvar_fig(fig, os.path.join(pasta, "pareto_conjunto_escolhido.png"))


def graf_composicao(sel, pasta):
    grupos = [g for g in GRUPOS if f"{g}_kcal" in sel.columns] or \
             [c[:-5] for c in sel.columns if c.endswith("_kcal")]
    perfis = sel[~sel["perfil"].str.startswith("Menor")]
    if perfis.empty or not grupos:
        return None
    fig, ax = plt.subplots(figsize=(11, 5))
    larg = 0.8 / len(perfis)
    x = np.arange(len(grupos))
    for k, (_, r) in enumerate(perfis.iterrows()):
        ax.bar(x + k * larg, [r[f"{g}_kcal"] for g in grupos], larg, label=r["perfil"])
    ax.set_xticks(x + larg * (len(perfis) - 1) / 2)
    ax.set_xticklabels([g.replace("_", " ").title() + (" (fixo)" if g in GRUPOS_FIXOS else "") for g in grupos],
                       rotation=55, ha="right", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("kcal")
    ax.set_title("Composição das dietas selecionadas (kcal por grupo)")
    ax.legend(fontsize=8)
    return salvar_fig(fig, os.path.join(pasta, "composicao_dietas.png"))


# ---------------------------------------------------------------- relatório HTML
CSS = """
body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;line-height:1.5;color:#1f2328;background:#fff}
h1{border-bottom:2px solid #4C72B0;padding-bottom:.3rem}h2{margin-top:2.2rem;color:#2b4b7a}
table.tbl{border-collapse:collapse;font-size:.82rem;margin:.6rem 0;display:block;overflow-x:auto}
table.tbl th,table.tbl td{border:1px solid #d0d7de;padding:3px 8px;text-align:right;white-space:nowrap}
table.tbl th{background:#f0f3f6}.nota{background:#eef4ff;border-left:4px solid #4C72B0;padding:.5rem .8rem;margin:.8rem 0}
.alerta{background:#fdecea;border-left:4px solid #c44e52;padding:.5rem .8rem;margin:.8rem 0}
.ok{background:#eaf6ee;border-left:4px solid #55A868;padding:.5rem .8rem;margin:.8rem 0}
img{max-width:100%;height:auto;margin:.5rem 0}.grid{display:flex;flex-wrap:wrap;gap:.5rem}.grid img{width:48%}
@media (prefers-color-scheme:dark){body{background:#0d1117;color:#e6edf3}h2{color:#8fb4ff}
table.tbl th{background:#161b22}table.tbl th,table.tbl td{border-color:#30363d}
.nota{background:#14233a}.alerta{background:#3a1d1d}.ok{background:#12301d}img{background:#fff}}
"""


def img_tag(caminho, legenda=""):
    with open(caminho, "rb") as f:
        b = base64.b64encode(f.read()).decode()
    return f'<figure><img src="data:image/png;base64,{b}" alt="{legenda}"><figcaption>{legenda}</figcaption></figure>'


def tabela_html(df, n=None):
    d = df.head(n) if n else df
    return d.to_html(index=False, classes="tbl", float_format=lambda v: f"{v:.4g}", na_rep="–", border=0)


# ---------------------------------------------------------------- principal
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", default="stats/pareto_final.csv")
    ap.add_argument("--evolucao", default=None)
    ap.add_argument("--limites", default=None, help="CSV grupo,min_kcal,max_kcal (opcional)")
    ap.add_argument("--saida", default="analise_nsga2")
    ap.add_argument("--rotulo", default="Execução")
    ap.add_argument("--criterio", choices=["score", "hv"], default="score",
                    help="critério do ranking (padrão: score = médias normalizadas)")
    ap.add_argument("--max-phdi", type=float, default=150.0, help="máximo teórico de fo_phdi")
    ap.add_argument("--max-ace", type=float, default=160.0, help="máximo teórico de fo_aceitabilidade")
    ap.add_argument("--tol", type=float, default=0.05, help="erro relativo máx. de calorias/custo p/ dieta viável")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--conjunto", type=int, default=None, help="forçar conjunto nas dietas finais")
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    a = ap.parse_args()

    pt, pg = os.path.join(a.saida, "tabelas"), os.path.join(a.saida, "graficos")
    os.makedirs(pt, exist_ok=True)
    os.makedirs(pg, exist_ok=True)
    nome_crit = "Score" if a.criterio == "score" else "Hipervolume normalizado"
    blocos = []

    # ---- 1. carga e verificações
    log("[1/6] Lendo pareto_final ...")
    limites = carregar_limites(a.limites)
    df = carregar_final(a.final, limites)
    fatores = [c for c in PARAMS if c in df.columns and df[c].nunique() > 1]
    log(f"  {len(df):,} linhas | {df['Conjunto'].nunique()} conjuntos | fatores variando: "
        f"{[PARAMS[f] for f in fatores] or 'nenhum'}")

    seeds = df.groupby("Conjunto")["Execucao"].apply(lambda s: tuple(sorted(s.unique())))
    seeds_iguais = seeds.nunique() == 1
    exec_por_conj = df.groupby("Conjunto")["Execucao"].nunique()
    dif_max = df["dif_pen"].max()
    n_neg, n_veg = int(df["tem_kcal_negativa"].sum()), int(df["viol_veg"].sum())
    n_lim = int((df["n_fora_limite"] > 0).sum())
    verif = [
        ("Indivíduos nos fronts finais", f"{len(df):,}"),
        ("Conjuntos / execuções por conjunto (mín–máx)",
         f"{df['Conjunto'].nunique()} / {exec_por_conj.min()}–{exec_por_conj.max()}"),
        ("Fatores que variam", ", ".join(PARAMS[f] for f in fatores) or "nenhum"),
        ("Alvo calórico / custo alvo (1ª linha)",
         f"{df['Alvo Calorico'].iloc[0]:.4f} kcal / {df['Custo Alvo'].iloc[0]:.4f}" if "Alvo Calorico" in df else "–"),
        ("Erro relativo de calorias: médio / máximo", f"{df['erro_cal'].mean():.3%} / {df['erro_cal'].max():.3%}"),
        ("Erro relativo de custo: médio / máximo", f"{df['erro_custo'].mean():.3%} / {df['erro_custo'].max():.3%}"),
        ("Penalidade (cal × custo) média / mínima", f"{df['pen_calc'].mean():.5f} / {df['pen_calc'].min():.5f}"),
        ("Dietas com kcal negativa", f"{n_neg:,}"),
        ("Dietas com DGV + REV > Vegetables", f"{n_veg:,}"),
        ("Dietas com algum gene fora de [min, max]", f"{n_lim:,}" if limites is not None else "não auditado (sem --limites)"),
        ("Dietas fisicamente válidas", f"{df['valida'].mean():.2%}"),
    ]
    blocos.append("<h2>1. Verificações dos dados</h2>" + tabela_html(pd.DataFrame(verif, columns=["Item", "Valor"])))
    if np.isfinite(dif_max):
        if dif_max > 1e-3:
            blocos.append(f"<div class='alerta'>A penalidade recalculada <b>não</b> reproduz fo/pontuação "
                          f"(diferença máx. {dif_max:.2e}). A fórmula do script pode estar desatualizada em relação ao "
                          "NSGA2.py; as colunas de erro/penalidade devem ser lidas com cautela.</div>")
        else:
            blocos.append(f"<div class='ok'>A penalidade recalculada a partir dos alvos reproduz fo/pontuação "
                          f"(diferença máx. {dif_max:.1e}).</div>")
    if not seeds_iguais:
        blocos.append("<div class='alerta'>Os conjuntos não têm as mesmas seeds (run incompleto?). Os testes pareados "
                      "usam apenas seeds presentes em todos os conjuntos e as médias marginais podem ficar desbalanceadas.</div>")
    if n_neg or n_veg or n_lim:
        blocos.append(f"<div class='alerta'>Violações físicas encontradas: {n_neg} (kcal negativa), {n_veg} (DGV+REV &gt; "
                      f"Vegetables), {n_lim} (gene fora de limites). Essas dietas são excluídas da escolha final.</div>")
    if limites is None:
        blocos.append("<div class='nota'>Sem <code>--limites</code>, genes fora de [p5, p95] (possível, pois SBX e "
                      "mutação não aplicam limites) não são detectados; &quot;0 kcal negativa&quot; não os exclui.</div>")

    # ---- 2. métricas e ranking
    log("[2/6] Métricas por execução e ranking ...")
    m = metricas_por_execucao(df, a.max_phdi, a.max_ace, a.criterio)
    cfg = tabela_configs(m, df)
    m.to_csv(os.path.join(pt, "metricas_por_execucao.csv"), index=False)
    cfg.to_csv(os.path.join(pt, "ranking_configuracoes.csv"), index=False)
    p_rank = graf_ranking(cfg, fatores, pg, nome_crit, a.top)
    p_box = graf_boxplot_top(m, cfg, pg, nome_crit, min(10, len(cfg)))
    p_disp = graf_dispersao_configs(cfg, pg)

    info, tw = testes_estatisticos(m, cfg)
    if len(tw):
        tw.to_csv(os.path.join(pt, "testes_vs_melhor.csv"), index=False)
    txt = ""
    if "friedman_p" in info:
        txt = (f"<p>Friedman ({info['k']} conjuntos × {info['n_blocos']} seeds pareadas): χ²={info['friedman_chi2']:.2f}, "
               f"p={info['friedman_p']:.3g}, W de Kendall={info['kendall_w']:.3f}.<br>Wilcoxon pareado do melhor "
               f"(#{info['melhor']}) contra cada outro, com Holm: <b>{info['n_indistinguiveis']}</b> de {info['k'] - 1} "
               f"conjuntos <b>não</b> são significativamente piores (α=0,05).</p>")
        p_min = 2.0 / 2 ** info["n_blocos"]
        if p_min >= 0.05:
            txt += (f"<div class='alerta'>Com {info['n_blocos']} seeds pareadas, o menor p possível do Wilcoxon bilateral é "
                    f"{p_min:.4f} (≥ 0,05): nenhum conjunto pode ser declarado significativamente pior. Isso reflete falta "
                    "de poder estatístico, não equivalência entre as configurações. Use ≥ 6 seeds (idealmente 30).</div>")
    conc = ""
    if len(cfg) >= 3:
        rho = stats.spearmanr(cfg["score_media"], cfg["hv_norm_media"])[0]
        conc = (f"<p>Concordância entre os rankings por score e por hipervolume (Spearman): <b>{rho:.3f}</b>. "
                "Valores baixos indicam que a escolha do critério muda a conclusão.</p>")
    cols_rank = ["ranking", "rank_score", "rank_hv", "Conjunto"] + [c for c in PARAMS if c in fatores] + [
        "phdi_media", "ace_media", "score_media", "hv_norm_media", "crit_ic95", "pct_validas",
        "n_pareto_medio", "tempo_medio_s", "nao_dominada"]
    blocos.append(
        "<h2>2. Ranking das configurações</h2>"
        f"<div class='nota'>Critério do ranking: <b>{nome_crit}</b>. Score = (média fo_phdi / {a.max_phdi:g} + "
        f"média fo_aceitabilidade / {a.max_ace:g}) / 2, com as médias calculadas sobre os indivíduos do front de cada "
        f"execução. Hipervolume com referência (0,0), normalizado por {a.max_phdi:g}×{a.max_ace:g}. A média do front "
        "não mede espalhamento: um front concentrado numa região boa pode pontuar mais que um front bem distribuído; "
        "o hipervolume captura os dois aspectos.</div>"
        + txt + conc + tabela_html(cfg[cols_rank], 20)
        + img_tag(p_rank, "Ranking (barras de erro: IC95% entre seeds)")
        + "<div class='grid'>" + img_tag(p_box, "Distribuição entre seeds") + img_tag(p_disp, "Médias por configuração") + "</div>")
    if len(tw):
        blocos.append("<h3>Testes pareados contra o melhor (10 menores p ajustados)</h3>" + tabela_html(tw, 10))

    # ---- 3. efeito dos parâmetros
    log("[3/6] Efeitos dos parâmetros e interações ...")
    marg, anovas = efeitos_fatores(m, df, fatores)
    if not marg.empty:
        marg.to_csv(os.path.join(pt, "medias_marginais.csv"), index=False)
        blocos.append("<h2>3. Efeito dos parâmetros e interações</h2>"
                      + img_tag(graf_efeitos(marg, pg), "Efeitos marginais"))
        for met, nome in METRICAS:
            if met in anovas:
                an = anovas[met]
                an.to_csv(os.path.join(pt, f"anova_{met}.csv"), index=False)
                blocos.append(f"<h3>ANOVA — {nome}</h3>" + tabela_html(an.sort_values("pct_variancia", ascending=False), 12))
        blocos.append("<div class='nota'>ANOVA tipo II com efeitos principais e interações de 2ª ordem, tendo como unidade "
                      "cada execução (seed × conjunto). Compare <i>pct_variancia</i> e <i>eta2_parcial</i> para a "
                      "importância relativa; com muitas observações, p-valores sinalizam até efeitos pequenos.</div>")
        inter = graf_interacoes(m, df, fatores, pg)
        if inter:
            blocos.append("<h3>Interações (critério médio por par de níveis)</h3><div class='grid'>"
                          + "".join(img_tag(p, t) for t, p in inter) + "</div>")
    else:
        blocos.append("<h2>3. Efeito dos parâmetros</h2><p>Nenhum parâmetro varia neste CSV (execução única ou "
                      "configuração única); seção omitida.</p>")

    # ---- 4. convergência
    log("[4/6] Convergência ...")
    if a.evolucao and os.path.exists(a.evolucao):
        ev = agregar_evolucao(a.evolucao, a.max_phdi, a.max_ace, a.chunksize)
        conv = ev.groupby(["Conjunto", "Geracao"]).agg(
            n_seeds=("Execucao", "nunique"), n_front=("n_front", "mean"), media_phdi=("media_phdi", "mean"),
            media_ace=("media_ace", "mean"), max_phdi=("max_phdi", "mean"), max_ace=("max_ace", "mean"),
            pen_media=("pen_media", "mean"), score=("score", "mean")).reset_index()
        conv.to_csv(os.path.join(pt, "convergencia_por_conjunto.csv"), index=False)
        est = estabilizacao(ev)
        est_cfg = est.groupby("Conjunto")["ger_estab"].agg(["mean", "std"]).reset_index()
        est_cfg.columns = ["Conjunto", "ger_estab_media", "ger_estab_dp"]
        est_cfg = cfg[["Conjunto", "ranking"] + [c for c in PARAMS if c in fatores]].merge(est_cfg, on="Conjunto")
        est_cfg.sort_values("ranking").to_csv(os.path.join(pt, "estabilizacao.csv"), index=False)
        melhor = int(cfg.iloc[0]["Conjunto"])
        blocos.append(
            "<h2>4. Convergência ao longo das gerações</h2>"
            + img_tag(graf_convergencia(ev, cfg, fatores, pg), "Convergência")
            + img_tag(graf_detalhe_melhor(ev, melhor, rotulo_conj(cfg.iloc[0], fatores), pg), "Melhor conjunto em detalhe")
            + "<div class='nota'>Estabilização = primeira geração em que a execução alcança 95% do ganho total do score "
              "(do início ao fim da própria execução). |F1| é o número de indivíduos não dominados registrados em cada "
              "geração; quando atinge N, o front ocupa toda a população.</div>")
        if len(cfg) > 2:
            blocos.append(img_tag(graf_estab(est, cfg, pg), "Velocidade x qualidade")
                          + "<h3>Estabilização por configuração</h3>" + tabela_html(est_cfg.sort_values("ranking"), 15))
    else:
        blocos.append("<h2>4. Convergência</h2><p>pareto_evolucao.csv não informado; seção omitida.</p>")

    # ---- 5. dietas finais
    log("[5/6] Dietas finais ...")
    conj = a.conjunto if a.conjunto is not None else int(cfg.iloc[0]["Conjunto"])
    sub, base, front, sel, aviso = escolher_dietas(df, conj, a.tol)
    front.to_csv(os.path.join(pt, f"front_combinado_conjunto{conj}.csv"), index=False)
    sel.to_csv(os.path.join(pt, f"dietas_selecionadas_conjunto{conj}.csv"), index=False)
    cols_sel = ["perfil", OBJ_PHDI, OBJ_ACE, "Pontuacao PHDI", "Pontuacao Aceitabilidade", "pen_calc",
                "Calorias Totais", "Custo Total", "erro_cal", "erro_custo", "Execucao", "Pareto ID"]
    blocos.append(
        f"<h2>5. Dietas finais — conjunto #{conj}</h2>"
        f"<p>Reúnem-se as dietas de todas as seeds do conjunto; mantêm-se as fisicamente válidas com erro de calorias e "
        f"custo ≤ {a.tol:.0%} e filtra-se o front não dominado ({len(front)} dietas).</p>"
        + (f"<div class='alerta'>{aviso}</div>" if aviso else "")
        + tabela_html(sel[[c for c in cols_sel if c in sel.columns]])
        + img_tag(graf_pareto_final(sub, base, front, sel, pg), "Front combinado")
        + (lambda p: img_tag(p, "Composição") if p else "")(graf_composicao(sel, pg))
        + f"<div class='nota'>Tabelas completas (kcal, % e pontuações por grupo): tabelas/dietas_selecionadas_conjunto{conj}.csv "
          f"e tabelas/front_combinado_conjunto{conj}.csv.</div>")

    # ---- 6. metodologia
    blocos.append(
        "<h2>6. Notas metodológicas</h2><ul>"
        "<li>Cada execução (seed) é a unidade amostral; as seeds são pareadas entre conjuntos (mesma seed = mesmo bloco).</li>"
        "<li>Os fronts analisados são os exportados pelo NSGA2.py (|F1| truncado em N quando F1 excede N).</li>"
        "<li>Máximos teóricos usados na normalização: "
        f"fo_phdi={a.max_phdi:g}, fo_aceitabilidade={a.max_ace:g} (ajustáveis por --max-phdi / --max-ace).</li>"
        "<li>Dietas &quot;válidas&quot;: sem kcal negativa, com DGV+REV ≤ Vegetables e, se --limites for informado, "
        "genes dentro de [min, max].</li></ul>")

    # ---- 7. relatório
    log("[6/6] Gerando relatório HTML ...")
    html = (f"<!DOCTYPE html><html lang='pt-BR'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Análise NSGA-II — {a.rotulo}</title><style>{CSS}</style></head><body>"
            f"<h1>Análise NSGA-II — {a.rotulo}</h1>" + "\n".join(blocos) + "</body></html>")
    with open(os.path.join(a.saida, "relatorio.html"), "w", encoding="utf-8") as f:
        f.write(html)
    log(f"Pronto. Abra: {os.path.join(a.saida, 'relatorio.html')}")


if __name__ == "__main__":
    main()