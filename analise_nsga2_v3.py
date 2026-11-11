#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análise consolidada dos resultados do NSGA-II (dieta PHDI x aceitabilidade cultural).

Entradas (geradas por NSGA2.py)
  --final     stats/pareto_final.csv     (obrigatório) front final de cada execução
  --evolucao  evolucao_agregada.csv      (opcional)    resumo do front por geração (formato novo, pequeno)
              ou pareto_evolucao.csv     (formato antigo, enorme: lido em blocos)
  --snapshots pareto_snapshots.csv       (opcional)    fronts em gerações selecionadas (auto-detectado ao lado do --final)
  --filtrar   "N=200,300;G=400"          (opcional)    analisa só uma sub-grade do fatorial
  Conjuntos com menos execuções que o máximo são EXCLUÍDOS automaticamente (use --manter-incompletos para evitar).
  --limites   limites.csv                (opcional)    colunas: grupo,min_kcal,max_kcal
                                         (limites [p5,p95] dos genes; habilita a auditoria
                                          de genes fora do intervalo)
  --limpar                               (opcional)    apaga a pasta --saida antes (pede 'sim'; recusa se ela contiver entradas)
  --banco     banco_INA_estratificado_ag.csv  (opcional)  alternativa a --limites: monta os limites direto do banco
                                         (linha da média nacional: p5/p95 de cada grupo; grupos de valor fixo
                                          = % fixo x kcal_consumo_phdi). Grava tabelas/limites_usados.csv

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
import math
import os
import shutil
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


# Grupo (nome usado nas colunas {grupo}_kcal do pareto_final) -> sufixo das colunas p5_/p95_ no banco INA.
# Espelha LIMITES_ACEITABILIDADE do código; valor fixo (% do alvo calórico) nos grupos de dieta fixa.
GRUPOS_BANCO = [
    ("nuts_and_peanuts", "nozes_amendoim", 11.6),
    ("legumes", "leguminosas", None),
    ("fruits", "frutas", None),
    ("vegetables", "veg_total", None),
    ("whole_cereals", "cereais_integrais", 0.8),
    ("eggs", "ovos", None),
    ("fish_and_seafood", "peixes", 1.27),
    ("tubers_and_potatoes", "batatas_tuber", None),
    ("dairy", "laticinios", None),
    ("vegetable_oils", "oleos_insat", None),
    ("dgv", "veg_verde", 0.11),
    ("rev", "veg_vermelho", None),
    ("red_meat", "carne_vermelha", None),
    ("chicken_and_substitutes", "aves_frango", None),
    ("animal_fats", "gord_animal", None),
    ("added_sugars", "acucares", None),
    ("refined_cereals", "cereais_refinados", 30.0),
]


def carregar_limites_banco(caminho):
    """Monta (grupo, min_kcal, max_kcal) a partir da linha nacional do banco INA (uf e faixa_renda vazias)."""
    banco = pd.read_csv(caminho)
    nac = banco[banco["uf"].isna() & banco["faixa_renda_per_capita"].isna()]
    if len(nac) != 1:
        sys.exit(f"ERRO: esperava 1 linha nacional (uf e faixa_renda_per_capita vazias) em {caminho}; achei {len(nac)}.")
    row = nac.iloc[0]
    alvo = float(row["kcal_consumo_phdi"])
    linhas = []
    for grupo, suf, fixo in GRUPOS_BANCO:
        if fixo is not None:
            lo = hi = fixo / 100.0 * alvo
        else:
            lo, hi = float(row[f"p5_{suf}_kcal"]), float(row[f"p95_{suf}_kcal"])
        linhas.append({"grupo": grupo, "min_kcal": lo, "max_kcal": hi, "fixo": fixo is not None})
    return pd.DataFrame(linhas)


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
        info["motivo"] = (f"só {len(piv)} seed(s) presente(s) em todos os {m['Conjunto'].nunique()} conjuntos "
                          "(são necessárias ≥ 3 seeds comuns a todos e ≥ 3 conjuntos)")
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
        return pd.DataFrame(), {}, True
    d = m.merge(df.groupby("Conjunto")[fatores].first().reset_index(), on="Conjunto")
    marg = []
    for f in fatores:
        for met, _ in METRICAS:
            for nivel, s in d.groupby(f)[met]:
                marg.append({"fator": PARAMS[f], "nivel": nivel, "metrica": met,
                             "media": s.mean(), "ic95": ic95(s), "n": len(s)})
    marg = pd.DataFrame(marg)

    anovas = {}
    completo = True
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
        completo = False
        for met, _ in METRICAS:
            ss_tot = ((dd[met] - dd[met].mean()) ** 2).sum()
            linhas = []
            for f in ren.values():
                grupos = [g[met].values for _, g in dd.groupby(f)]
                ss_b = sum(len(g) * (g.mean() - dd[met].mean()) ** 2 for g in grupos)
                p = stats.f_oneway(*grupos).pvalue if len(grupos) > 1 else np.nan
                linhas.append({"termo": f[2:], "sum_sq": ss_b, "PR(>F)": p, "pct_variancia": 100 * ss_b / ss_tot})
            anovas[met] = pd.DataFrame(linhas)
    return marg, anovas, completo


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


def estabilizacao(ev, frac=0.95, col="score"):
    """Primeira geração em que a execução atinge `frac` do ganho total do score."""
    linhas = []
    for (c, e), g in ev.sort_values("Geracao").groupby(["Conjunto", "Execucao"]):
        v, ger = g[col].values, g["Geracao"].values
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
    paineis = [("media_phdi", "Média fo_phdi do front"), ("media_ace", "Média fo_aceitabilidade do front"),
               ("score", "Score (máx. teórico = 1)")]
    if "hv_norm" in ev.columns and ev["hv_norm"].notna().any():
        paineis.append(("hv_norm", "Hipervolume normalizado"))
    cols = [c for c, _ in paineis]
    media_geral = ev.groupby("Geracao")[cols].mean()
    fig, axs = plt.subplots(1, len(paineis), figsize=(5 * len(paineis), 4.3))
    for ax, (col, tit) in zip(axs, paineis):
        if len(cfg) > 1:
            ax.plot(media_geral.index, media_geral[col], color="#888888", lw=2.5, ls="--", label="Média de todas")
        for c in ids:
            sub_ = ev[ev["Conjunto"] == c].groupby("Geracao")[col].mean()
            linha = cfg.loc[cfg["Conjunto"] == c].iloc[0]
            ax.plot(sub_.index, sub_.values, lw=1.4,
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


# ---------------------------------------------------------------- filtro, evolução (2 formatos), snapshots
def aplicar_filtro(df, spec):
    """spec: 'N=200,300;G=400' -> mantém só conjuntos cujos parâmetros estão nesses níveis."""
    if not spec:
        return df
    inv = {v: k for k, v in PARAMS.items()}
    for parte in spec.split(";"):
        if not parte.strip():
            continue
        nome, _, niveis = parte.partition("=")
        nome = nome.strip()
        if nome not in inv or inv[nome] not in df.columns:
            sys.exit(f"ERRO: --filtrar: parâmetro '{nome}' desconhecido (use {list(inv)}).")
        vals = np.array([float(v) for v in niveis.split(",")])
        col = df[inv[nome]].values.astype(float)
        df = df[np.isclose(col[:, None], vals[None, :]).any(axis=1)]
    if df.empty:
        sys.exit("ERRO: o filtro --filtrar não deixou nenhum dado.")
    return df.copy()


def carregar_evolucao(caminho, max_phdi, max_ace, chunksize):
    """Aceita o formato novo (evolucao_agregada.csv) ou o antigo (pareto_evolucao.csv, em blocos)."""
    cab = pd.read_csv(caminho, nrows=0).columns
    if {"n_front", "hv", "tam_pop"} <= set(cab):
        log("  formato novo (agregado) detectado")
        ev = pd.read_csv(caminho)
        ev["score"] = 0.5 * (ev["media_phdi"] / max_phdi + ev["media_ace"] / max_ace)
        ev["hv_norm"] = ev["hv"] / (max_phdi * max_ace)
        return ev, "agregado"
    log("  formato antigo detectado (lendo em blocos; pode demorar)")
    ev = agregar_evolucao(caminho, max_phdi, max_ace, chunksize)
    ev["hv_norm"] = np.nan
    return ev, "antigo"


def graf_snapshots(snap, conj, pasta):
    s_ = snap[snap["Conjunto"] == conj]
    if s_.empty:
        return None
    gers = sorted(s_["Geracao"].unique())
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for k, g in enumerate(gers):
        p = s_[s_["Geracao"] == g]
        f = p[mascara_nao_dominada(p[OBJ_PHDI].values, p[OBJ_ACE].values)].sort_values(OBJ_PHDI)
        ax.plot(f[OBJ_PHDI], f[OBJ_ACE], marker=".", ms=3, lw=1.3, color=cmap(k / max(len(gers) - 1, 1)),
                label=f"geração {int(g)}")
    ax.set_xlabel("fo_phdi")
    ax.set_ylabel("fo_aceitabilidade")
    ax.set_title(f"Evolução do front (melhor front combinado das seeds) — conjunto #{conj}")
    ax.legend(fontsize=8)
    return salvar_fig(fig, os.path.join(pasta, "evolucao_do_front.png"))


# ---------------------------------------------------------------- principal
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", default="stats/pareto_final.csv")
    ap.add_argument("--evolucao", default=None)
    ap.add_argument("--snapshots", default=None, help="pareto_snapshots.csv (padrão: ao lado do --final, se existir)")
    ap.add_argument("--execucoes", default=None, help="execucoes.csv (padrão: ao lado do --final, se existir)")
    ap.add_argument("--limites", default=None, help="CSV grupo,min_kcal,max_kcal (opcional)")
    ap.add_argument("--banco", default=None,
                    help="banco_INA_estratificado_ag.csv: monta os limites a partir da linha nacional (alternativa a --limites)")
    ap.add_argument("--filtrar", default=None, help="sub-grade, ex.: \"N=200,300;G=400,500\"")
    ap.add_argument("--manter-incompletos", action="store_true",
                    help="não excluir conjuntos com menos execuções que o máximo (não recomendado)")
    ap.add_argument("--saida", default="analise_nsga2")
    ap.add_argument("--limpar", action="store_true",
                    help="APAGA a pasta --saida antes de analisar (pede confirmação; recusa se ela contiver os arquivos de entrada)")
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

    if a.limpar and os.path.isdir(a.saida):
        saida_abs = os.path.realpath(a.saida)
        protegidos = {os.path.realpath(os.getcwd()), os.path.realpath(os.path.expanduser("~")), os.path.realpath(os.sep)}
        entradas = [x for x in (a.final, a.evolucao, a.snapshots, a.execucoes, a.limites, a.banco) if x]
        dentro = [x for x in entradas
                  if os.path.realpath(x) == saida_abs or os.path.realpath(x).startswith(saida_abs + os.sep)]
        if saida_abs in protegidos or any(pr.startswith(saida_abs + os.sep) for pr in protegidos):
            sys.exit(f"ERRO: --limpar recusado: {saida_abs} é (ou contém) a pasta atual, o home ou a raiz.")
        if dentro:
            sys.exit(f"ERRO: --limpar recusado: a pasta de saída contém arquivos de entrada ({', '.join(dentro)}).")
        if input(f"Apagar TUDO em {saida_abs}? digite 'sim': ").strip().lower() != "sim":
            sys.exit("Cancelado.")
        shutil.rmtree(saida_abs)
    pt, pg = os.path.join(a.saida, "tabelas"), os.path.join(a.saida, "graficos")
    os.makedirs(pt, exist_ok=True)
    os.makedirs(pg, exist_ok=True)
    # a pasta de saída se ignora sozinha no git (o '*' também ignora este arquivo)
    gi = os.path.join(a.saida, ".gitignore")
    if not os.path.exists(gi):
        with open(gi, "w") as f:
            f.write("*\n")
    nome_crit = "Score" if a.criterio == "score" else "Hipervolume normalizado"
    pasta_final = os.path.dirname(os.path.abspath(a.final))
    blocos = []

    # ---- 1. carga, filtro e verificações
    log("[1/6] Lendo pareto_final ...")
    if a.limites and a.banco:
        sys.exit("ERRO: use --limites OU --banco, não os dois.")
    limites = carregar_limites_banco(a.banco) if a.banco else carregar_limites(a.limites)
    if a.banco:
        limites.to_csv(os.path.join(pt, "limites_usados.csv"), index=False)
    df = carregar_final(a.final, limites)
    if limites is not None:
        sem_col = [g for g in limites["grupo"] if f"{g}_kcal" not in df.columns]
        if sem_col:
            log(f"[aviso] limites sem coluna correspondente no pareto_final (NÃO auditados): {', '.join(sem_col)}")
    n_conj_arquivo, n_linhas_arquivo = df["Conjunto"].nunique(), len(df)
    df = aplicar_filtro(df, a.filtrar)

    contagem = df.groupby("Conjunto")["Execucao"].nunique()
    n_max = int(contagem.max())
    incompletos = contagem[contagem < n_max]
    contagem.rename("execucoes").reset_index().to_csv(os.path.join(pt, "execucoes_por_conjunto.csv"), index=False)
    # Regra de balanceamento:
    #  - diferença pequena (todas as configurações têm quase as mesmas seeds, p.ex. run interrompida no meio de
    #    uma seed): usa só as seeds COMUNS a todos os conjuntos e mantém todos os conjuntos;
    #  - diferença grande (alguns conjuntos com muito menos seeds): EXCLUI esses conjuntos.
    modo_balanceamento = "nenhum"
    execs_usadas = None
    conjuntos_conjuntos = df.groupby("Conjunto")["Execucao"].apply(lambda s_: set(s_.unique()))
    comum = set.intersection(*conjuntos_conjuntos.tolist())
    desbalanceado = len(incompletos) > 0 or any(len(x) != len(comum) for x in conjuntos_conjuntos)
    if desbalanceado and not a.manter_incompletos:
        if len(comum) >= max(3, math.ceil(0.8 * n_max)):
            modo_balanceamento = "truncar"
            execs_usadas = comum
            df = df[df["Execucao"].isin(comum)].copy()
        elif len(incompletos):
            modo_balanceamento = "excluir"
            df = df[~df["Conjunto"].isin(incompletos.index)].copy()
        if df.empty:
            sys.exit("ERRO: nenhum dado restou após o balanceamento.")

    fatores = [c for c in PARAMS if c in df.columns and df[c].nunique() > 1]
    log(f"  {len(df):,} linhas | {df['Conjunto'].nunique()} conjuntos usados | fatores variando: "
        f"{[PARAMS[f] for f in fatores] or 'nenhum'}")

    seeds = df.groupby("Conjunto")["Execucao"].apply(lambda s_: tuple(sorted(s_.unique())))
    seeds_iguais = seeds.nunique() == 1
    exec_por_conj = df.groupby("Conjunto")["Execucao"].nunique()
    dif_max = df["dif_pen"].max()
    n_neg, n_veg = int(df["tem_kcal_negativa"].sum()), int(df["viol_veg"].sum())
    n_lim = int((df["n_fora_limite"] > 0).sum())

    caminho_exec = a.execucoes or os.path.join(pasta_final, "execucoes.csv")
    versoes = []
    if os.path.exists(caminho_exec):
        try:
            versoes = sorted(set(pd.read_csv(caminho_exec)["versao_codigo"].astype(str)))
        except Exception:
            versoes = []

    verif = [
        ("Conjuntos no arquivo / usados na análise", f"{n_conj_arquivo} / {df['Conjunto'].nunique()}"),
        ("Balanceamento aplicado", {"nenhum": "nenhum (dados já balanceados)", "truncar": f"seeds comuns ({len(comum)})",
                                    "excluir": "conjuntos incompletos excluídos"}[modo_balanceamento]),
        ("Execuções por conjunto usado (mín–máx)", f"{exec_por_conj.min()}–{exec_por_conj.max()}"),
        ("Indivíduos nos fronts finais usados", f"{len(df):,} (de {n_linhas_arquivo:,} no arquivo)"),
        ("Fatores que variam", ", ".join(PARAMS[f] for f in fatores) or "nenhum"),
        ("Versão(ões) do código (execucoes.csv)", ", ".join(versoes) if versoes else "não informada"),
        ("Alvo calórico / custo alvo (1ª linha)",
         f"{df['Alvo Calorico'].iloc[0]:.4f} kcal / {df['Custo Alvo'].iloc[0]:.4f}" if "Alvo Calorico" in df else "–"),
        ("Erro relativo de calorias: médio / máximo", f"{df['erro_cal'].mean():.3%} / {df['erro_cal'].max():.3%}"),
        ("Erro relativo de custo: médio / máximo", f"{df['erro_custo'].mean():.3%} / {df['erro_custo'].max():.3%}"),
        ("Penalidade (cal × custo) média / mínima", f"{df['pen_calc'].mean():.5f} / {df['pen_calc'].min():.5f}"),
        ("Dietas com kcal negativa", f"{n_neg:,}"),
        ("Dietas com DGV + REV > Vegetables", f"{n_veg:,}"),
        ("Dietas com algum gene fora de [min, max]", f"{n_lim:,}" if limites is not None else "não auditado (sem --limites/--banco)"),
        ("Dietas fisicamente válidas", f"{df['valida'].mean():.2%}"),
    ]
    blocos.append("<h2>1. Verificações dos dados</h2>" + tabela_html(pd.DataFrame(verif, columns=["Item", "Valor"])))
    if len(incompletos) or modo_balanceamento != "nenhum":
        lista = ", ".join(f"#{c} ({n})" for c, n in list(incompletos.items())[:15]) + (" …" if len(incompletos) > 15 else "")
        if a.manter_incompletos:
            blocos.append(f"<div class='alerta'>{len(incompletos)} conjunto(s) com menos execuções que o máximo ({n_max}): "
                          f"{lista}. Mantidos (--manter-incompletos): testes pareados e médias marginais ficam comprometidos.</div>")
        elif modo_balanceamento == "truncar":
            blocos.append(f"<div class='alerta'>As execuções não são iguais entre os conjuntos ({len(incompletos)} com menos "
                          f"que o máximo de {n_max}: {lista}). Para preservar o balanceamento, foram usadas só as "
                          f"<b>{len(comum)} seeds comuns a todos os conjuntos</b>; nenhum conjunto foi excluído. "
                          "Tabela completa: tabelas/execucoes_por_conjunto.csv.</div>")
        elif modo_balanceamento == "excluir":
            blocos.append(f"<div class='alerta'>{len(incompletos)} conjunto(s) com muito menos execuções que o máximo "
                          f"({n_max}): {lista}. Foram <b>EXCLUÍDOS</b> da análise para preservar o balanceamento e os "
                          "testes pareados. Tabela completa: tabelas/execucoes_por_conjunto.csv.</div>")
    if len(versoes) > 1:
        blocos.append("<div class='alerta'>Há mais de uma versão do código nos dados (ver execucoes.csv): "
                      "execuções de versões diferentes não são comparáveis.</div>")
    if np.isfinite(dif_max):
        if dif_max > 1e-3:
            blocos.append(f"<div class='alerta'>A penalidade recalculada <b>não</b> reproduz fo/pontuação "
                          f"(diferença máx. {dif_max:.2e}). A fórmula do script pode estar desatualizada em relação ao "
                          "NSGA2.py/dieta.py; as colunas de erro/penalidade devem ser lidas com cautela.</div>")
        else:
            blocos.append(f"<div class='ok'>A penalidade recalculada a partir dos alvos reproduz fo/pontuação "
                          f"(diferença máx. {dif_max:.1e}).</div>")
    if not seeds_iguais:
        blocos.append("<div class='alerta'>Os conjuntos usados não têm as mesmas seeds. Os testes pareados "
                      "usam apenas seeds presentes em todos os conjuntos.</div>")
    if n_neg or n_veg or n_lim:
        blocos.append(f"<div class='alerta'>Violações físicas encontradas: {n_neg} (kcal negativa), {n_veg} (DGV+REV &gt; "
                      f"Vegetables), {n_lim} (gene fora de limites). Essas dietas são excluídas da escolha final.</div>")
    if limites is None:
        blocos.append("<div class='nota'>Sem <code>--limites</code>/<code>--banco</code>, genes fora de [p5, p95] não são auditados.</div>")

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
    elif "motivo" in info:
        txt = (f"<div class='alerta'><b>Testes pareados (Friedman/Wilcoxon) NÃO foram calculados:</b> {info['motivo']}. "
               "Sem eles, as diferenças entre configurações abaixo são apenas descritivas.</div>")
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
    marg, anovas, anova_completa = efeitos_fatores(m, df, fatores)
    if not marg.empty:
        marg.to_csv(os.path.join(pt, "medias_marginais.csv"), index=False)
        blocos.append("<h2>3. Efeito dos parâmetros e interações</h2>")
        esperado = int(np.prod([df[f].nunique() for f in fatores]))
        if len(cfg) < esperado:
            blocos.append(f"<div class='alerta'>A grade fatorial está incompleta: {len(cfg)} de {esperado} combinações. "
                          "Médias marginais e ANOVA ficam confundidas (alguns níveis só aparecem combinados com outros). "
                          "Para conclusões limpas sobre efeitos, use uma sub-grade completa com <code>--filtrar</code>.</div>")
        blocos.append(img_tag(graf_efeitos(marg, pg), "Efeitos marginais"))
        for met, nome in METRICAS:
            if met in anovas:
                an = anovas[met]
                an.to_csv(os.path.join(pt, f"anova_{met}.csv"), index=False)
                blocos.append(f"<h3>ANOVA — {nome}</h3>" + tabela_html(an.sort_values("pct_variancia", ascending=False), 12))
        if anova_completa:
            blocos.append("<div class='nota'>ANOVA tipo II com efeitos principais e interações de 2ª ordem, tendo como unidade "
                          "cada execução (seed × conjunto). Compare <i>pct_variancia</i> e <i>eta2_parcial</i> para a "
                          "importância relativa; com muitas observações, p-valores sinalizam até efeitos pequenos.</div>")
        else:
            blocos.append("<div class='alerta'><b>ANOVA simplificada:</b> o pacote <code>statsmodels</code> não está "
                          "disponível (instale com <code>pip install statsmodels</code>). As tabelas acima trazem apenas "
                          "efeitos principais, cada um testado isoladamente (sem interações e sem ajustar pelos outros "
                          "fatores).</div>")
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
        ev, formato = carregar_evolucao(a.evolucao, a.max_phdi, a.max_ace, a.chunksize)
        ev = ev[ev["Conjunto"].isin(cfg["Conjunto"])]
        if execs_usadas is not None:
            ev = ev[ev["Execucao"].isin(execs_usadas)]
        ev = ev.copy()
        conv = ev.groupby(["Conjunto", "Geracao"]).agg(
            n_seeds=("Execucao", "nunique"), n_front=("n_front", "mean"), media_phdi=("media_phdi", "mean"),
            media_ace=("media_ace", "mean"), max_phdi=("max_phdi", "mean"), max_ace=("max_ace", "mean"),
            pen_media=("pen_media", "mean"), score=("score", "mean"), hv_norm=("hv_norm", "mean")).reset_index()
        conv.to_csv(os.path.join(pt, "convergencia_por_conjunto.csv"), index=False)

        est = estabilizacao(ev)
        est_cfg = est.groupby("Conjunto")["ger_estab"].agg(["mean", "std"]).reset_index()
        est_cfg.columns = ["Conjunto", "ger_estab_media", "ger_estab_dp"]
        if ev["hv_norm"].notna().any():
            est_hv = estabilizacao(ev, col="hv_norm").groupby("Conjunto")["ger_estab"].mean().rename("ger_estab_hv_media")
            est_cfg = est_cfg.merge(est_hv.reset_index(), on="Conjunto")
        est_cfg = cfg[["Conjunto", "ranking"] + [c for c in PARAMS if c in fatores]].merge(est_cfg, on="Conjunto")
        est_cfg.sort_values("ranking").to_csv(os.path.join(pt, "estabilizacao.csv"), index=False)
        melhor = int(cfg.iloc[0]["Conjunto"])

        # auditoria da evolução (só possível no formato novo)
        aud = []
        if "tam_pop" in ev.columns:
            ev_n = ev.merge(cfg[["Conjunto", "Numero de Individuos"]], on="Conjunto")
            aud.append(("Gerações-execução registradas", f"{len(ev):,}"))
            aud.append(("Gerações com população menor que N", f"{int((ev_n['tam_pop'] < ev_n['Numero de Individuos']).sum()):,}"))
        if "n_invalidas" in ev.columns:
            valido = ev["n_invalidas"] >= 0
            aud.append(("Indivíduos inválidos em qualquer geração (população inteira)",
                        f"{int(ev.loc[valido, 'n_invalidas'].sum()):,}" if valido.any() else "não auditado"))
        blocos.append("<h2>4. Convergência ao longo das gerações</h2>"
                      + (tabela_html(pd.DataFrame(aud, columns=["Auditoria da evolução", "Valor"])) if aud else "")
                      + img_tag(graf_convergencia(ev, cfg, fatores, pg), "Convergência")
                      + img_tag(graf_detalhe_melhor(ev, melhor, rotulo_conj(cfg.iloc[0], fatores), pg), "Melhor conjunto em detalhe")
                      + "<div class='nota'>Estabilização = primeira geração em que a execução alcança 95% do ganho total "
                        "da métrica (do início ao fim da própria execução). |F1| é o número de indivíduos não dominados "
                        "em cada geração; quando atinge N, o front ocupa toda a população. A média do front pode oscilar "
                        "no início porque o front é pequeno; o hipervolume é o indicador mais estável.</div>")
        caminho_snap = a.snapshots or os.path.join(pasta_final, "pareto_snapshots.csv")
        if os.path.exists(caminho_snap):
            p_snap = graf_snapshots(pd.read_csv(caminho_snap), melhor, pg)
            if p_snap:
                blocos.append(img_tag(p_snap, "Evolução do front do melhor conjunto"))
        if len(cfg) > 2:
            blocos.append(img_tag(graf_estab(est, cfg, pg), "Velocidade x qualidade")
                          + "<h3>Estabilização por configuração</h3>" + tabela_html(est_cfg.sort_values("ranking"), 15))
    else:
        blocos.append("<h2>4. Convergência</h2><p>Arquivo de evolução não informado; seção omitida.</p>")

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
        + f"<div class='nota'>Tabelas completas (kcal por grupo): tabelas/dietas_selecionadas_conjunto{conj}.csv "
          f"e tabelas/front_combinado_conjunto{conj}.csv. Percentuais e pontuações por grupo podem ser recalculados "
          "a partir das kcal com <code>construir_dieta(x)</code>.</div>")

    # ---- 6. metodologia
    blocos.append(
        "<h2>6. Notas metodológicas</h2><ul>"
        "<li>Cada execução (seed) é a unidade amostral; as seeds são pareadas entre conjuntos (mesma seed = mesmo bloco).</li>"
        "<li>Conjuntos com menos execuções que o máximo são excluídos para manter o balanceamento (ver seção 1).</li>"
        "<li>Os fronts analisados são os exportados pelo NSGA2.py (|F1| truncado em N quando F1 excede N).</li>"
        "<li>Máximos teóricos usados na normalização: "
        f"fo_phdi={a.max_phdi:g}, fo_aceitabilidade={a.max_ace:g} (ajustáveis por --max-phdi / --max-ace).</li>"
        "<li>Dietas &quot;válidas&quot;: sem kcal negativa, com DGV+REV ≤ Vegetables e, se --limites for informado, "
        "genes dentro de [min, max].</li></ul>")

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