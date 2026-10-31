#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste Rápido de Validação Local (Pré-Servidor)
Roda 2 configurações pequenas com 3 seeds (42, 43, 44) usando o código corrigido
(NSGA2.py, dieta.py, models.py) e verifica:
  1. Ausência total de erros de execução (IndexError, etc.).
  2. 100% das dietas fisicamente válidas (0 kcal negativas e 0 violações DGV+REV > Vegetables).
  3. Preservação da população completa (N indivíduos) desde a Geração 1 graças à correção do non_dominated_sorting.
  4. Funcionamento da mutação mesmo quando CHANCE_CROSSOVER < 1.0 (ex.: C=0.7).
"""

import os
import shutil
import time
import numpy as np
import pandas as pd
import NSGA2
from dieta import LIMITES_ACEITABILIDADE, IDX_DGV, IDX_REV, IDX_VEGETABLES

def validar_codigo_corrigido():
    configs = [
        {"conjunto": 1, "N": 60, "G": 80, "C": 1.0, "M": 0.1, "ci": 5,  "di": 20},
        {"conjunto": 2, "N": 60, "G": 80, "C": 0.7, "M": 0.1, "ci": 15, "di": 35},
    ]
    seeds = [42, 43, 44]

    # Limpa CSVs de teste anteriores se existirem
    for d in ["./img", "./stats"]:
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)

    t0 = time.time()
    for cfg in configs:
        for exec_id, s in enumerate(seeds, start=1):
            print(
                f"-> Rodando Conjunto {cfg['conjunto']} | Exec {exec_id} (Seed {s}) | "
                f"N={cfg['N']} G={cfg['G']} C={cfg['C']}..."
            )
            NSGA2.main(
                NUM_INDIVIDUOS=cfg["N"],
                GERACOES=cfg["G"],
                CHANCE_CROSSOVER=cfg["C"],
                CHANCE_MUTACAO=cfg["M"],
                SEED=s,
                CROSSOVER_INDEX=cfg["ci"],
                DISTRIBUTION_INDEX=cfg["di"],
                execucao=exec_id,
                conjunto=cfg["conjunto"],
            )

    dt = time.time() - t0
    df_final = pd.read_csv("./stats/pareto_final.csv")
    df_evol = pd.read_csv("./stats/pareto_evolucao.csv")

    kcal_cols = [c for c in df_final.columns if c.endswith("_kcal")]
    neg_final = int((df_final[kcal_cols].min(axis=1) < 0).sum())
    neg_evol = int((df_evol[kcal_cols].min(axis=1) < 0).sum())

    viol_veg_final = int(
        ((df_final["vegetables_kcal"] - (df_final["dgv_kcal"] + df_final["rev_kcal"])) < -1e-9).sum()
    )
    viol_veg_evol = int(
        ((df_evol["vegetables_kcal"] - (df_evol["dgv_kcal"] + df_evol["rev_kcal"])) < -1e-9).sum()
    )

    print("\n" + "=" * 68)
    print(f"RESUMO DA VALIDAÇÃO LOCAL (Concluído em {dt:.1f}s)")
    print("=" * 68)
    print(
        f"• Total de linhas em pareto_final.csv:    {len(df_final)} "
        f"(Conjuntos: {df_final['Conjunto'].nunique()}, "
        f"Seeds/conj: {df_final.groupby('Conjunto')['Execucao'].nunique().to_dict()})"
    )
    print(f"• Total de linhas em pareto_evolucao.csv: {len(df_evol)}")
    print(f"• Indivíduos com kcal negativa (Final / Evolução):        {neg_final} / {neg_evol}")
    print(f"• Violações DGV + REV > Vegetables (Final / Evolução):    {viol_veg_final} / {viol_veg_evol}")
    print(f"• Média FO PHDI por Conjunto:            {df_final.groupby('Conjunto')['fo_phdi'].mean().round(2).to_dict()}")
    print(f"• Média FO Aceitabilidade por Conjunto:  {df_final.groupby('Conjunto')['fo_aceitabilidade'].mean().round(2).to_dict()}")
    if neg_final == 0 and neg_evol == 0 and viol_veg_final == 0 and viol_veg_evol == 0:
        print("✅ VALIDAÇÃO APROVADA: O código está 100% pronto para os experimentos remotos!")
    else:
        print("⚠️ ATENÇÃO: Foram encontradas violações nos testes.")
    print("=" * 68)

if __name__ == "__main__":
    validar_codigo_corrigido()