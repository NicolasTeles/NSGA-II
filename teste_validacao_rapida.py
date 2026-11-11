#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste rápido de validação local (rode ANTES de disparar uma run longa).

Roda 2 configurações pequenas com 3 seeds usando o NSGA2.py/dieta.py atuais e confere:
  1. Nenhum erro de execução.
  2. 100% das dietas fisicamente válidas: kcal >= 0 e DGV + REV <= Vegetables
     (no front final E na população inteira de TODAS as gerações, via evolucao_agregada.csv).
  3. Grupos de valor fixo constantes.
  4. População com N indivíduos em todas as gerações (ordenação não dominada correta).
  5. Mutação funcionando com C < 1 (front com soluções distintas).
  6. Reprodutibilidade: mesma seed => mesmo resultado.
  7. Tamanho dos arquivos de saída.
"""
import os
import shutil
import time

import numpy as np
import pandas as pd

import NSGA2

PASTA = "./teste_validacao"


def rodar(cfg, seed, execucao, pasta):
    return NSGA2.main(
        NUM_INDIVIDUOS=cfg["N"], GERACOES=cfg["G"], CHANCE_CROSSOVER=cfg["C"], CHANCE_MUTACAO=cfg["M"],
        SEED=seed, CROSSOVER_INDEX=cfg["ci"], DISTRIBUTION_INDEX=cfg["di"],
        execucao=execucao, conjunto=cfg["conjunto"],
        pasta_saida=pasta, salvar_graficos=False, salvar_log_txt=False, verbose=False,
    )


def validar():
    configs = [
        {"conjunto": 1, "N": 60, "G": 80, "C": 1.0, "M": 0.1, "ci": 5, "di": 20},
        {"conjunto": 2, "N": 60, "G": 80, "C": 0.7, "M": 0.1, "ci": 15, "di": 35},
    ]
    seeds = [42, 43, 44]

    shutil.rmtree(PASTA, ignore_errors=True)
    os.makedirs(PASTA)
    t0 = time.time()
    for cfg in configs:
        for exec_id, s in enumerate(seeds, start=1):
            print(f"-> Conjunto {cfg['conjunto']} | Exec {exec_id} (seed {s}) | N={cfg['N']} G={cfg['G']} C={cfg['C']}")
            rodar(cfg, s, exec_id, PASTA)
    dt = time.time() - t0

    final = pd.read_csv(f"{PASTA}/pareto_final.csv")
    evol = pd.read_csv(f"{PASTA}/evolucao_agregada.csv")
    kcal_cols = [c for c in final.columns if c.endswith("_kcal")]

    neg_final = int((final[kcal_cols].min(axis=1) < -1e-9).sum())
    viol_final = int(((final["vegetables_kcal"] - final["dgv_kcal"] - final["rev_kcal"]) < -1e-9).sum())
    inv_evol = int(evol.loc[evol["n_invalidas"] >= 0, "n_invalidas"].sum())
    auditoria_evol = bool((evol["n_invalidas"] >= 0).all())

    fixos = ["nuts_and_peanuts_kcal", "whole_cereals_kcal", "fish_and_seafood_kcal", "dgv_kcal", "refined_cereals_kcal"]
    fixos_ok = all(final[c].std() < 1e-6 for c in fixos)

    n_por_conj = {c["conjunto"]: c["N"] for c in configs}
    pop_ok = bool((evol["tam_pop"] == evol["Conjunto"].map(n_por_conj)).all())
    diversos = final[final["Conjunto"] == 2].groupby("Execucao")["fo_phdi"].nunique().min() > 1

    # reprodutibilidade: repete a 1ª execução em outra pasta
    rep = PASTA + "_repeticao"
    shutil.rmtree(rep, ignore_errors=True)
    rodar(configs[0], seeds[0], 1, rep)
    a = final[(final.Conjunto == 1) & (final.Execucao == 1)][["fo_phdi", "fo_aceitabilidade"]].values
    b = pd.read_csv(f"{rep}/pareto_final.csv")[["fo_phdi", "fo_aceitabilidade"]].values
    reprodutivel = a.shape == b.shape and np.allclose(a, b)
    shutil.rmtree(rep, ignore_errors=True)

    tam = {f: os.path.getsize(f"{PASTA}/{f}") / 1024 for f in
           ("pareto_final.csv", "evolucao_agregada.csv", "pareto_snapshots.csv")}

    ok = lambda c: "OK " if c else "FALHA"
    print("\n" + "=" * 70)
    print(f"RESUMO DA VALIDAÇÃO LOCAL (concluído em {dt:.1f}s)")
    print("=" * 70)
    print(f"[{ok(neg_final == 0)}] kcal negativa no front final: {neg_final}")
    print(f"[{ok(viol_final == 0)}] violações DGV + REV > Vegetables no front final: {viol_final}")
    print(f"[{ok(auditoria_evol and inv_evol == 0)}] indivíduos inválidos em todas as gerações (população inteira): "
          f"{inv_evol if auditoria_evol else 'auditoria indisponível'}")
    print(f"[{ok(fixos_ok)}] grupos de valor fixo constantes")
    print(f"[{ok(pop_ok)}] população com N indivíduos em todas as gerações")
    print(f"[{ok(diversos)}] front com soluções distintas também com C < 1")
    print(f"[{ok(reprodutivel)}] mesma seed => mesmo resultado")
    print("Tamanho dos arquivos: " + ", ".join(f"{k} {v:.0f} KB" for k, v in tam.items()))
    print(f"Médias por conjunto — fo_phdi: {final.groupby('Conjunto')['fo_phdi'].mean().round(2).to_dict()} | "
          f"fo_aceitabilidade: {final.groupby('Conjunto')['fo_aceitabilidade'].mean().round(2).to_dict()}")
    aprovado = (neg_final == 0 and viol_final == 0 and auditoria_evol and inv_evol == 0 and fixos_ok
                and pop_ok and diversos and reprodutivel)
    print("VALIDAÇÃO APROVADA: pronto para os experimentos." if aprovado
          else "ATENÇÃO: há falhas acima; não dispare a run longa antes de corrigi-las.")
    print("=" * 70)
    return aprovado


if __name__ == "__main__":
    validar()