import NSGA2
import os
import shutil

def clear():
    for diretorio in ["./img", "./stats"]:
        shutil.rmtree(diretorio)
        os.makedirs(diretorio)

def main():
    # --- Run 1: algorithm parameters, crossover hyperparameters fixed at neutral ---
    NUMS_INDIVIDUOS = [100, 300]
    NUMS_GERACOES   = [200, 500]
    CHANCES_CROSS   = [0.7, 1.0]
    CHANCES_MUT     = [0.1, 0.3]
    SEEDS           = [42, 43, 44]

    CROSSOVER_INDEX_FIXO    = 20
    DISTRIBUTION_INDEX_FIXO = 20

    clear()

    conjunto = 0
    arquivo_tempos = open("./stats/tempo.txt", "w")

    for n in NUMS_INDIVIDUOS:
        for g in NUMS_GERACOES:
            for c in CHANCES_CROSS:
                for m in CHANCES_MUT:
                    conjunto += 1
                    tempo_total = 0.0

                    for execucao, s in enumerate(SEEDS, start=1):
                        print(f"[Conjunto {conjunto}] N={n} G={g} C={c} M={m} SEED={s}")
                        tempo = NSGA2.main(
                            NUM_INDIVIDUOS=n,
                            GERACOES=g,
                            CHANCE_CROSSOVER=c,
                            CHANCE_MUTACAO=m,
                            SEED=s,
                            CROSSOVER_INDEX=CROSSOVER_INDEX_FIXO,
                            DISTRIBUTION_INDEX=DISTRIBUTION_INDEX_FIXO,
                            execucao=execucao,
                            conjunto=conjunto
                        )
                        tempo_total += tempo

                    media_tempo = tempo_total / len(SEEDS)
                    arquivo_tempos.write(
                        f"Conjunto {conjunto}: N={n} G={g} C={c} M={m} | "
                        f"tempo médio = {media_tempo:.2f}s\n"
                    )

    arquivo_tempos.write("\nRun 1 concluído.\n\n")
    arquivo_tempos.close()
    print("Run 1 concluído. CSVs em ./stats/pareto_final.csv e ./stats/pareto_evolucao.csv")

if __name__ == '__main__':
    main()