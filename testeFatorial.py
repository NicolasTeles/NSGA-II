import NSGA2
import os
import shutil

def clear():
    for diretorio in ["./img", "./stats"]:
        shutil.rmtree(diretorio, ignore_errors=True)
        os.makedirs(diretorio)

def run1():
    # --- Run 1: algorithm parameters, crossover hyperparameters fixed at neutral ---
    NUMS_INDIVIDUOS = [100, 300]
    NUMS_GERACOES   = [200, 500]
    CHANCES_CROSS   = [0.7, 1.0]
    CHANCES_MUT     = [0.1, 0.3]
    SEEDS           = [_ for _ in range(30)]

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
    
def run2():
    # --- Run 2: hyperparameters, generational parameters fixed at neutral ---
        CROSSOVER_INDEXES    = [2, 5, 10, 15, 20, 40, 55]
        DISTRIBUTION_INDEXES = [5, 10, 15, 20, 35, 45, 65]
        SEEDS              = [_ for _ in range(30)]
    
        NUM_INDIVIDUOS_FIXO = 300
        NUM_GERACOES_FIXO   = 500
        CHANCE_CROSS_FIXO   = 1.0
        CHANCE_MUT_FIXO     = 0.1
    
    
        clear()
    
        conjunto = 0
        arquivo_tempos = open("./stats/tempo.txt", "w")
    
        for ci in CROSSOVER_INDEXES:
            for di in DISTRIBUTION_INDEXES:
                        conjunto += 1
                        tempo_total = 0.0
    
                        for execucao, s in enumerate(SEEDS, start=1):
                            print(f"[Conjunto {conjunto}] CROSSOVER INDEX={ci} DISTRIBUTION INDEX={di} SEED={s}")
                            tempo = NSGA2.main(
                                NUM_INDIVIDUOS=NUM_INDIVIDUOS_FIXO,
                                GERACOES=NUM_GERACOES_FIXO,
                                CHANCE_CROSSOVER=CHANCE_CROSS_FIXO,
                                CHANCE_MUTACAO=CHANCE_MUT_FIXO,
                                SEED=s,
                                CROSSOVER_INDEX=ci,
                                DISTRIBUTION_INDEX=di,
                                execucao=execucao,
                                conjunto=conjunto
                            )
                            tempo_total += tempo
    
                        media_tempo = tempo_total / len(SEEDS)
                        arquivo_tempos.write(
                            f"Conjunto {conjunto}: CI={ci} DI={di} | "
                            f"tempo médio = {media_tempo:.2f}s\n"
                        )
    
        arquivo_tempos.write("\nRun 1 concluído.\n\n")
        arquivo_tempos.close()
        print("Run 2 concluído. CSVs em ./stats/pareto_final.csv e ./stats/pareto_evolucao.csv")

def run3():
    NUMS_INDIVIDUOS  = [200, 300, 400]      # centered on 300, exploring slightly wider
    NUMS_GERACOES    = [400, 500, 600]      # centered on 500, exploring slightly wider
    CROSSOVER_INDEXES    = [5, 10, 15]      # top 3 from Run 2, tightly around winner
    DISTRIBUTION_INDEXES = [15, 20, 35]     # centered on 20, exploring nearby range
    SEEDS = [_ for _ in range(30, 60)]

    CHANCE_CROSS_FIXO = 1.0   # settled in Run 1
    CHANCE_MUT_FIXO   = 0.1   # settled in Run 1

    clear()

    conjunto = 0
    arquivo_tempos = open("./stats/tempo.txt", "w")

    for n in NUMS_INDIVIDUOS:
        for g in NUMS_GERACOES:
            for ci in CROSSOVER_INDEXES:
                for di in DISTRIBUTION_INDEXES:
                    conjunto += 1
                    tempo_total = 0.0

                    for execucao, s in enumerate(SEEDS, start=1):
                        print(f"[Conjunto {conjunto}] N={n} G={g} CI={ci} DI={di} SEED={s}")
                        tempo = NSGA2.main(
                            NUM_INDIVIDUOS=n,
                            GERACOES=g,
                            CHANCE_CROSSOVER=CHANCE_CROSS_FIXO,
                            CHANCE_MUTACAO=CHANCE_MUT_FIXO,
                            SEED=s,
                            CROSSOVER_INDEX=ci,
                            DISTRIBUTION_INDEX=di,
                            execucao=execucao,
                            conjunto=conjunto
                        )
                        tempo_total += tempo

                    media_tempo = tempo_total / len(SEEDS)
                    arquivo_tempos.write(
                        f"Conjunto {conjunto}: N={n} G={g} CI={ci} DI={di} | "
                        f"tempo médio = {media_tempo:.2f}s\n"
                    )

    arquivo_tempos.write("\nRun 3 concluído.\n\n")
    arquivo_tempos.close()
    print("Run 3 concluído. CSVs em ./stats/pareto_final.csv e ./stats/pareto_evolucao.csv")

def main():
    run3()

if __name__ == '__main__':
    main()