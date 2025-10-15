import NSGA2
import os
import glob

QTDS_INDIVIDUOS = [50, 100, 200]
QTDS_GERACOES = [5, 10, 20]
CHANCES_CROSS = [0.6, 0.8, 1.0]
CHANCES_MUT = [0.01, 0.05, 0.1]
SEEDS = range(42, 46)

diretorio = "./img"
arqs = glob.glob(os.path.join(diretorio, "*"))

for arq in arqs:
    if os.path.isfile(arq):
        os.remove(arq)

diretorio = "./stats"
arqs = glob.glob(os.path.join(diretorio, "*"))

for arq in arqs:
    if os.path.isfile(arq):
        os.remove(arq)

for i in QTDS_INDIVIDUOS:
    for g in QTDS_GERACOES:
        for c in CHANCES_CROSS:
            for m in CHANCES_MUT:
                for s in SEEDS:
                    NSGA2.main(i, g, c, m, s)