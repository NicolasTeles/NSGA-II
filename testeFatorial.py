import NSGA2
import os
import glob

QTDS_INDIVIDUOS_SIMPLES = [50, 100, 200]
QTDS_GERACOES_SIMPLES = [5, 10, 20]
CHANCES_CROSS = [0.6, 0.8, 1.0]
CHANCES_MUT = [0.01, 0.05, 0.1]
SEEDS = range(42, 46)

QTDS_INDIVIDUOS_COMPLEXO = [200, 500, 700]
QTDS_GERACOES_COMPLEXO = [10, 20, 35]

diretorio = "./img"
arqs = glob.glob(os.path.join(diretorio, "*"))

for arq in arqs:
    if os.path.isfile(arq):
        os.remove(arq)
    if os.path.isdir(arq):
        for f in arq:
            if os.path.isfile(arq):
                os.remove(arq)

diretorio = "./stats"
arqs = glob.glob(os.path.join(diretorio, "*"))

for arq in arqs:
    if os.path.isfile(arq):
        os.remove(arq)
    if os.path.isdir(arq):
        for f in arq:
            if os.path.isfile(arq):
                os.remove(arq)

arquivo_tempos = open(f"./stats/tempo.txt", "w")

input_text = """Selecione a opcao que deseja:
1: Funcoes e valores simples
2: Funcoes e valores complexos
3: Funcoes simples e valores complexos
4: Funcoes complexas e valores simples\n"""

opcao = int(input(input_text))

match opcao:
    case 1:
        QTDS_INDIVIDUOS = QTDS_INDIVIDUOS_SIMPLES
        QTDS_GERACOES = QTDS_GERACOES_SIMPLES
        FUNCS_COMPLEXAS = False
    case 2:
        QTDS_INDIVIDUOS = QTDS_INDIVIDUOS_COMPLEXO
        QTDS_GERACOES = QTDS_GERACOES_COMPLEXO
        FUNCS_COMPLEXAS = True
    case 3:
        QTDS_INDIVIDUOS = QTDS_INDIVIDUOS_COMPLEXO
        QTDS_GERACOES = QTDS_GERACOES_COMPLEXO
        FUNCS_COMPLEXAS = False
    case 4:
        QTDS_INDIVIDUOS = QTDS_INDIVIDUOS_SIMPLES
        QTDS_GERACOES = QTDS_GERACOES_SIMPLES
        FUNCS_COMPLEXAS = True
        
for i in QTDS_INDIVIDUOS:
    for g in QTDS_GERACOES:
        tempo_execucao = 0
        cont=0
        for s in SEEDS:
            for c in CHANCES_CROSS:
                for m in CHANCES_MUT:
                    tempo_execucao += NSGA2.main(i, g, c, m, s, FUNCS_COMPLEXAS)
                    cont += 1
        arquivo_tempos.write(f"Media de tempo de execucao para {i} individuos e {g} geracoes: {(tempo_execucao / cont):.6f} segundos\n\n")