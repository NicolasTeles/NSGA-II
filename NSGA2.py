import random
import matplotlib.pyplot as plt
import time
import copy
import numpy as np
import os
import math

def autonomia_simples(diametro, potencia, capacidade):
    return capacidade / (potencia * diametro)

def tempo_aceleracao_simples(diametro, potencia, capacidade):
    return (capacidade + diametro) / potencia

def autonomia_complexo(diametro, potencia, capacidade):
    denominador = max(potencia * abs(math.sin(diametro)), 0.1)
    return math.sqrt(capacidade) / denominador

def tempo_aceleracao_complexo(diametro, potencia, capacidade):
    return math.sqrt(capacidade + abs(math.cos(diametro))) / math.sqrt(potencia)

class Solucao:
    def __init__(self, diametro_roda, potencia_motor, capacidade_bateria):
        self.diametro_roda = diametro_roda
        self.potencia_motor = potencia_motor
        self.capacidade_bateria = capacidade_bateria
        
        self.autonomia = autonomia_complexo(diametro_roda, potencia_motor, capacidade_bateria)
        self.tempo_aceleracao = tempo_aceleracao_complexo(diametro_roda, potencia_motor, capacidade_bateria)
        
        self.domination_count = 0
        self.dominates = []
        
        self.front = 0
        self.crowding_distance = -1

def gerar_individuos(n=10):
    individuos = []
    for _ in range(n):
        diametro_roda = random.uniform(10, 30)  # exemplo: entre 10 e 30
        potencia_motor = random.uniform(50, 200)  # exemplo: entre 50 e 200
        capacidade_bateria = random.uniform(100, 500)  # exemplo: entre 100 e 500
        individuo = Solucao(diametro_roda, potencia_motor, capacidade_bateria)
        individuos.append(individuo)
    return individuos

def crossover(parents, chance_mutacao: float) -> Solucao:
    parent1, parent2 = parents[:]
    if random.random() < chance_mutacao:
        capacidade = random.uniform(100, 500)
    elif(random.random() < 0.5):
        capacidade = parent1.capacidade_bateria
    else:
        capacidade = parent2.capacidade_bateria
    
    if random.random() < chance_mutacao:
        potencia = random.uniform(50, 200)
    elif(random.random() < 0.5):
        potencia = parent1.potencia_motor
    else:
        potencia = parent2.potencia_motor
    
    if random.random() < chance_mutacao:
        diametro = random.uniform(10, 30)
    elif(random.random() < 0.5):
        diametro = parent1.diametro_roda
    else:
        diametro = parent2.diametro_roda
    
    return Solucao(diametro, potencia, capacidade)

def offspring_aleatoria(populacao: list[Solucao], chance_crossover: float, chance_mutacao: float):
    nova_pop = populacao[:]
    for _ in range(len(populacao)):
        index1, index2 = random.sample(range(len(populacao)), 2)
        parents = [populacao[index1], populacao[index2]]
        if random.random() <= chance_crossover:
            novo_ind = crossover(parents, chance_mutacao)
        else:
            novo_ind = copy.deepcopy(random.choice(parents))
        nova_pop.append(novo_ind)
    return nova_pop

def offspring(populacao: list[Solucao], chance_crossover: float, chance_mutacao: float):
    nova_pop = populacao[:]
    for i in range(len(populacao)):
        parents = []
        for _ in range(2):
            index1, index2 = random.sample(range(len(populacao)), 2)
            candidate1, candidate2 = populacao[index1], populacao[index2]
            
            if candidate1.front == candidate2.front:
                if candidate1.crowding_distance > candidate2.crowding_distance:
                    parents.append(candidate1)
                else:
                    parents.append(candidate2)
            elif candidate1.front < candidate2.front:
                parents.append(candidate1)
            else:
                parents.append(candidate2)
        if random.random() <= chance_crossover:
            novo_ind = crossover(parents, chance_mutacao)
        else:
            novo_ind = copy.deepcopy(random.choice(parents))
        nova_pop.append(novo_ind)
    return nova_pop

def domina(dominates: Solucao, dominated: Solucao):
    condicao1 = (dominates.autonomia >= dominated.autonomia and dominates.tempo_aceleracao <= dominated.tempo_aceleracao)
    condicao2 = (dominates.autonomia > dominated.autonomia or dominates.tempo_aceleracao < dominated.tempo_aceleracao)
    return condicao1 and condicao2

def non_dominated_sorting(pop: list[Solucao]):
    for ind1 in pop:
        for ind2 in pop:
            if domina(ind1, ind2):
                ind2.domination_count += 1
                ind1.dominates.append(ind2)
                
    fronts = []      
    fronts.append([])
    for individuo in pop:
        if individuo.domination_count == 0:
            individuo.front = 1
            fronts[0].append(individuo)
    i = 0
    while True:
        for individuo in fronts[i]:
            for dominated in individuo.dominates:
                dominated.domination_count -= 1
        
        indiv_sem_frente = False
        fronts.append([])
        for individuo in pop:
            if individuo.front == 0:
                indiv_sem_frente = True
                if individuo.domination_count == 0:
                    individuo.front = i+2
                    fronts[i+1].append(individuo)
        i += 1
        if not indiv_sem_frente:
            break
    
    # for indiv in pop:
    #     print(f"auton: {indiv.autonomia} aceler: {indiv.tempo_aceleracao} domination_count:{indiv.domination_count} front: {indiv.front}")
    return fronts

def crowding_distance_sorting(front: list[Solucao]):
    for solucao in front:
        solucao.crowding_distance = 0
        
    # para autonomia
        
    front.sort(key=lambda sol: sol.autonomia, reverse=True)
    front[0].crowding_distance, front[-1].crowding_distance = float('inf'), float('inf')
    
    max, min = front[0], front[-1]
    
    for i, solucao in enumerate(front[1:-1], start=1):
        solucao.crowding_distance += (front[i-1].autonomia - front[i+1].autonomia) / (max.autonomia - min.autonomia)
        
    # para aceleracao
    
    front.sort(key=lambda sol: sol.tempo_aceleracao)
    front[0].crowding_distance, front[-1].crowding_distance = float('inf'), float('inf')
    
    max, min = front[-1], front[0]
    
    for i, solucao in enumerate(front[1:-1], start=1):
        solucao.crowding_distance += (front[i+1].tempo_aceleracao - front[i-1].tempo_aceleracao) / (max.tempo_aceleracao - min.tempo_aceleracao)
    
    front.sort(key=lambda sol: sol.crowding_distance, reverse=True)
    return front

# REPASSAR FRENTES COMPLETAS PARA A PROXIMA GERACAO E FAZER CROWD DISTANCE SORTING
def gerar_prox_geracao(fronts: list[list[Solucao]], tam_pop_inicial):
    tam_pop_atual = 0
    nova_pop = []
    for front in fronts:
        if len(front)+tam_pop_atual <= tam_pop_inicial:
            nova_pop.extend(front)
            tam_pop_atual += len(front)
        else:
            new_front = crowding_distance_sorting(front)
            i = 0
            while len(nova_pop) < tam_pop_inicial:
                nova_pop.append(new_front[i])
                i += 1
            break
    return nova_pop


def plot(pop, filename: str):
    autonomias = [ind.autonomia for ind in pop]
    tempos = [ind.tempo_aceleracao for ind in pop]

    plt.figure(figsize=(10, 6))
    plt.scatter(autonomias, tempos, color='blue', label='Indivíduos')
    plt.xlabel('Autonomia')
    plt.ylabel('Tempo de Aceleração')
    plt.title('Indivíduos: Autonomia vs Tempo de Aceleração')
    plt.legend()
    plt.grid(True)

    plt.xlim(min(autonomias), max(autonomias))
    plt.ylim(min(tempos), max(tempos))

    plt.savefig(filename + ".png")
    plt.close()

def resetar_valores(pop: list[Solucao]):
    for ind in pop:
        ind.domination_count = 0
        ind.dominates = []
        
        ind.front = 0
        ind.crowding_distance = -1

def calcular_estatisticas(pop: list[Solucao]):
    autonomias = np.array([ind.autonomia for ind in pop])
    aceleracoes = np.array([ind.tempo_aceleracao for ind in pop])
    
    media_autonomia = autonomias.mean()
    desvio_autonomia = autonomias.std(ddof=1)
    media_aceleracao = aceleracoes.mean()
    desvio_aceleracao = aceleracoes.std(ddof=1)

    return {
        "media_autonomia": media_autonomia,
        "desvio_autonomia": desvio_autonomia,
        "media_aceleracao": media_aceleracao,
        "desvio_aceleracao": desvio_aceleracao
    }

def main(NUM_INDIVIDUOS=500, GERACOES=20, CHANCE_CROSSOVER=1.0, CHANCE_MUTACAO=0.05, SEED=42):
    random.seed(SEED)
    
    medias_autonomia = []
    desvios_autonomia = []
    medias_aceleracao = []
    desvios_aceleracao = []
    
    arquivo_texto = open(f"./stats/stats{SEED}.txt", "a")
    arquivo_texto.write(f"{NUM_INDIVIDUOS} individuos, {GERACOES} geracoes {CHANCE_CROSSOVER} chance de crossover e {CHANCE_MUTACAO} chance de mutacao\n")
    
    start = time.process_time()

    populacao_inicial = gerar_individuos(NUM_INDIVIDUOS)
    for i, ind in enumerate(populacao_inicial):
        print(f"Indivíduo {i+1}: diametro_roda={ind.diametro_roda:.2f}, potencia_motor={ind.potencia_motor:.2f}, capacidade_bateria={ind.capacidade_bateria:.2f}")

    populacao = populacao_inicial[:]
    nova_populacao = None
    populacao = offspring_aleatoria(populacao_inicial, CHANCE_CROSSOVER, CHANCE_MUTACAO)

    for i in range(GERACOES):
        if i != 0:
            populacao = offspring(populacao, CHANCE_CROSSOVER, CHANCE_MUTACAO)
            resetar_valores(populacao)
        # print(len(populacao))
        frentes = non_dominated_sorting(populacao)
        nova_populacao = gerar_prox_geracao(frentes, len(populacao_inicial))
        populacao = nova_populacao[:]
        print(f"\nFIM DA GERAÇÃO {i+1}\n")
        stats = calcular_estatisticas(populacao)
        
        arquivo_texto.write(f"Geração {i+1}: Autonomia média = {stats['media_autonomia']:.4f} ± {stats['desvio_autonomia']:.4f}\n")
        arquivo_texto.write(f"Geração {i+1}: Aceleração média = {stats['media_aceleracao']:.4f} ± {stats['desvio_aceleracao']:.4f}\n\n")
        
        medias_autonomia.append(stats["media_autonomia"])
        desvios_autonomia.append(stats["desvio_autonomia"])
        medias_aceleracao.append(stats["media_aceleracao"])
        desvios_aceleracao.append(stats["desvio_aceleracao"])

    end = time.process_time()

    print(f"CPU time: {end - start:.6f} seconds")
    
    arquivo_texto.write(f"CPU time: {end - start:.6f} seconds\n\n")
    
    folder_path = f"./img/seed{SEED}"
    os.makedirs(folder_path, exist_ok=True)

    plot(populacao_inicial, f"./img/seed{SEED}/populacao inicial")
    plot(nova_populacao, f"./img/seed{SEED}/pop{NUM_INDIVIDUOS}-{GERACOES}-{CHANCE_CROSSOVER}-{CHANCE_MUTACAO}")
    
    folder_path = f"./stats/seed{SEED}"
    os.makedirs(folder_path, exist_ok=True)
    
    plt.figure()
    plt.plot(range(GERACOES), medias_autonomia, label="Autonomia média")
    plt.fill_between(range(GERACOES),
                    np.array(medias_autonomia) - np.array(desvios_autonomia),
                    np.array(medias_autonomia) + np.array(desvios_autonomia),
                    alpha=0.2)
    plt.legend()
    plt.xlabel("Geração")
    plt.ylabel("Autonomia")
    plt.title("Evolução da média e desvio padrão da autonomia")
    plt.savefig(f"./stats/seed{SEED}/autonomia{NUM_INDIVIDUOS}-{GERACOES}-{CHANCE_CROSSOVER}-{CHANCE_MUTACAO}.png")
    
    plt.figure()
    plt.plot(range(GERACOES), medias_aceleracao, label="Tempo de aceleracao médio")
    plt.fill_between(range(GERACOES),
                    np.array(medias_aceleracao) - np.array(desvios_aceleracao),
                    np.array(medias_aceleracao) + np.array(desvios_aceleracao),
                    alpha=0.2)
    plt.legend()
    plt.xlabel("Geração")
    plt.ylabel("Tempo de aceleracao")
    plt.title("Evolução da média e desvio padrão da aceleracao")
    plt.savefig(f"./stats/seed{SEED}/aceleracao{NUM_INDIVIDUOS}-{GERACOES}-{CHANCE_CROSSOVER}-{CHANCE_MUTACAO}.png")
    
    plt.close('all')
    return (end - start)

if __name__ == '__main__':
    main()