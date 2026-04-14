import random
import matplotlib.pyplot as plt
import time
import copy
import numpy as np
import os
import math
from enum import Enum, auto

class Otimizacao(Enum):
    MIN = auto()
    MAX = auto()

class Problema:
    def __init__(self, funcoes_objetivo, limites, tipos_otimizacao):
        """
        funcoes_objetivo: list of callables, each f(x) returning a scalar
        limites: list of (min, max) tuples for each decision variable
        tipos_otimizacao: list[Otimizacao] same length as funcoes_objetivo
        """
        self.funcoes_objetivo = funcoes_objetivo
        self.limites = limites
        self.tipos_otimizacao = tipos_otimizacao

    def gerar_individuo(self):
        """Cria uma lista de variáveis aleatórias dentro dos limites"""
        return [random.uniform(l[0], l[1]) for l in self.limites]

    def avaliar(self, variaveis):
        """Retorna a lista dos valores das funções objetivo"""
        return [f(variaveis) for f in self.funcoes_objetivo]

class Solucao:
    def __init__(self, x, problema: Problema):
        self.x = x
        self.objetivos = problema.avaliar(x)

        self.domination_count = 0
        self.dominates = []
        self.front = 0
        self.crowding_distance = -1

def autonomia_simples(diametro, potencia, capacidade):
    return capacidade / (potencia * diametro)

def tempo_aceleracao_simples(diametro, potencia, capacidade):
    return (capacidade + diametro) / potencia

def autonomia_complexo(diametro, potencia, capacidade):
    denominador = max(potencia * abs(math.sin(diametro)), 0.1)
    return math.sqrt(capacidade) / denominador

def tempo_aceleracao_complexo(diametro, potencia, capacidade):
    return math.sqrt(capacidade + abs(math.cos(diametro))) / math.sqrt(potencia)


def gerar_individuos(n=10):
    individuos = []
    for _ in range(n):
        diametro_roda = random.uniform(10, 30)  # exemplo: entre 10 e 30
        potencia_motor = random.uniform(50, 200)  # exemplo: entre 50 e 200
        capacidade_bateria = random.uniform(100, 500)  # exemplo: entre 100 e 500
        individuo = Solucao(diametro_roda, potencia_motor, capacidade_bateria)
        individuos.append(individuo)
    return individuos

def crossover(p1: Solucao, p2: Solucao, problema: Problema, chance_mutacao: float):
    x = []
    for (min_lim, max_lim), v1, v2 in zip(problema.limites, p1.x, p2.x):
        if random.random() < chance_mutacao:
            x.append(random.uniform(min_lim, max_lim))
        else:
            x.append(random.choice([v1, v2]))
    return Solucao(x, problema)
    
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

def domina(a: Solucao, b: Solucao, problema: Problema):
    melhor_ou_igual = True
    melhor_pelo_menos_um = False
    
    for i, tipo in enumerate(problema.tipos_otimizacao):
        val_a, val_b = a.objetivos[i], b.objetivos[i]
        
        if tipo == Otimizacao.MIN:
            if val_a > val_b:
                melhor_ou_igual = False
            elif val_a < val_b:
                melhor_pelo_menos_um = True
        elif tipo == Otimizacao.MAX:
            if val_a < val_b:
                melhor_ou_igual = False
            elif val_a > val_b:
                melhor_pelo_menos_um = True
                
    return melhor_ou_igual and melhor_pelo_menos_um

def non_dominated_sorting(pop: list[Solucao], problema: Problema):
    # reset informacos sobre dominacao
    for ind in pop:
        ind.domination_count = 0
        ind.dominates = []
        ind.front = 0

    # computar dominacoes
    for p in pop:
        for q in pop:
            if p is q:
                continue
            if domina(p, q, problema):
                p.dominates.append(q)
            elif domina(q, p, problema):
                p.domination_count += 1
    
    fronts = []
    
    first_front = [ind for ind in pop if ind.domination_count == 0]
    for ind in first_front:
        ind.front = 1
    fronts.append(first_front)
    
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in p.dominates:
                q.domination_count -= 1
                if q.domination_count == 0:
                    q.front = i + 2
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
        
        
        # ultima frente vai estar vazia, remover ela
        if fronts and not fronts[-1]:
            fronts.pop()
        return fronts

def crowding_distance_sorting(front: list[Solucao], problema: Problema):
    if not front:
        return front

    # inicializar
    for sol in front:
        sol.crowding_distance = 0.0
        
    if len(front) <= 2:
        for s in front:
            s.crowding_distance = float('inf')
        return front

    num_obj = len(problema.funcoes_objetivo)

    for m in range(num_obj):
        front_sorted = sorted(front, key=lambda s: s.objetivos[m])
        
        front_sorted[0].crowding_distance = float('inf')
        front_sorted[-1].crowding_distance = float('inf')

        obj_min = front_sorted[0].objetivos[m]
        obj_max = front_sorted[-1].objetivos[m]
        denom = obj_max - obj_min

        # evitar divisao por 0 em 
        if denom == 0:
            continue

        for i in range(1, len(front_sorted) - 1):
            prev_val = front_sorted[i - 1].objetivos[m]
            next_val = front_sorted[i + 1].objetivos[m]
            # normalized distance
            front_sorted[i].crowding_distance += (next_val - prev_val) / denom

    front.sort(key=lambda s: s.crowding_distance, reverse=True)
    return front

# repassar frentes completas para a proxima geracao e fazer crowding distance sorting
def gerar_prox_geracao(fronts: list[list[Solucao]], tam_pop_inicial, problema: Problema):
    tam_pop_atual = 0
    nova_pop = []
    for front in fronts:
        if tam_pop_atual + len(front) <= tam_pop_inicial:
            nova_pop.extend(front)
            tam_pop_atual += len(front)
        else:
            frente_com_cd = crowding_distance_sorting(front, problema)
            i = 0
            while tam_pop_atual < tam_pop_inicial and i < len(frente_com_cd):
                nova_pop.append(frente_com_cd[i])
                tam_pop_atual += 1
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

funcoes_objetivo = [("Autonomia", lambda x: autonomia_simples(x[0], x[1], x[2])), ("Aceleração", lambda x: tempo_aceleracao_simples(x[0], x[1], x[2]))]

limites_variaveis = [
    (10.0, 30.0),   # diametro_roda
    (50.0, 200.0),  # potencia_motor
    (100.0, 500.0)  # capacidade_bateria
]

# --- 3) tipos de otimizacao (usar Enum Otimizacao que você já tem) ---
tipos_otimizacao = [Otimizacao.MAX, Otimizacao.MIN]  # Autonomia (max), Aceleração (min)

# extrair nomes e funcoes para passar ao Problema
nomes_objetivos, funcoes = map(list, zip(*funcoes_objetivo))

problema_generico = Problema(funcoes_objetivo=funcoes, limites=limites_variaveis, tipos_otimizacao=tipos_otimizacao)

# ============================================================
# INÍCIO: BLOCO PAPER - Instância 3BC
# ============================================================
# Este bloco implementa uma instância prática do benchmark 3BC (bi-objetivo).
#
# Note: variáveis usam prefixo BASIN_ para evitar conflitos com outros nomes.
# ============================================================

# 1) definição dos nós (posições) e valores ms (valores alvo) e profundidade |s|
BASIN_XS = {
    's0': (0.0, 0.0),   # root
    's1': (0.25, 0.0),
    's2': (-0.25, 0.0),
    's3': (0.0, 0.25),  # escolhemos s3 como global (menor ms)
    's4': (0.0, -0.25),
}
BASIN_MS = {
    's0': -1.0,
    's1': -2.0,
    's2': -2.1,
    's3': -2.5,   # global (melhor)
    's4': -2.2,
}
# profundidade |s|
BASIN_DEPTH = {'s0': 0, 's1': 1, 's2': 1, 's3': 1, 's4': 1}

def basin_alpha_for_node(node):
    """Inclinação do poço: alpha = 4^{|s|} conforme exemplo prático."""
    return 4 ** BASIN_DEPTH[node]

# 2) projeção para o diamante L1: |x1| + |x2| <= 1
def basin_project_to_diamond(x1, x2):
    s = abs(x1) + abs(x2)
    if s <= 1.0:
        return x1, x2
    # escala preservando o sinal para ficar na fronteira do diamante
    factor = 1.0 / s
    return x1 * factor, x2 * factor

# 3) função primitiva f(x) = min_s ( m_s + alpha_s * L1(x, x_s) )
def basin_primitive_f(x1, x2):
    best = float('inf')
    for node in BASIN_XS:
        xs1, xs2 = BASIN_XS[node]
        dist = abs(x1 - xs1) + abs(x2 - xs2)  # L1
        val = BASIN_MS[node] + basin_alpha_for_node(node) * dist
        if val < best:
            best = val
    return best

# 4) função que calcula os dois objetivos rotacionados (retorna tupla)
def basin_threebc_objectives(decision_vector):
    """
    decision_vector: [t, x1, x2]
    retorna (o1, o2, o3)
    """
    t = float(decision_vector[0])
    x1 = float(decision_vector[1])
    x2 = float(decision_vector[2])

    # projetar x para o diamante (opcional: outra estratégia é penalizar)
    x1p, x2p = basin_project_to_diamond(x1, x2)
    fval = basin_primitive_f(x1p, x2p)

    o1 = (t + fval) / math.sqrt(2.0)
    o2 = (fval - t) / math.sqrt(2.0)
    
    # adicionando terceiro objetivo:
    o3 = fval + 2.0 * abs(t - 1.0)
    
    return o1, o2, o3

# 5) conversão para o formato do Problema: precisamos de callables,
#    cada um recebendo a lista 'variaveis' e retornando um escalar.
def basin_obj_o1(variaveis):
    return basin_threebc_objectives(variaveis)[0]

def basin_obj_o2(variaveis):
    return basin_threebc_objectives(variaveis)[1]

def basin_obj_o3(variaveis):
    return basin_threebc_objectives(variaveis)[2]

# 6) limites das variáveis: [t, x1, x2]
BASIN_LIMITES = [
    (0.0, 2.0),   # t
    (-1.0, 1.0),  # x1
    (-1.0, 1.0),  # x2
]

# 7) tipos de otimização: minimizar ambos (ajuste se necessário)
BASIN_TIPOS_OTIMIZACAO = [Otimizacao.MIN, Otimizacao.MIN, Otimizacao.MIN]

# 8) nomes das funções (opcional, para documentação/relatórios)
BASIN_NOMES_OBJETIVOS = ["O1_ProgressoRotacionado", "O2_ValorRotacionado", "O3"]

# 9) função utilitária para descobrir a qual basin uma posição x pertence (por distancia L1 mínima)
def basin_of(x1, x2):
    best_node = None
    best_dist = float('inf')
    for node in BASIN_XS:
        xs1, xs2 = BASIN_XS[node]
        d = abs(x1 - xs1) + abs(x2 - xs2)
        if d < best_dist:
            best_dist = d
            best_node = node
    return best_node

# 10) factory function: cria e retorna uma instância Problema pronto para uso
def create_3bc_problema(ProblemaClass):
    """
    Recebe a classe Problema e devolve uma instancia configurada.
    """
    funcoes = [basin_obj_o1, basin_obj_o2, basin_obj_o3]
    problema = ProblemaClass(funcoes_objetivo=funcoes, limites=BASIN_LIMITES, tipos_otimizacao=BASIN_TIPOS_OTIMIZACAO)
    # anexar metadados úteis (não obrigatório mas prático)
    problema.nomes_objetivos = BASIN_NOMES_OBJETIVOS
    # global_node: nó com menor ms (mais negativo) = global optimum
    global_node = min(BASIN_MS.keys(), key=lambda k: BASIN_MS[k])
    problema._3bc_metadata = {
        'xs': BASIN_XS,
        'ms': BASIN_MS,
        'depth': BASIN_DEPTH,
        'global_node': global_node
    }
    return problema

# criar a instância pronta para uso (variável: problema_3bc)
problema_3bc = create_3bc_problema(Problema)

# ============================================================
# FIM: BLOCO PAPER - Instância 3BC
# ============================================================

def calcular_estatisticas(pop: list[Solucao], nomes_objetivos=None):
    """
    Calcula estatísticas (média, desvio padrão, min, max) para cada objetivo
    presente em Solucao.objetivos.

    Argumentos:
    - pop: lista de Solucao
    - nomes_objetivos: lista de strings com os nomes dos objetivos (opcional).

    Retorno:
    - Se houver exatamente 2 objetivos, retorna um dicionário com chaves
        baseadas nos nomes passados em `nomes_objetivos` (quando fornecidos),
        e mantém também as chaves legadas 'media_autonomia', 'desvio_autonomia',
        'media_aceleracao', 'desvio_aceleracao' para compatibilidade com o
        restante do código.
    - Para qualquer outro número de objetivos, adiciona chaves genéricas
        'media_obj_i', 'desvio_obj_i', 'min_obj_i', 'max_obj_i'.
    Mensagens/erros em português.
    """
    if not pop:
        raise ValueError("População vazia — não há estatísticas para calcular.")

    # Extrair matriz (N_individuos x N_objetivos)
    try:
        matriz = np.array([ind.objetivos for ind in pop], dtype=float)
    except Exception as e:
        raise ValueError("Erro ao construir matriz de objetivos: verifique 'ind.objetivos'.") from e

    # número de indivíduos e objetivos
    n_ind, n_obj = matriz.shape

    stats = {}

    # calcular por objetivo
    medias = matriz.mean(axis=0)
    # desvio padrão: ddof=1 quando possível (mais apropriado para amostras)
    if n_ind > 1:
        desvios = matriz.std(axis=0, ddof=1)
    else:
        # com 1 indivíduo, std = 0 (evita ddof > n-1)
        desvios = np.zeros(n_obj, dtype=float)

    mins = matriz.min(axis=0)
    maxs = matriz.max(axis=0)

    # preencher chaves genéricas
    for j in range(n_obj):
        stats[f"media_obj_{j}"] = float(medias[j])
        stats[f"desvio_obj_{j}"] = float(desvios[j])
        stats[f"min_obj_{j}"] = float(mins[j])
        stats[f"max_obj_{j}"] = float(maxs[j])

    # Compatibilidade com o código existente (se houver exatamente 2 objetivos)
    if n_obj == 2:
        # decidir nomes a partir do parâmetro fornecido, com fallback
        if nomes_objetivos and len(nomes_objetivos) >= 2:
            nome0 = nomes_objetivos[0]
            nome1 = nomes_objetivos[1]
        else:
            nome0 = "autonomia"
            nome1 = "aceleracao"

        # chaves baseadas nos nomes fornecidos
        stats[f"media_{nome0}"] = float(medias[0])
        stats[f"desvio_{nome0}"] = float(desvios[0])
        stats[f"media_{nome1}"] = float(medias[1])
        stats[f"desvio_{nome1}"] = float(desvios[1])

        stats[f"max_{nome0}"] = float(maxs[0])
        stats[f"min_{nome0}"] = float(mins[0])
        stats[f"max_{nome1}"] = float(maxs[1])
        stats[f"min_{nome1}"] = float(mins[1])

        # manter chaves legadas para compatibilidade com o restante do código
        stats["media_autonomia"] = float(medias[0])
        stats["desvio_autonomia"] = float(desvios[0])
        stats["media_aceleracao"] = float(medias[1])
        stats["desvio_aceleracao"] = float(desvios[1])

        stats["max_autonomia"] = float(maxs[0])
        stats["min_autonomia"] = float(mins[0])
        stats["max_aceleracao"] = float(maxs[1])
        stats["min_aceleracao"] = float(mins[1])

    else:
        # Aviso opcional (não lançar erro) — o restante do código pode ignorar isso.
        # Se quiser tratar esse aviso como exceção, substitua print por raise.
        print(f"Aviso: {n_obj} objetivos detectados. Foram calculadas estatísticas genéricas 'media_obj_i', etc.")

    return stats

def main(
    NUM_INDIVIDUOS=750,
    GERACOES=25,
    CHANCE_CROSSOVER=1.0,
    CHANCE_MUTACAO=0.05,
    SEED=42,
    problema=problema_3bc
):
    # Configurações iniciais
    random.seed(SEED)
    np.random.seed(SEED)

    # --- 5) criar populacao inicial (usando Problema.gerar_individuo + Solucao genérico) ---
    populacao_inicial = [Solucao(problema.gerar_individuo(), problema) for _ in range(NUM_INDIVIDUOS)]

    # imprimir indivíduos iniciais (mantendo estilo do seu código; se quiser, comente)
    for i, ind in enumerate(populacao_inicial):
        # exibir os genes (x) em vez de atributos antigos
        d, p, c = ind.x
        print(f"Indivíduo {i+1}: diametro_roda={d:.2f}, potencia_motor={p:.2f}, capacidade_bateria={c:.2f}")

    populacao = populacao_inicial[:]

    # resolver nomes dos objetivos a partir do problema (fallbacks disponíveis)
    if hasattr(problema, 'nomes_objetivos') and problema.nomes_objetivos:
        objective_names = list(problema.nomes_objetivos)
    elif 'nomes_objetivos' in globals() and nomes_objetivos:
        objective_names = list(nomes_objetivos)
    else:
        objective_names = list(BASIN_NOMES_OBJETIVOS)

    # salvar figura da população inicial (diretório: img/seed{SEED}/popOriginal)
    folder_pop_original = f"./img/seed{SEED}"
    os.makedirs(folder_pop_original, exist_ok=True)
    # só plotar se houver exatamente 2 objetivos (compatível com o restante do código)
    if len(problema.funcoes_objetivo) == 2:
        xs_init = [ind.objetivos[0] for ind in populacao_inicial]
        ys_init = [ind.objetivos[1] for ind in populacao_inicial]
        plt.figure(figsize=(8,6))
        plt.scatter(xs_init, ys_init, label="População inicial")
        plt.xlabel(objective_names[0])
        plt.ylabel(objective_names[1])
        plt.title("População inicial")
        plt.grid(True)
        plt.savefig(f"{folder_pop_original}/pop_original_{NUM_INDIVIDUOS}-{SEED}.png")
        plt.close()

    # históricos por objetivo (para plot de médias/desvios)
    historico_medias = {nome: [] for nome in objective_names}
    historico_desvios = {nome: [] for nome in objective_names}

    # arquivo de stats (mantendo seu formato)
    os.makedirs("./stats", exist_ok=True)
    arquivo_texto = open(f"./stats/stats{SEED}.txt", "a")
    arquivo_texto.write(f"{NUM_INDIVIDUOS} individuos, {GERACOES} geracoes {CHANCE_CROSSOVER} chance de crossover e {CHANCE_MUTACAO} chance de mutacao\n")

    start = time.process_time()

    # --- helper local: torneio binário baseado em front e crowding_distance (igual ao que você já usa) ---
    def torneio_binario(pop):
        a, b = random.sample(pop, 2)
        if a.front == b.front:
            return a if a.crowding_distance > b.crowding_distance else b
        return a if a.front < b.front else b

    # --- loop principal ---
    for g in range(GERACOES):
        if g != 0:
            # seleção e reprodução simples: gerar filhos do tamanho da população atual
            filhos = []
            while len(filhos) < len(populacao):
                p1 = torneio_binario(populacao)
                p2 = torneio_binario(populacao)
                # crossover -> usa a assinatura que existe no seu código
                if random.random() <= CHANCE_CROSSOVER:
                    filho = crossover(p1, p2, problema, CHANCE_MUTACAO)
                else:
                    filho = copy.deepcopy(random.choice([p1, p2]))
                filhos.append(filho)
            # combinar pais+filhos como na NSGA-II
            populacao = populacao + filhos

        # reset valores de dominacao / crowding antes de sortear
        resetar_valores(populacao)

        # non-dominated sorting usando sua função (passando problema)
        frentes = non_dominated_sorting(populacao, problema)

        # gerar proxima geracao com suas funcoes (usa crowding internamente)
        nova_pop = gerar_prox_geracao(frentes, NUM_INDIVIDUOS, problema)

        populacao = nova_pop[:]

        print(f"\nFIM DA GERAÇÃO {g+1}\n")

        # calcular estatisticas usando a sua função genérica (ela retorna chaves compatíveis para 2 objetivos)
        # passar nomes dos objetivos a partir do problema em uso
        stats = calcular_estatisticas(populacao, problema.nomes_objetivos if hasattr(problema, 'nomes_objetivos') else None)

        # escrever no arquivo: preferir nomes vindos do problema, com fallback para chaves legadas
        if len(objective_names) >= 2 and f"media_{objective_names[0]}" in stats:
            # usar chaves baseadas nos nomes do problema
            arquivo_texto.write(f"Geração {g+1}: {objective_names[0]} média = {stats[f'media_{objective_names[0]}']:.4f} ± {stats[f'desvio_{objective_names[0]}']:.4f}\n")
            arquivo_texto.write(f"Geração {g+1}: {objective_names[1]} média = {stats[f'media_{objective_names[1]}']:.4f} ± {stats[f'desvio_{objective_names[1]}']:.4f}\n")
            arquivo_texto.write(f"{objective_names[0]}: {stats[f'min_{objective_names[0]}']} - {stats[f'max_{objective_names[0]}']}\n")
            arquivo_texto.write(f"{objective_names[1]}: {stats[f'min_{objective_names[1]}']} - {stats[f'max_{objective_names[1]}']}\n\n")
        elif f"media_autonomia" in stats:
            # fallback para chaves legadas
            arquivo_texto.write(f"Geração {g+1}: Autonomia média = {stats['media_autonomia']:.4f} ± {stats['desvio_autonomia']:.4f}\n")
            arquivo_texto.write(f"Geração {g+1}: Aceleração média = {stats['media_aceleracao']:.4f} ± {stats['desvio_aceleracao']:.4f}\n")
            arquivo_texto.write(f"Autonomia: {stats['min_autonomia']} - {stats['max_autonomia']}\n")
            arquivo_texto.write(f"Tempo de aceleracao: {stats['min_aceleracao']} - {stats['max_aceleracao']}\n\n")
        else:
            # populacao com != 2 objetivos -> escrever estatisticas genéricas por objetivo
            arquivo_texto.write(f"Geração {g+1}: Estatísticas por objetivo:\n")
            n_obj = len(populacao[0].objetivos)
            for j in range(n_obj):
                arquivo_texto.write(f"  Objetivo {j}: média = {stats[f'media_obj_{j}']:.4f} ± {stats[f'desvio_obj_{j}']:.4f} (min {stats[f'min_obj_{j}']}, max {stats[f'max_obj_{j}']})\n")
            arquivo_texto.write("\n")

        # atualizar históricos por objetivo (usar chaves baseadas nos nomes do problema)
        if len(objective_names) >= 2:
            key0 = f"media_{objective_names[0]}"
            key0_std = f"desvio_{objective_names[0]}"
            key1 = f"media_{objective_names[1]}"
            key1_std = f"desvio_{objective_names[1]}"

            if key0 in stats and key1 in stats:
                historico_medias[objective_names[0]].append(stats[key0])
                historico_desvios[objective_names[0]].append(stats[key0_std])
                historico_medias[objective_names[1]].append(stats[key1])
                historico_desvios[objective_names[1]].append(stats[key1_std])
            elif "media_autonomia" in stats:
                # fallback para chaves legadas
                historico_medias[objective_names[0]].append(stats["media_autonomia"])
                historico_desvios[objective_names[0]].append(stats["desvio_autonomia"])
                historico_medias[objective_names[1]].append(stats["media_aceleracao"])
                historico_desvios[objective_names[1]].append(stats["desvio_aceleracao"])
            else:
                # última opção: usar chaves genéricas por índice
                n_obj = len(populacao[0].objetivos)
                for j in range(n_obj):
                    nome = objective_names[j] if j < len(objective_names) else f"obj_{j}"
                    historico_medias[nome].append(stats[f"media_obj_{j}"])
                    historico_desvios[nome].append(stats[f"desvio_obj_{j}"])
        else:
            # não há nomes suficientes: usar chaves genéricas
            n_obj = len(populacao[0].objetivos)
            for j in range(n_obj):
                nome = objective_names[j] if j < len(objective_names) else f"obj_{j}"
                historico_medias[nome].append(stats[f"media_obj_{j}"])
                historico_desvios[nome].append(stats[f"desvio_obj_{j}"])

    end = time.process_time()

    print(f"CPU time: {end - start:.6f} seconds")
    arquivo_texto.write(f"CPU time: {end - start:.6f} seconds\n\n")
    arquivo_texto.close()

    # --- criar pastas para salvar figuras ---
    folder_img = f"./img/seed{SEED}"
    os.makedirs(folder_img, exist_ok=True)
    folder_stats = f"./stats/seed{SEED}"
    os.makedirs(folder_stats, exist_ok=True)

    # --- gerar gráfico da população final SOMENTE se houver 2 objetivos ---
    if len(funcoes) == 2:
        xs = [ind.objetivos[0] for ind in populacao]
        ys = [ind.objetivos[1] for ind in populacao]
        plt.figure(figsize=(8,6))
        plt.scatter(xs, ys, label="Indivíduos")
        plt.xlabel(objective_names[0])
        plt.ylabel(objective_names[1])
        plt.title("População final: " + objective_names[0] + " vs " + objective_names[1])
        plt.grid(True)
        plt.savefig(f"{folder_img}/pop_{NUM_INDIVIDUOS}-{GERACOES}-{CHANCE_CROSSOVER}-{CHANCE_MUTACAO}.png")
        plt.close()
    else:
        print("Mais de 2 objetivos — gráficos de indivíduos desativados.")

    # --- gerar gráficos de médias e desvios por objetivo (sempre) ---
    for nome in objective_names:
        medias = historico_medias[nome]
        desvios = historico_desvios[nome]
        plt.figure()
        plt.plot(range(1, len(medias)+1), medias, label=f"{nome} média")
        lower = np.array(medias) - np.array(desvios)
        upper = np.array(medias) + np.array(desvios)
        plt.fill_between(range(1, len(medias)+1), lower, upper, alpha=0.2)
        plt.xlabel("Geração")
        plt.ylabel(nome)
        plt.title(f"Evolução da média e desvio padrão de {nome}")
        plt.legend()
        plt.savefig(f"{folder_stats}/{nome}_{NUM_INDIVIDUOS}-{GERACOES}-{CHANCE_CROSSOVER}-{CHANCE_MUTACAO}.png")
        plt.close()

    return (end - start)


if __name__ == '__main__':
    main()