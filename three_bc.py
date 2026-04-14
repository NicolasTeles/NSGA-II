import math
from NSGA2 import Problema, Otimizacao  # adjust to your main file's name

# --- Nodes, target values and depth ---
BASIN_XS = {
    's0': (0.0, 0.0),
    's1': (0.25, 0.0),
    's2': (-0.25, 0.0),
    's3': (0.0, 0.25),
    's4': (0.0, -0.25),
}
BASIN_MS = {
    's0': -1.0,
    's1': -2.0,
    's2': -2.1,
    's3': -2.5,
    's4': -2.2,
}
BASIN_DEPTH = {'s0': 0, 's1': 1, 's2': 1, 's3': 1, 's4': 1}

def basin_alpha_for_node(node):
    return 4 ** BASIN_DEPTH[node]

def basin_project_to_diamond(x1, x2):
    s = abs(x1) + abs(x2)
    if s <= 1.0:
        return x1, x2
    factor = 1.0 / s
    return x1 * factor, x2 * factor

def basin_primitive_f(x1, x2):
    best = float('inf')
    for node in BASIN_XS:
        xs1, xs2 = BASIN_XS[node]
        dist = abs(x1 - xs1) + abs(x2 - xs2)
        val = BASIN_MS[node] + basin_alpha_for_node(node) * dist
        if val < best:
            best = val
    return best

def basin_threebc_objectives(decision_vector):
    t  = float(decision_vector[0])
    x1 = float(decision_vector[1])
    x2 = float(decision_vector[2])
    x1p, x2p = basin_project_to_diamond(x1, x2)
    fval = basin_primitive_f(x1p, x2p)
    o1 = (t + fval) / math.sqrt(2.0)
    o2 = (fval - t) / math.sqrt(2.0)
    o3 = fval + 2.0 * abs(t - 1.0)
    return o1, o2, o3

def basin_obj_o1(variaveis): return basin_threebc_objectives(variaveis)[0]
def basin_obj_o2(variaveis): return basin_threebc_objectives(variaveis)[1]
def basin_obj_o3(variaveis): return basin_threebc_objectives(variaveis)[2]

def basin_of(x1, x2):
    best_node, best_dist = None, float('inf')
    for node in BASIN_XS:
        xs1, xs2 = BASIN_XS[node]
        d = abs(x1 - xs1) + abs(x2 - xs2)
        if d < best_dist:
            best_dist = d
            best_node = node
    return best_node

BASIN_LIMITES = [(0.0, 2.0), (-1.0, 1.0), (-1.0, 1.0)]
BASIN_TIPOS_OTIMIZACAO = [Otimizacao.MIN, Otimizacao.MIN, Otimizacao.MIN]
BASIN_NOMES_OBJETIVOS = ["O1_ProgressoRotacionado", "O2_ValorRotacionado", "O3"]

def create_3bc_problema():
    funcoes = [basin_obj_o1, basin_obj_o2, basin_obj_o3]
    problema = Problema(
        funcoes_objetivo=funcoes,
        limites=BASIN_LIMITES,
        tipos_otimizacao=BASIN_TIPOS_OTIMIZACAO
    )
    problema.nomes_objetivos = BASIN_NOMES_OBJETIVOS
    global_node = min(BASIN_MS.keys(), key=lambda k: BASIN_MS[k])
    problema._3bc_metadata = {
        'xs': BASIN_XS, 'ms': BASIN_MS,
        'depth': BASIN_DEPTH, 'global_node': global_node
    }
    return problema

problema_3bc = create_3bc_problema()