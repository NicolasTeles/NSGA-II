import random
import matplotlib.pyplot as plt
import time
import copy
import numpy as np
import os
from models import Problema, Otimizacao
import pandas as pd
import json

class Solucao:
    def __init__(self, x, problema: Problema):
        # Trava de segurança: garante que x respeite os limites antes de avaliar
        if hasattr(problema, 'aplicar_limites'):
            x = problema.aplicar_limites(x)
        self.x = x
        self.objetivos = problema.avaliar(self.x)
        self.calorias_totais = problema.get_calorias(self.x) if hasattr(problema, 'get_calorias') else None

        self.domination_count = 0
        self.dominates = []
        self.front = 0
        self.crowding_distance = -1

def mutacao_polinomial(individuo, limites, chance_mutacao, distribution_index: int = 20):
    from dieta import LIMITES_ACEITABILIDADE
    
    for i, (min_lim, max_lim) in enumerate(limites):
        if 'valor_fixo' in LIMITES_ACEITABILIDADE[i]:
            continue
        
        if random.random() < chance_mutacao:
            u = random.random()
            mutation_strenght = None
            
            if u < 0.5:
                mutation_strenght = (2*u) ** ( 1/(distribution_index+1) ) - 1
            else:
                mutation_strenght = 1 - ( 2*(1-u) ) ** ( 1/(distribution_index+1) )
            
            novo_valor = individuo[i] + mutation_strenght * (max_lim - min_lim)
            # Clamp estrito dentro dos limites [min_lim, max_lim]
            individuo[i] = min(max(novo_valor, min_lim), max_lim)

def SBX(parent1: Solucao, parent2: Solucao, limites: list, crossover_index: int = 20) -> tuple:
    from dieta import LIMITES_ACEITABILIDADE
    spread_factor = None
    
    filho0 = []
    filho1 = []
    
    for i, (gene_p1, gene_p2) in enumerate(zip(parent1.x, parent2.x)):
        if 'valor_fixo' in LIMITES_ACEITABILIDADE[i]:
            filho0.append(gene_p1)
            filho1.append(gene_p2)
            continue
        
        min_lim, max_lim = limites[i]
        u = random.random()
        if u <= 0.5:
            spread_factor = (2*u) ** ( 1/(crossover_index+1) )
        else:
            spread_factor = ( 1/(2*(1-u)) ) ** ( 1/(crossover_index+1) )
            
        gene_f0 = 0.5 * ((1 + spread_factor)*gene_p1 + (1 - spread_factor)*gene_p2)
        gene_f1 = 0.5 * ((1 - spread_factor)*gene_p1 + (1 + spread_factor)*gene_p2)
        
        # Clamp estrito dentro dos limites [min_lim, max_lim]
        gene_f0 = min(max(gene_f0, min_lim), max_lim)
        gene_f1 = min(max(gene_f1, min_lim), max_lim)
        
        filho0.append(gene_f0)
        filho1.append(gene_f1)
        
    return filho0, filho1

def blx_alpha_beta(parent1: Solucao, parent2: Solucao, limites: list = None) -> list:
    from dieta import LIMITES_ACEITABILIDADE
    filho = []
    alpha = 0.75
    beta = 0.25
    
    for i, (gene_p1, gene_p2) in enumerate(zip(parent1.x, parent2.x)):
        if 'valor_fixo' in LIMITES_ACEITABILIDADE[i]:
            filho.append(gene_p1)
            continue
            
        dist = abs(gene_p1 - gene_p2)
        min_gene, max_gene = sorted((gene_p1, gene_p2))
        
        lower = min_gene - alpha * dist
        upper = max_gene + beta * dist
        valor = random.uniform(lower, upper)
        
        if limites is not None:
            min_lim, max_lim = limites[i]
            valor = min(max(valor, min_lim), max_lim)
            
        filho.append(valor)
    
    return filho

def crossover(p1, p2, problema, chance_mutacao, crossover_index=20, distribution_index=20):
    x, y = SBX(p1, p2, problema.limites, crossover_index)
    mutacao_polinomial(x, problema.limites, chance_mutacao, distribution_index)
    mutacao_polinomial(y, problema.limites, chance_mutacao, distribution_index)
    return [Solucao(x, problema), Solucao(y, problema)]

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
    for ind in pop:
        ind.domination_count = 0
        ind.dominates = []
        ind.front = 0

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
        
        if not next_front:
            break
        
        fronts.append(next_front)
        i += 1
        
    return fronts

def crowding_distance_sorting(front: list[Solucao], problema: Problema):
    if not front:
        return front

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

        if denom == 0:
            continue

        for i in range(1, len(front_sorted) - 1):
            prev_val = front_sorted[i - 1].objetivos[m]
            next_val = front_sorted[i + 1].objetivos[m]
            front_sorted[i].crowding_distance += (next_val - prev_val) / denom

    front.sort(key=lambda s: s.crowding_distance, reverse=True)
    return front

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

def resetar_valores(pop: list[Solucao]):
    for ind in pop:
        ind.domination_count = 0
        ind.dominates = []
        ind.front = 0
        ind.crowding_distance = -1

from dieta import problema_dieta, construir_dieta

def calcular_estatisticas(pop: list[Solucao], nomes_objetivos=None):
    if not pop:
        raise ValueError("População vazia — não há estatísticas para calcular.")

    try:
        matriz = np.array([ind.objetivos for ind in pop], dtype=float)
    except Exception as e:
        raise ValueError("Erro ao construir matriz de objetivos: verifique 'ind.objetivos'.") from e

    n_ind, n_obj = matriz.shape
    stats = {}

    medias = matriz.mean(axis=0)
    if n_ind > 1:
        desvios = matriz.std(axis=0, ddof=1)
    else:
        desvios = np.zeros(n_obj, dtype=float)

    mins = matriz.min(axis=0)
    maxs = matriz.max(axis=0)

    for j in range(n_obj):
        stats[f"media_obj_{j}"] = float(medias[j])
        stats[f"desvio_obj_{j}"] = float(desvios[j])
        stats[f"min_obj_{j}"] = float(mins[j])
        stats[f"max_obj_{j}"] = float(maxs[j])

    if n_obj == 2:
        if nomes_objetivos and len(nomes_objetivos) >= 2:
            nome0 = nomes_objetivos[0]
            nome1 = nomes_objetivos[1]
        else:
            nome0 = "autonomia"
            nome1 = "aceleracao"

        stats[f"media_{nome0}"] = float(medias[0])
        stats[f"desvio_{nome0}"] = float(desvios[0])
        stats[f"media_{nome1}"] = float(medias[1])
        stats[f"desvio_{nome1}"] = float(desvios[1])
        stats[f"max_{nome0}"] = float(maxs[0])
        stats[f"min_{nome0}"] = float(mins[0])
        stats[f"max_{nome1}"] = float(maxs[1])
        stats[f"min_{nome1}"] = float(mins[1])
    else:
        print(f"Aviso: {n_obj} objetivos detectados. Foram calculadas estatísticas genéricas 'media_obj_i', etc.")

    return stats

import pandas as pd
import json

def exportar_pareto_csv(pareto, caminho_csv, uf, faixa_renda, execucao, conjunto,
                         num_individuos, geracoes, chance_crossover, chance_mutacao,
                         crossover_index=20, distribution_index=20,
                         tempo_execucao=None, alvo_calorico=None, alvo_custo=None):
    rows = []

    for i, individuo in enumerate(pareto):
        dieta = construir_dieta(individuo.x)

        row = {
            'UF': uf,
            'Faixa de Renda': faixa_renda,
            'Alvo Calorico': alvo_calorico,
            'Custo Alvo': alvo_custo,
            'Conjunto': conjunto,
            'Execucao': execucao,
            'Tempo de Execucao (s)': tempo_execucao,
            'Numero de Individuos': num_individuos,
            'Numero de Geracoes': geracoes,
            'Taxa de Cruzamento': chance_crossover,
            'Taxa de Mutacao': chance_mutacao,
            'Crossover Index': crossover_index,
            'Distribution Index': distribution_index,
            'Pareto ID': i,
            'Crowding Distance': individuo.crowding_distance,
            'fo_phdi': individuo.objetivos[0],
            'fo_aceitabilidade': individuo.objetivos[1],
            'Calorias Totais': dieta.calorias_totais,
            'Custo Total': dieta.custo_total,
            'Pontuacao PHDI': dieta.pontuacao_phdi,
            'Pontuacao Aceitabilidade': dieta.pontuacao_aceitabilidade,
        }

        for grupo in dieta.grupos_alimentares:
            nome = grupo.nome.replace(' ', '_').replace('(', '').replace(')', '').lower()
            row[f'{nome}_kcal'] = grupo.calorias
            row[f'{nome}_perc'] = grupo.percentual_calorico
            row[f'{nome}_phdi'] = grupo.pontuacao_phdi
            row[f'{nome}_aceitabilidade'] = grupo.pontuacao_aceitabilidade

        rows.append(row)

    df = pd.DataFrame(rows)
    if not os.path.exists(caminho_csv):
        df.to_csv(caminho_csv, index=False)
    else:
        df.to_csv(caminho_csv, mode='a', index=False, header=False)


def exportar_pareto_evolucao_csv(historico_pareto, caminho_csv, uf, faixa_renda, execucao, conjunto,
                                  num_individuos, geracoes, chance_crossover, chance_mutacao,
                                  crossover_index=20, distribution_index=20,
                                  tempo_execucao=None, alvo_calorico=None, alvo_custo=None):
    rows = []

    for g, pareto in enumerate(historico_pareto):
        for i, individuo in enumerate(pareto):
            dieta = construir_dieta(individuo.x)

            row = {
                'UF': uf,
                'Faixa de Renda': faixa_renda,
                'Alvo Calorico': alvo_calorico,
                'Custo Alvo': alvo_custo,
                'Conjunto': conjunto,
                'Execucao': execucao,
                'Tempo de Execucao (s)': tempo_execucao,
                'Geracao': g + 1,
                'Numero de Individuos': num_individuos,
                'Numero de Geracoes': geracoes,
                'Taxa de Cruzamento': chance_crossover,
                'Taxa de Mutacao': chance_mutacao,
                'Crossover Index': crossover_index,
                'Distribution Index': distribution_index,
                'Pareto ID': i,
                'Crowding Distance': individuo.crowding_distance,
                'fo_phdi': individuo.objetivos[0],
                'fo_aceitabilidade': individuo.objetivos[1],
                'Calorias Totais': dieta.calorias_totais,
                'Custo Total': dieta.custo_total,
                'Pontuacao PHDI': dieta.pontuacao_phdi,
                'Pontuacao Aceitabilidade': dieta.pontuacao_aceitabilidade,
            }

            for grupo in dieta.grupos_alimentares:
                nome = grupo.nome.replace(' ', '_').replace('(', '').replace(')', '').lower()
                row[f'{nome}_kcal'] = grupo.calorias
                row[f'{nome}_perc'] = grupo.percentual_calorico
                row[f'{nome}_phdi'] = grupo.pontuacao_phdi
                row[f'{nome}_aceitabilidade'] = grupo.pontuacao_aceitabilidade

            rows.append(row)

    df = pd.DataFrame(rows)
    if not os.path.exists(caminho_csv):
        df.to_csv(caminho_csv, index=False)
    else:
        df.to_csv(caminho_csv, mode='a', index=False, header=False)

def main(
        NUM_INDIVIDUOS=100,
        GERACOES=300,
        CHANCE_CROSSOVER=1.0,
        CHANCE_MUTACAO=0.1,
        SEED=42,
        CROSSOVER_INDEX=5,
        DISTRIBUTION_INDEX=20,
        execucao=1,
        conjunto=1,
        problema=problema_dieta
    ):
    random.seed(SEED)
    np.random.seed(SEED)

    populacao_inicial = [Solucao(problema.gerar_individuo(), problema) for _ in range(NUM_INDIVIDUOS)]
    populacao = populacao_inicial[:]
    
    print([ind.calorias_totais for ind in populacao_inicial])
    print([ind.objetivos for ind in populacao_inicial])

    if hasattr(problema, 'nomes_objetivos') and problema.nomes_objetivos:
        objective_names = list(problema.nomes_objetivos)
    else:
        objective_names = [f"obj_{i}" for i in range(len(populacao_inicial[0].objetivos))]

    # --- pastas ---
    folder_pop_original = f"./img/seed{SEED}"
    folder_img = f"./img/seed{SEED}"
    folder_stats = f"./stats/seed{SEED}"
    os.makedirs(folder_pop_original, exist_ok=True)
    os.makedirs(folder_img, exist_ok=True)
    os.makedirs(folder_stats, exist_ok=True)
    os.makedirs("./stats", exist_ok=True)

    # --- plot população inicial ---
    if len(problema.funcoes_objetivo) == 2:
        xs_init = [ind.objetivos[0] for ind in populacao_inicial]
        ys_init = [ind.objetivos[1] for ind in populacao_inicial]
        plt.figure(figsize=(8, 6))
        plt.scatter(xs_init, ys_init, label="População inicial")
        plt.xlabel(objective_names[0])
        plt.ylabel(objective_names[1])
        plt.title("População inicial")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{folder_pop_original}/pop_original_{NUM_INDIVIDUOS}-{SEED}.png")
        plt.close()

    historico_pareto = []
    historico_medias = {nome: [] for nome in objective_names}
    historico_desvios = {nome: [] for nome in objective_names}

    arquivo_texto = open(f"./stats/stats{SEED}.txt", "a")
    arquivo_texto.write(f"{NUM_INDIVIDUOS} individuos, {GERACOES} geracoes, {CHANCE_CROSSOVER} chance de crossover, {CHANCE_MUTACAO} chance de mutacao\n\n")

    start = time.process_time()

    def torneio_binario(pop):
        a, b = random.sample(pop, 2)
        if a.front == b.front:
            return a if a.crowding_distance > b.crowding_distance else b
        return a if a.front < b.front else b

    # --- loop principal ---
    for g in range(GERACOES):
        if g != 0:
            filhos = []
            while len(filhos) < len(populacao):
                p1 = torneio_binario(populacao)
                p2 = torneio_binario(populacao)
                
                if random.random() <= CHANCE_CROSSOVER:
                    # Gera descendentes por crossover SBX
                    x, y = SBX(
                        p1, p2, problema.limites, CROSSOVER_INDEX
                    )
                else:
                    # Sem crossover: cada descendente copia um dos pais
                    x = copy.deepcopy(p1.x)
                    y = copy.deepcopy(p2.x)

                # Mutação independente da ocorrência de crossover
                mutacao_polinomial(
                    x, problema.limites, CHANCE_MUTACAO, DISTRIBUTION_INDEX
                )
                mutacao_polinomial(
                    y, problema.limites, CHANCE_MUTACAO, DISTRIBUTION_INDEX
                )

                # Recalcula os objetivos após a mutação
                novos_filhos = [
                    Solucao(x, problema),
                    Solucao(y, problema)
                ]

                filhos.extend(novos_filhos)

            populacao = populacao + filhos

        resetar_valores(populacao)
        frentes = non_dominated_sorting(populacao, problema)
        nova_pop = gerar_prox_geracao(frentes, NUM_INDIVIDUOS, problema)
        populacao = nova_pop[:]

        print(f"\nFIM DA GERAÇÃO {g+1}\n")

        # --- estatísticas dos objetivos ---
        stats = calcular_estatisticas(populacao, problema.nomes_objetivos if hasattr(problema, 'nomes_objetivos') else None)

        arquivo_texto.write(f"=== Geração {g+1} ===\n")

        if len(objective_names) >= 2 and f"media_{objective_names[0]}" in stats:
            for nome in objective_names:
                arquivo_texto.write(f"  {nome}: média = {stats[f'media_{nome}']:.4f} ± {stats[f'desvio_{nome}']:.4f} (min {stats[f'min_{nome}']:.4f}, max {stats[f'max_{nome}']:.4f})\n")
        else:
            n_obj = len(populacao[0].objetivos)
            for j in range(n_obj):
                arquivo_texto.write(f"  Objetivo {j}: média = {stats[f'media_obj_{j}']:.4f} ± {stats[f'desvio_obj_{j}']:.4f} (min {stats[f'min_obj_{j}']:.4f}, max {stats[f'max_obj_{j}']:.4f})\n")

        # --- estatísticas de calorias ---
        if hasattr(problema, 'get_calorias'):
            calorias_pop = [ind.calorias_totais for ind in populacao if ind.calorias_totais is not None]
            media_cal = np.mean(calorias_pop)
            desvio_cal = np.std(calorias_pop, ddof=1)
            arquivo_texto.write(f"  Calorias totais: média = {media_cal:.2f} ± {desvio_cal:.2f}\n")

        # --- tamanho do Pareto front ---
        pareto = [ind for ind in populacao if ind.front == 1]
        arquivo_texto.write(f"  Pareto front: {len(pareto)} indivíduos\n\n")
        historico_pareto.append(pareto[:])

        # --- históricos ---
        for j, nome in enumerate(objective_names):
            key_media = f"media_{nome}"
            key_desvio = f"desvio_{nome}"
            if key_media in stats:
                historico_medias[nome].append(stats[key_media])
                historico_desvios[nome].append(stats[key_desvio])
            else:
                historico_medias[nome].append(stats[f"media_obj_{j}"])
                historico_desvios[nome].append(stats[f"desvio_obj_{j}"])

    end = time.process_time()

    print(f"CPU time: {end - start:.6f} seconds")
    arquivo_texto.write(f"CPU time: {end - start:.6f} seconds\n\n")
    arquivo_texto.close()

    # --- plot Pareto front final ---
    pareto = [ind for ind in populacao if ind.front == 1]
    print(f"Pareto front size: {len(pareto)}")
    
    from dieta import alvo_calorico, custo_alvo

    try:
        # caso receba uma uf e faixa de renda especifica
        from dieta import UF, FAIXA_RENDA
    except ImportError:
        # caso nao: media do país
        UF = 100
        FAIXA_RENDA = 100

    exportar_pareto_csv(
        pareto, f"./stats/pareto_final.csv",
        uf=UF, faixa_renda=FAIXA_RENDA, execucao=execucao, conjunto=conjunto,
        num_individuos=NUM_INDIVIDUOS, geracoes=GERACOES,
        chance_crossover=CHANCE_CROSSOVER, chance_mutacao=CHANCE_MUTACAO,
        crossover_index=CROSSOVER_INDEX, distribution_index=DISTRIBUTION_INDEX,
        tempo_execucao=end - start, alvo_calorico=alvo_calorico, alvo_custo=custo_alvo
    )

    exportar_pareto_evolucao_csv(
        historico_pareto, f"./stats/pareto_evolucao.csv",
        uf=UF, faixa_renda=FAIXA_RENDA, execucao=execucao, conjunto=conjunto,
        num_individuos=NUM_INDIVIDUOS, geracoes=GERACOES,
        chance_crossover=CHANCE_CROSSOVER, chance_mutacao=CHANCE_MUTACAO,
        crossover_index=CROSSOVER_INDEX, distribution_index=DISTRIBUTION_INDEX,
        tempo_execucao=end - start, alvo_calorico=alvo_calorico, alvo_custo=custo_alvo
    )

    if len(problema.funcoes_objetivo) == 2:
        xs_pareto = [ind.objetivos[0] for ind in pareto]
        ys_pareto = [ind.objetivos[1] for ind in pareto]
        plt.figure(figsize=(8, 6))
        plt.scatter(xs_pareto, ys_pareto, label="Pareto Front", color='red')
        plt.xlabel(objective_names[0])
        plt.ylabel(objective_names[1])
        plt.title(f"Pareto Front: {objective_names[0]} vs {objective_names[1]}")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{folder_img}/pareto_{NUM_INDIVIDUOS}-{GERACOES}-{CHANCE_CROSSOVER}-{CHANCE_MUTACAO}.png")
        plt.close()
    else:
        print("Mais de 2 objetivos — gráfico de Pareto desativado.")

    # --- gráficos de médias e desvios por objetivo ---
    for nome in objective_names:
        medias = historico_medias[nome]
        desvios = historico_desvios[nome]
        plt.figure()
        plt.plot(range(1, len(medias) + 1), medias, label=f"{nome} média")
        lower = np.array(medias) - np.array(desvios)
        upper = np.array(medias) + np.array(desvios)
        plt.fill_between(range(1, len(medias) + 1), lower, upper, alpha=0.2)
        plt.xlabel("Geração")
        plt.ylabel(nome)
        plt.title(f"Evolução da média e desvio padrão de {nome}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{folder_stats}/{nome}_{NUM_INDIVIDUOS}-{GERACOES}-{CHANCE_CROSSOVER}-{CHANCE_MUTACAO}.png")
        plt.close()

    return (end - start)


if __name__ == '__main__':
    main()