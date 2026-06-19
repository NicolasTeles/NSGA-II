from models import Problema, Otimizacao
from typing import List, Optional, Dict
import pandas as pd

"""
Dicionário global com a definição completa dos grupos alimentares do PHDI.
"""
COMPONENTES_PHDI = [            
    {'grupo': 'Nuts and peanuts', 'limite': 11.6, 'tipo': 'adequacao'},
    {'grupo': 'Legumes', 'limite': 11.3, 'tipo': 'adequacao'},
    {'grupo': 'Fruits', 'limite': 5.0, 'tipo': 'adequacao'},
    {'grupo': 'Vegetables', 'limite': 3.1, 'tipo': 'adequacao'},
    {'grupo': 'Whole cereals', 'limite': 32.4, 'tipo': 'adequacao'},

    {'grupo': 'Eggs', 'limite_inferior': 0.0, 'limite_otimo': 0.8, 'limite_superior': 1.5, 'tipo': 'otimo'},
    {'grupo': 'Fish and seafood', 'limite_inferior': 0.0, 'limite_otimo': 1.6, 'limite_superior': 5.7, 'tipo': 'otimo'},
    {'grupo': 'Tubers and potatoes', 'limite_inferior': 0.0, 'limite_otimo': 1.6, 'limite_superior': 3.1, 'tipo': 'otimo'},
    {'grupo': 'Dairy', 'limite_inferior': 0.0, 'limite_otimo': 6.1, 'limite_superior': 12.2, 'tipo': 'otimo'},
    {'grupo': 'Vegetable oils', 'limite_inferior': 0.0, 'limite_otimo': 16.5, 'limite_superior': 30.7, 'tipo': 'otimo'},

    {'grupo': 'DGV', 'limite_inferior': 0.0, 'limite_otimo': 29.5, 'limite_superior': 100.0, 'tipo': 'otimo'},
    {'grupo': 'REV', 'limite_inferior': 0.0, 'limite_otimo': 38.5, 'limite_superior': 100.0, 'tipo': 'otimo'},

    {'grupo': 'Red meat', 'limite': 2.4, 'tipo': 'moderacao'},
    {'grupo': 'Chicken and substitutes', 'limite': 5.0, 'tipo': 'moderacao'},
    {'grupo': 'Animal fats', 'limite': 1.4, 'tipo': 'moderacao'},
    {'grupo': 'Added sugars', 'limite': 4.8, 'tipo': 'moderacao'},
]

UF = 31
FAIXA_RENDA = 1
LIMITES_ACEITABILIDADE = []

def extrai_da_linha(LIMITES_ACEITABILIDADE, row):
    LIMITES_ACEITABILIDADE.extend([
        {
            "grupo": "Nuts and peanuts", 
            "minimo": row.p5_nozes_amendoim_kcal, 
            "p25": row.p25_nozes_amendoim_kcal, 
            "p75": row.p75_nozes_amendoim_kcal, 
            "maximo": row.p95_nozes_amendoim_kcal,
            "custo": row.preco_nozes_amendoim_por_kcal
        },
        {
            "grupo": "Legumes", 
            "minimo": row.p5_leguminosas_kcal, 
            "p25": row.p25_leguminosas_kcal, 
            "p75": row.p75_leguminosas_kcal, 
            "maximo": row.p95_leguminosas_kcal,
            "custo": row.preco_leguminosas_por_kcal
        },
        {
            "grupo": "Fruits", 
            "minimo": row.p5_frutas_kcal, 
            "p25": row.p25_frutas_kcal, 
            "p75": row.p75_frutas_kcal, 
            "maximo": row.p95_frutas_kcal,
            "custo": row.preco_frutas_por_kcal
        },
        {
            "grupo": "Vegetables", 
            "minimo": row.p5_veg_total_kcal, 
            "p25": row.p25_veg_total_kcal, 
            "p75": row.p75_veg_total_kcal, 
            "maximo": row.p95_veg_total_kcal,
            "custo": row.preco_veg_totais_por_kcal
        },
        {
            "grupo": "Whole cereals", 
            "minimo": row.p5_cereais_integrais_kcal, 
            "p25": row.p25_cereais_integrais_kcal, 
            "p75": row.p75_cereais_integrais_kcal, 
            "maximo": row.p95_cereais_integrais_kcal,
            "custo": row.preco_cereais_integrais_por_kcal
        },
        {
            "grupo": "Eggs", 
            "minimo": row.p5_ovos_kcal, 
            "p25": row.p25_ovos_kcal, 
            "p75": row.p75_ovos_kcal, 
            "maximo": row.p95_ovos_kcal,
            "custo": row.preco_ovos_por_kcal
        },
        {
            "grupo": "Fish and seafood", 
            "minimo": row.p5_peixes_kcal, 
            "p25": row.p25_peixes_kcal, 
            "p75": row.p75_peixes_kcal, 
            "maximo": row.p95_peixes_kcal,
            "custo": row.preco_peixes_por_kcal
        },
        {
            "grupo": "Tubers and potatoes", 
            "minimo": row.p5_batatas_tuber_kcal, 
            "p25": row.p25_batatas_tuber_kcal, 
            "p75": row.p75_batatas_tuber_kcal, 
            "maximo": row.p95_batatas_tuber_kcal,
            "custo": row.preco_batatas_tuber_por_kcal
        },
        {
            "grupo": "Dairy", 
            "minimo": row.p5_laticinios_kcal, 
            "p25": row.p25_laticinios_kcal, 
            "p75": row.p75_laticinios_kcal, 
            "maximo": row.p95_laticinios_kcal,
            "custo": row.preco_laticinios_por_kcal
        },
        {
            "grupo": "Vegetable oils", 
            "minimo": row.p5_oleos_insat_kcal, 
            "p25": row.p25_oleos_insat_kcal, 
            "p75": row.p75_oleos_insat_kcal, 
            "maximo": row.p95_oleos_insat_kcal,
            "custo": row.preco_oleos_insat_por_kcal
        },
        {
            "grupo": "DGV", 
            "minimo": row.p5_veg_verde_kcal, 
            "p25": row.p25_veg_verde_kcal, 
            "p75": row.p75_veg_verde_kcal, 
            "maximo": row.p95_veg_verde_kcal,
            "custo": row.preco_veg_verde_por_kcal
        },
        {
            "grupo": "REV", 
            "minimo": row.p5_veg_vermelho_kcal, 
            "p25": row.p25_veg_vermelho_kcal, 
            "p75": row.p75_veg_vermelho_kcal, 
            "maximo": row.p95_veg_vermelho_kcal,
            "custo": row.preco_veg_vermelho_por_kcal
        },
        {
            "grupo": "Red meat", 
            "minimo": row.p5_carne_vermelha_kcal, 
            "p25": row.p25_carne_vermelha_kcal, 
            "p75": row.p75_carne_vermelha_kcal, 
            "maximo": row.p95_carne_vermelha_kcal,
            "custo": row.preco_carne_vermelha_por_kcal
        },
        {
            "grupo": "Chicken and substitutes", 
            "minimo": row.p5_aves_frango_kcal, 
            "p25": row.p25_aves_frango_kcal, 
            "p75": row.p75_aves_frango_kcal, 
            "maximo": row.p95_aves_frango_kcal,
            "custo": row.preco_aves_frango_por_kcal
        },
        {
            "grupo": "Animal fats", 
            "minimo": row.p5_gord_animal_kcal, 
            "p25": row.p25_gord_animal_kcal, 
            "p75": row.p75_gord_animal_kcal, 
            "maximo": row.p95_gord_animal_kcal,
            "custo": row.preco_gord_animal_por_kcal
        },
        {
            "grupo": "Added sugars", 
            "minimo": row.p5_acucares_kcal, 
            "p25": row.p25_acucares_kcal, 
            "p75": row.p75_acucares_kcal, 
            "maximo": row.p95_acucares_kcal,
            "custo": row.preco_acucares_por_kcal
        }
    ])

df_INA = pd.read_csv('banco_INA_estratificado_ag.csv')

for row in df_INA.itertuples():
    if row.uf == UF and row.faixa_renda_per_capita == FAIXA_RENDA:
                alvo_calorico = row.kcal_consumo_phdi

                print(f'Alvo calorico: {alvo_calorico}')
                custo_alvo = row.preco_total_phdi
                print(f"Valor alvo: {custo_alvo}")

                extrai_da_linha(LIMITES_ACEITABILIDADE, row)

class GrupoAlimentar:
    def __init__(self, nome, valor_componente: Optional[float] = None, calorias: Optional[float] = None):
        self.nome = nome
        self.valor_componente = valor_componente
        self.calorias = calorias
        self.percentual_calorico = None
        self.pontuacao_fo = None
        self.pontuacao_phdi = None
        self.pontuacao_aceitabilidade = None
        self.penalizado = False
        self.penalizacao_aplicada = None

    def __str__(self):
        return (
            f'{self.nome} = Calorias: {self.calorias}, '
            f'Percentual Calórico: {self.percentual_calorico}, '
            f'PHDI: {self.pontuacao_phdi}, '
            f'Aceitabilidade: {self.pontuacao_aceitabilidade}, '
            f'FO: {self.pontuacao_fo}, '
            f'Penalizado?: {self.penalizado}, '
            f'Penalização aplicada: {self.penalizacao_aplicada}'
        )

    def copia(self):
        copia_grupo = GrupoAlimentar(self.nome)
        copia_grupo.valor_componente = self.valor_componente
        copia_grupo.calorias = self.calorias
        copia_grupo.percentual_calorico = self.percentual_calorico
        copia_grupo.pontuacao_fo = self.pontuacao_fo
        copia_grupo.pontuacao_phdi = self.pontuacao_phdi
        copia_grupo.pontuacao_aceitabilidade = self.pontuacao_aceitabilidade
        copia_grupo.penalizado = self.penalizado
        copia_grupo.penalizacao_aplicada = self.penalizacao_aplicada
        return copia_grupo

    def reset(self) -> None:
        self.calorias = None
        self.percentual_calorico = None
        self.pontuacao_phdi = None
        self.pontuacao_fo = None
        self.pontuacao_aceitabilidade = None
        self.penalizado = False
        self.penalizacao_aplicada = None

    def calculaPHDI(self):
        componente = next((c for c in COMPONENTES_PHDI if c['grupo'].lower() == self.nome.lower()), None)
        
        if componente is None:
            raise ValueError(f"Grupo '{self.nome}' não encontrado em COMPONENTES_PHDI.")
        
        valor = self.percentual_calorico

        match componente['tipo']:
            case 'adequacao':
                self.pontuacao_phdi = min(valor / componente['limite'] * 10, 10)

            case 'otimo':
                if valor <= componente['limite_otimo']:
                    self.pontuacao_phdi = min((valor / componente['limite_otimo']) * 10, 10)
                else:
                    faixa = componente['limite_superior'] - componente['limite_otimo']
                    excesso = valor - componente['limite_otimo']
                    self.pontuacao_phdi = min(max(10 - (excesso / faixa) * 10, 0), 10)

            case 'moderacao':
                self.pontuacao_phdi = min(max(10 - (valor / componente['limite']) * 10, 0), 10)
    
    def calculaAceitabilidadeCultural(self):
        componente = next((c for c in LIMITES_ACEITABILIDADE if c['grupo'].lower() == self.nome.lower()), None)

        if componente is None:
            raise ValueError(f"Grupo '{self.nome}' não encontrado em LIMITES_ACEITABILIDADE.")

        valor = self.percentual_calorico

        minimo = componente["minimo"]
        p25 = componente["p25"]
        p75 = componente["p75"]
        maximo = componente["maximo"]

        # Fora dos limites absolutos → 0
        if valor <= minimo or valor >= maximo:
            self.pontuacao_aceitabilidade = 0
            return

        # Região ideal → 10
        if p25 <= valor <= p75:
            self.pontuacao_aceitabilidade = 10
            return

        # Crescimento linear (min → p25)
        if minimo < valor < p25:
            self.pontuacao_aceitabilidade = (valor - minimo) / (p25 - minimo) * 10
            return

        # Decaimento linear (p75 → max)
        if p75 < valor < maximo:
            self.pontuacao_aceitabilidade = (maximo - valor) / (maximo - p75) * 10
            return

class Dieta:
    """
    Representa uma dieta completa com múltiplos grupos alimentares e suas métricas agregadas.
    
    Atributos:
        grupos_alimentares (List[GrupoAlimentar]): Lista de grupos que compõem a dieta.
        alvo_calorico (float, opcional): Meta calórica desejada para a dieta.
        custo_alvo (float, opcional): Meta de custo máximo desejado.
        calorias_totais (float, opcional): Soma das calorias de todos os grupos.
        custo_total (float): Custo total calculado da dieta.
        pontuacao_phdi (float, opcional): Pontuação total da dieta no índice PHDI.
        pontuacao_fo (float, opcional): Pontuação total no cálculo FO.
        pontuacao_aceitabilidade (float, opcional): Pontuação total de aceitabilidade cultural.
        penalizado (bool): Indica se a dieta como um todo foi penalizada.
        penalizacao_aplicada (float, opcional): Valor total da penalização aplicada.
        penalizacao_aceitabilidade (float): Valor total da penalização por Aceitabilidade Cultural.
    """

    def __init__(self, grupos_alimentares: List, alvo_calorico: Optional[float] = None, custo_alvo: Optional[float] = None) -> None:
        self.grupos_alimentares = grupos_alimentares
        self.alvo_calorico = alvo_calorico
        self.custo_alvo = custo_alvo
        self.calorias_totais = None
        self.custo_total = None
        self.pontuacao_phdi = None
        self.pontuacao_fo = None
        self.pontuacao_aceitabilidade = None
        self.penalizado = False
        self.penalizacao_aplicada = None
        self.penalizacao_aceitabilidade = None

    def calcular_calorias_totais(self) -> None:
        self.calorias_totais = 0
        for grupo in self.grupos_alimentares:
            if grupo.nome != 'DGV' and grupo.nome != 'REV':
                self.calorias_totais += grupo.calorias

    def calcular_percentual_calorico(self) -> None:
        if self.calorias_totais is not None:
            for grupo in self.grupos_alimentares:
                if grupo.nome != 'DGV' and grupo.nome != 'REV':
                    grupo.percentual_calorico = grupo.calorias / self.calorias_totais * 100
                else:
                    grupo.percentual_calorico = grupo.calorias / self.grupos_alimentares[3].calorias * 100

    def calcular_custo_dieta(self) -> None:
        if self.custo_total is None:
            self.custo_total = 0
            for grupo in self.grupos_alimentares:
                if grupo.nome == 'Vegetables':
                    calorias_outros_vegetais = grupo.calorias - (
                        self.grupos_alimentares[10].calorias + self.grupos_alimentares[11].calorias
                    )
                    self.custo_total += calorias_outros_vegetais * grupo.valor_componente
                else:
                    self.custo_total += grupo.calorias * grupo.valor_componente

    def __str__(self) -> str:
        string = (
            f'Dieta: Calorias totais: {self.calorias_totais}, '
            f'Custo total: {self.custo_total}, '
            f'Pontuação PHDI: {self.pontuacao_phdi}, '
            f'Pontuação FO: {self.pontuacao_fo}, '
            f'Pontuação Aceitabilidade: {self.pontuacao_aceitabilidade}, '
            f'Penalizado?: {self.penalizado}, '
            f'Penalizacao aplicada: {self.penalizacao_aplicada}'
        )
        for grupo_alimentar in self.grupos_alimentares:
            string += f'\n   {grupo_alimentar}'
        return string

    def reset(self) -> None:
        self.calorias_totais = None
        self.custo_total = None
        self.pontuacao_phdi = None
        self.pontuacao_fo = None
        self.pontuacao_aceitabilidade = None
        self.penalizado = False
        self.penalizacao_aplicada = None
        self.penalizacao_aceitabilidade = None

    def copia(self):
        copia_dieta = Dieta([grupo.copia() for grupo in self.grupos_alimentares])
        copia_dieta.alvo_calorico = self.alvo_calorico
        copia_dieta.custo_alvo = self.custo_alvo
        copia_dieta.calorias_totais = self.calorias_totais
        copia_dieta.custo_total = self.custo_total
        copia_dieta.pontuacao_phdi = self.pontuacao_phdi
        copia_dieta.pontuacao_fo = self.pontuacao_fo
        copia_dieta.pontuacao_aceitabilidade = self.pontuacao_aceitabilidade
        copia_dieta.penalizado = self.penalizado
        copia_dieta.penalizacao_aplicada = self.penalizacao_aplicada
        copia_dieta.penalizacao_aceitabilidade = self.penalizacao_aceitabilidade

        return copia_dieta
    
    def calcular_phdi_dieta(self) -> None:
        """
        Calcula a pontuação total do PHDI da dieta somando as pontuações
        individuais de cada grupo alimentar.
        
        Pré-condição:
        - percentual_calorico já calculado
        """
        total = 0.0

        for grupo in self.grupos_alimentares:
            grupo.calculaPHDI()
            total += grupo.pontuacao_phdi or 0.0

        self.pontuacao_phdi = total
        
    def calcular_aceitabilidade_dieta(self) -> None:
        """
        Calcula a pontuação total de aceitabilidade cultural da dieta
        somando as pontuações individuais de cada grupo alimentar.
        
        Pré-condição:
        - percentual_calorico já calculado
        """
        total = 0.0

        for grupo in self.grupos_alimentares:
            grupo.calculaAceitabilidadeCultural()
            total += grupo.pontuacao_aceitabilidade or 0.0

        self.pontuacao_aceitabilidade = total

def _avaliar_dieta(variaveis: list, alvo_calorico: float=None, alvo_custo: float = None) -> tuple:
    nomes_grupos = [c['grupo'] for c in COMPONENTES_PHDI]
    custos_grupos = [c['custo'] for c in LIMITES_ACEITABILIDADE]
    grupos = [
        GrupoAlimentar(nome=nome, calorias=cal, valor_componente=custo)
        for nome, cal, custo in zip(nomes_grupos, variaveis, custos_grupos)
    ]

    dieta = Dieta(grupos)
    dieta.calcular_calorias_totais()
    dieta.calcular_percentual_calorico()
    dieta.calcular_phdi_dieta()
    dieta.calcular_aceitabilidade_dieta()
    dieta.calcular_custo_dieta()
    
    # print(f"custo total: {dieta.custo_total}")

    penalidade_calorica = 1
    if alvo_calorico is not None:
        erro_calorico = abs(dieta.calorias_totais - alvo_calorico) / alvo_calorico
        penalidade_calorica = max(1 - erro_calorico, 0.0)

    penalidade_custo = 1
    if alvo_custo is not None:
        erro_custo = abs(dieta.custo_total - alvo_custo) / alvo_custo
        penalidade_custo = max(1 - erro_custo, 0.0)

    return (
        dieta.pontuacao_phdi, 
        dieta.pontuacao_aceitabilidade,
        dieta.calorias_totais,
        dieta.custo_total,
        penalidade_calorica,
        penalidade_custo
    )

def construir_dieta(x: list) -> Dieta:

    nomes_grupos = [c['grupo'] for c in COMPONENTES_PHDI]
    custos_grupos = [c['custo'] for c in LIMITES_ACEITABILIDADE]

    grupos = [
        GrupoAlimentar(
            nome=nome,
            calorias=cal,
            valor_componente=custo
        )
        for nome, cal, custo in zip(nomes_grupos, x, custos_grupos)
    ]

    dieta = Dieta(grupos)

    dieta.calcular_calorias_totais()
    dieta.calcular_percentual_calorico()
    dieta.calcular_phdi_dieta()
    dieta.calcular_aceitabilidade_dieta()
    dieta.calcular_custo_dieta()

    return dieta

def calcular_custo_individuo(x: list) -> float:
    custos_unitarios = [entry['custo'] for entry in LIMITES_ACEITABILIDADE]
    
    custo_total = 0.0
    
    for i, calorias in enumerate(x):
        nome_grupo = COMPONENTES_PHDI[i]['grupo']
        custo_unitario = custos_unitarios[i]
        
        if nome_grupo == 'Vegetables':
            # Lógica especial: Vegetables total - (DGV + REV)
            # DGV está no índice 10, REV no índice 11
            calorias_liquidas = calorias - (x[10] + x[11])
            custo_total += calorias_liquidas * custo_unitario
        else:
            custo_total += calorias * custo_unitario
            
    return custo_total

def createDietProblem(alvo_calorico: float, alvo_custo: float) -> Problema:
    limites = [(entry['minimo'], entry['maximo']) for entry in LIMITES_ACEITABILIDADE]

    _cache = {'x': None, 'result': None}

    def _avaliar_com_cache(x):
        # Convertemos para tupla para o cache funcionar (listas não são hashable)
        xt = tuple(x)
        if _cache['x'] != xt:
            _cache['x'] = xt
            _cache['result'] = _avaliar_dieta(x, alvo_calorico, alvo_custo)
        return _cache['result']

    # Funções para o Problema extraírem dados do retorno de _avaliar_dieta
    def obj_phdi(x): return _avaliar_com_cache(x)[0]
    def obj_aceitabilidade(x): return _avaliar_com_cache(x)[1]
    def get_calorias(x): return _avaliar_com_cache(x)[2]
    def get_custo(x): return _avaliar_com_cache(x)[3]
    def penalidade_calorica(x): return _avaliar_com_cache(x)[4]
    def penalidade_custo(x): return _avaliar_com_cache(x)[5]
    
    def fo_phdi(x): return obj_phdi(x) * penalidade_calorica(x) * penalidade_custo(x)
    def fo_aceitabilidade(x): return obj_aceitabilidade(x) * penalidade_calorica(x) * penalidade_custo(x) 

    def gerar_individuo_dieta():
        import random
        # 1. Gera aleatório respeitando os limites do CSV
        x = [random.uniform(lim[0], lim[1]) for lim in limites]
        
        # 2. Normaliza calorias (idêntico ao que você já tinha)
        indices_excluidos = {10, 11}
        indices_caloricos = [i for i in range(len(x)) if i not in indices_excluidos]
        total_kcal = sum(x[i] for i in indices_caloricos)
        
        if total_kcal > 0:
            fator_kcal = alvo_calorico / total_kcal
            for i in indices_caloricos:
                x[i] *= fator_kcal
                x[i] = min(max(x[i], limites[i][0]), limites[i][1])
        
        # 3. OPCIONAL: forçar o custo aqui também, 
        # a lógica é mais complexa pois mexer no custo altera as calorias.

        return x


    problema = Problema(
        funcoes_objetivo=[fo_phdi, fo_aceitabilidade],
        limites=limites,
        tipos_otimizacao=[Otimizacao.MAX, Otimizacao.MAX]
    )
    
    problema.nomes_objetivos = ["FO PHDI", "FO Aceitabilidade"]
    problema.get_calorias = get_calorias
    problema.get_custo = get_custo
    problema.gerar_individuo = gerar_individuo_dieta 
    
    return problema

# Ao criar o problema, passe o custo_alvo que veio do CSV
problema_dieta = createDietProblem(alvo_calorico=alvo_calorico, alvo_custo=custo_alvo)