from models import Problema, Otimizacao
from typing import List, Optional, Dict
import pandas as pd

# --- Índices dos grupos alimentares no vetor de decisão ---
IDX_NUTS            = 0
IDX_LEGUMES         = 1
IDX_FRUITS          = 2
IDX_VEGETABLES      = 3
IDX_WHOLE_CEREALS   = 4
IDX_EGGS            = 5
IDX_FISH            = 6
IDX_TUBERS          = 7
IDX_DAIRY           = 8
IDX_OILS            = 9
IDX_DGV             = 10
IDX_REV             = 11
IDX_RED_MEAT        = 12
IDX_CHICKEN         = 13
IDX_ANIMAL_FATS     = 14
IDX_ADDED_SUGARS    = 15
IDX_REFINED_CEREALS = 16

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

    {'grupo': 'DGV', 'limite_inferior': 0.0, 'limite_otimo': 29.5, 'limite_superior': 100.0, 'tipo': 'razao'},
    {'grupo': 'REV', 'limite_inferior': 0.0, 'limite_otimo': 38.5, 'limite_superior': 100.0, 'tipo': 'razao'},

    {'grupo': 'Red meat', 'limite': 2.4, 'tipo': 'moderacao'},
    {'grupo': 'Chicken and substitutes', 'limite': 5.0, 'tipo': 'moderacao'},
    {'grupo': 'Animal fats', 'limite': 1.4, 'tipo': 'moderacao'},
    {'grupo': 'Added sugars', 'limite': 4.8, 'tipo': 'moderacao'},
]

LIMITES_ACEITABILIDADE = []

# Grupo do código -> (sufixo das colunas no banco INA, ajuste do p25/p75, valor fixo em % do alvo calórico).
# ajuste: 'adequacao' => p25 = max(p25, média); 'moderacao' => p75 = min(p75, média); None => sem ajuste.
GRUPOS_BANCO = [
    ("Nuts and peanuts",        "nozes",       "adequacao", 11.6),
    ("Legumes",                 "leguminosas", "adequacao", None),
    ("Fruits",                  "frutas",      "adequacao", None),
    ("Vegetables",              "veg_totais",  "adequacao", None),
    ("Whole cereals",           "cereais_int", "adequacao", 0.8),
    ("Eggs",                    "ovos",        None,        None),
    ("Fish and seafood",        "peixes",      None,        1.27),
    ("Tubers and potatoes",     "batatas",     None,        None),
    ("Dairy",                   "laticinios",  None,        None),
    ("Vegetable oils",          "oleos",       None,        None),
    ("DGV",                     "veg_verde",   None,        0.11),
    ("REV",                     "veg_verm",    None,        None),
    ("Red meat",                "carnes_verm", "moderacao", None),
    ("Chicken and substitutes", "frango",      "moderacao", None),
    ("Animal fats",             "gordura",     "moderacao", None),
    ("Added sugars",            "acucar",      "moderacao", None),
    ("Refined Cereals",         "cereais_ref", None,        30),
]


def custo_por_kcal(row, sufixo):
    """
    Custo por kcal do grupo: preco_<sufixo> é o custo TOTAL do consumo médio (R$/pessoa/dia),
    então basta dividir pelas kcal médias do grupo.

    Exceção para Vegetables: o cálculo do custo da dieta multiplica esse valor apenas pelas kcal
    de "outros vegetais" (Vegetables - DGV - REV), logo preço e kcal precisam excluir os subgrupos.
    """
    if sufixo == "veg_totais":
        preco = row["preco_veg_totais"] - row["preco_veg_verde"] - row["preco_veg_verm"]
        kcal = row["veg_totais_kcal"] - row["veg_verde_kcal"] - row["veg_verm_kcal"]
    else:
        preco = row[f"preco_{sufixo}"]
        kcal = row[f"{sufixo}_kcal"]
    return float(preco) / float(kcal)


def extrai_da_linha(LIMITES_ACEITABILIDADE, row, alvo_calorico):
    for grupo, sufixo, ajuste, valor_fixo in GRUPOS_BANCO:
        media = float(row[f"{sufixo}_kcal"])
        p25 = float(row[f"p25_{sufixo}_kcal"])
        p75 = float(row[f"p75_{sufixo}_kcal"])

        if ajuste == "adequacao":
            p25 = max(p25, media)
        elif ajuste == "moderacao":
            p75 = min(p75, media)

        entrada = {
            "grupo": grupo,
            "minimo": float(row[f"p10_{sufixo}_kcal"]),
            "p25": p25,
            "p75": p75,
            "maximo": float(row[f"p90_{sufixo}_kcal"]),
            "custo": custo_por_kcal(row, sufixo),
            "kcal_total": alvo_calorico,
        }
        if valor_fixo is not None:
            entrada["valor_fixo"] = valor_fixo

        LIMITES_ACEITABILIDADE.append(entrada)


df_INA = pd.read_csv('banco_INA_estratificado_ag.csv')

# Linha nacional: uf = 99 e faixa_renda_per_capita vazia. A faixa de renda vazia é o identificador.
_nacional = df_INA[df_INA["faixa_renda_per_capita"].isna()]
if len(_nacional) != 1:
    raise ValueError(
        f"linha nacional não encontrada no banco: esperava exatamente 1 linha com "
        f"faixa_renda_per_capita vazia em 'banco_INA_estratificado_ag.csv'; encontrei {len(_nacional)}."
    )

row = _nacional.iloc[0]

# veg_totais já inclui veg_verde e veg_verm, então kcal_totais/preco_total contam os subgrupos
# duas vezes (uma dentro de veg_totais e outra como subgrupo). Descontamos a duplicata.
alvo_calorico = float(row["kcal_totais"] - row["veg_verde_kcal"] - row["veg_verm_kcal"])
print(f'Alvo calorico: {alvo_calorico}')

custo_alvo = float(row["preco_total"] - row["preco_veg_verde"] - row["preco_veg_verm"])
print(f"Valor alvo: {custo_alvo}")

extrai_da_linha(LIMITES_ACEITABILIDADE, row, alvo_calorico)

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
            self.pontuacao_phdi = 0.0
            return
        
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
                
            case 'razao':
                if valor <= componente['limite_otimo']:
                    self.pontuacao_phdi = min((valor / componente['limite_otimo']) * 5, 5)
                else:
                    faixa = componente['limite_superior'] - componente['limite_otimo']
                    excesso = valor - componente['limite_otimo']
                    self.pontuacao_phdi = min(max(5 - (excesso / faixa) * 5, 0), 5)
    
    
    def calculaAceitabilidadeCultural(self, kcal_ref=None, max_pts=10):
        componente = next((c for c in LIMITES_ACEITABILIDADE if c['grupo'].lower() == self.nome.lower()), None)

        if componente is None:
            raise ValueError(f"Grupo '{self.nome}' não encontrado em LIMITES_ACEITABILIDADE.")

        valor = self.percentual_calorico
        ref = kcal_ref if kcal_ref is not None else componente["kcal_total"]

        minimo = componente["minimo"] / ref * 100
        p25 = componente["p25"]    / ref * 100
        p75 = componente["p75"]    / ref * 100
        maximo = componente["maximo"] / ref * 100

         # Fora dos limites absolutos → 0
        if valor <= minimo or valor >= maximo:
            self.pontuacao_aceitabilidade = 0
            return

        # Região ideal → max
        if p25 <= valor <= p75:
            self.pontuacao_aceitabilidade = max_pts
            return

        # Crescimento linear (min → p25)
        if minimo < valor < p25:
            self.pontuacao_aceitabilidade = (valor - minimo) / (p25 - minimo) * max_pts
            return

        # Decaimento linear (p75 → max)
        if p75 < valor < maximo:
            self.pontuacao_aceitabilidade = (maximo - valor) / (maximo - p75) * max_pts
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
                        self.grupos_alimentares[IDX_DGV].calorias + self.grupos_alimentares[IDX_REV].calorias
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
        veg = next(g for g in self.grupos_alimentares if g.nome == 'Vegetables')

        for grupo in self.grupos_alimentares:
            if grupo.nome in ('DGV', 'REV'):
                grupo.calculaAceitabilidadeCultural(kcal_ref=veg.calorias, max_pts=5)
            else:
                grupo.calculaAceitabilidadeCultural()
            total += grupo.pontuacao_aceitabilidade or 0.0

        self.pontuacao_aceitabilidade = total

def _avaliar_dieta(variaveis: list, alvo_calorico: float=None, alvo_custo: float = None) -> tuple:
    nomes_grupos = [c['grupo'] for c in LIMITES_ACEITABILIDADE]
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
    
    nomes_grupos = [c['grupo'] for c in LIMITES_ACEITABILIDADE]
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
        nome_grupo = LIMITES_ACEITABILIDADE[i]['grupo']
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
        xt = tuple(x)
        if _cache['x'] != xt:
            _cache['x'] = xt
            _cache['result'] = _avaliar_dieta(x, alvo_calorico, alvo_custo)
        return _cache['result']

    def obj_phdi(x): return _avaliar_com_cache(x)[0]
    def obj_aceitabilidade(x): return _avaliar_com_cache(x)[1]
    def get_calorias(x): return _avaliar_com_cache(x)[2]
    def get_custo(x): return _avaliar_com_cache(x)[3]
    def penalidade_calorica(x): return _avaliar_com_cache(x)[4]
    def penalidade_custo(x): return _avaliar_com_cache(x)[5]
    
    def fo_phdi(x): return obj_phdi(x) * penalidade_calorica(x) * penalidade_custo(x)
    def fo_aceitabilidade(x): return obj_aceitabilidade(x) * penalidade_calorica(x) * penalidade_custo(x) 

    def aplicar_limites_dieta(x: list) -> list:
        x_lim = list(x)
        for i, (grupo, lim) in enumerate(zip(LIMITES_ACEITABILIDADE, limites)):
            if 'valor_fixo' in grupo:
                x_lim[i] = (grupo['valor_fixo'] / 100) * alvo_calorico
            else:
                x_lim[i] = min(max(x_lim[i], lim[0]), lim[1])

        # Garante consistência física dos subgrupos: DGV + REV <= Vegetables
        soma_sub_veg = x_lim[IDX_DGV] + x_lim[IDX_REV]
        if x_lim[IDX_VEGETABLES] < soma_sub_veg:
            # Se possível, eleva Vegetables até soma_sub_veg (respeitando o máximo de Vegetables)
            x_lim[IDX_VEGETABLES] = min(soma_sub_veg, limites[IDX_VEGETABLES][1])
            # Se ainda assim DGV + REV superar Vegetables, reduz REV até caber em Vegetables
            if x_lim[IDX_VEGETABLES] < soma_sub_veg:
                x_lim[IDX_REV] = max(limites[IDX_REV][0], x_lim[IDX_VEGETABLES] - x_lim[IDX_DGV])

        return x_lim

    def gerar_individuo_dieta():
        import random
        x = []
        for grupo, lim in zip(LIMITES_ACEITABILIDADE, limites):
            if 'valor_fixo' in grupo:
                kcal_fixo = (grupo['valor_fixo'] / 100) * alvo_calorico
                x.append(kcal_fixo)
            else:
                x.append(random.uniform(lim[0], lim[1]))
        
        # 2. Normaliza calorias
        indices_excluidos = {IDX_REV}
        indices_fixos = {i for i, e in enumerate(LIMITES_ACEITABILIDADE) if 'valor_fixo' in e}
        indices_caloricos = [i for i in range(len(x)) if i not in indices_excluidos and i not in indices_fixos]

        total_livre = sum(x[i] for i in indices_caloricos)
        total_fixo = sum(x[i] for i in indices_fixos if i not in indices_excluidos)
        
        if total_livre > 0:
            alvo_restante = alvo_calorico - total_fixo
            fator = alvo_restante / total_livre
            for i in indices_caloricos:
                x[i] *= fator
                x[i] = min(max(x[i], limites[i][0]), limites[i][1])

        return aplicar_limites_dieta(x)

    problema = Problema(
        funcoes_objetivo=[fo_phdi, fo_aceitabilidade],
        limites=limites,
        tipos_otimizacao=[Otimizacao.MAX, Otimizacao.MAX]
    )
    
    problema.nomes_objetivos = ["FO PHDI", "FO Aceitabilidade"]
    problema.get_calorias = get_calorias
    problema.get_custo = get_custo
    problema.gerar_individuo = gerar_individuo_dieta
    problema.aplicar_limites = aplicar_limites_dieta
    
    return problema

# Ao criar o problema, passe o custo_alvo que veio do CSV
problema_dieta = createDietProblem(alvo_calorico=alvo_calorico, alvo_custo=custo_alvo)