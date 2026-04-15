from enum import Enum, auto

class Otimizacao(Enum):
    MIN = auto()
    MAX = auto()

class Problema:
    def __init__(self, funcoes_objetivo, limites, tipos_otimizacao):
        self.funcoes_objetivo = funcoes_objetivo
        self.limites = limites
        self.tipos_otimizacao = tipos_otimizacao

    def gerar_individuo(self):
        import random
        return [random.uniform(l[0], l[1]) for l in self.limites]

    def avaliar(self, variaveis):
        return [f(variaveis) for f in self.funcoes_objetivo]