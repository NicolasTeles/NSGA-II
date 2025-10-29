import math
EPS = 1e-6

# diametro_roda => entre 10 e 30
# potencia_motor => exemplo: entre 50 e 200
# capacidade_bateria => exemplo: entre 100 e 500

def aut_ant(diametro, potencia, capacidade):
    return capacidade / (potencia * diametro)

def temp_acc_ant(diametro, potencia, capacidade):
    return (capacidade + diametro) / potencia

def aut_novo(diametro, potencia, capacidade):
    return math.sqrt(capacidade) / (potencia * abs(math.sin(diametro)) + EPS)

def temp_acc_novo(diametro, potencia, capacidade):
    return math.log1p(capacidade + abs(math.cos(diametro))) / math.sqrt(potencia)

def minmax_trig(valor_inicial):
    k = 0
    
    minmax_sen = {
        "min": 1,
        "max": 0
    }
    minmax_cos = {
        "min": 1,
        "max": 0
    }
    
    retorno = {
        "sen": {
            "min": 10,
            "max": 30
        },
        
        "cos":{
            "min": 10,
            "max": 30
        }
    }
    
    while valor_inicial + k*math.pi <= 30:
        if valor_inicial + k*math.pi < 10:
            k += math.pi
            continue
        
        seno = math.sin(valor_inicial + k*math.pi)
        if seno < minmax_sen["min"]:
            minmax_sen["min"] = seno
            retorno["sen"]["min"] = valor_inicial + k*math.pi
            
        if seno > minmax_sen["max"]:
            minmax_sen["max"] = seno
            retorno["sen"]["max"] = valor_inicial + k*math.pi
            
        cosseno = math.cos(valor_inicial + k*math.pi) 
        if cosseno < minmax_cos["min"]:
            minmax_cos["min"] = cosseno
            retorno["cos"]["min"] = valor_inicial + k*math.pi
            
        if cosseno > minmax_cos["max"]:
            minmax_cos["max"] = cosseno
            retorno["cos"]["max"] = valor_inicial + k*math.pi
        
        k += math.pi
            
    return retorno

trig_inteiro = minmax_trig(0)
trig_sobre2 = minmax_trig(math.pi/2)

D, P, C = 30, 200, 100
print("MIN AUTONOMIA ANTIGA")
# print(f"AUTONOMIA: {aut_ant(D, P, C)} TEMPO ACELERACAO: {temp_acc_ant(D, P, C)}\n\n")
print(f"({aut_ant(D, P, C)}, {temp_acc_ant(D, P, C)})\n\n")


# min autonomina => min sen(diametro) => diametro = k*pi
D = trig_inteiro["sen"]["min"]
print("MIN AUTONOMIA NOVA")
# print(f"AUTONOMIA: {aut_ant(D, P, C)} TEMPO ACELERACAO: {temp_acc_ant(D, P, C)}\n\n")
print(f"({aut_novo(D, P, C)}, {temp_acc_novo(D, P, C)})\n\n")

print("======================================================================")

D, P, C = 10, 50, 500
print("MAX AUTONOMIA ANTIGA")
# print(f"AUTONOMIA: {aut_ant(D, P, C)} TEMPO ACELERACAO: {temp_acc_ant(D, P, C)}\n\n")
print(f"({aut_ant(D, P, C)}, {temp_acc_ant(D, P, C)})\n\n")


# max autonomia => max sen(diametro) => diametro = pi/2+k*pi
D = trig_sobre2["sen"]["max"]
print("MAX AUTONOMIA NOVA")
# print(f"AUTONOMIA: {aut_ant(D, P, C)} TEMPO ACELERACAO: {temp_acc_ant(D, P, C)}\n\n")
print(f"({aut_novo(D, P, C)}, {temp_acc_novo(D, P, C)})\n\n")

print("======================================================================")

D, P, C = 10, 200, 100
print("MIN TEMPO ACELERACAO ANTIGO")
# print(f"AUTONOMIA: {aut_ant(D, P, C)} TEMPO ACELERACAO: {temp_acc_ant(D, P, C)}\n\n")
print(f"({aut_ant(D, P, C)}, {temp_acc_ant(D, P, C)})\n\n")


# min tempo aceleracao => min cos(diametro) => diametro = pi/2+k*pi
D = trig_sobre2["cos"]["min"]
print("MIN TEMPO ACELERACAO NOVO")
# print(f"AUTONOMIA: {aut_ant(D, P, C)} TEMPO ACELERACAO: {temp_acc_ant(D, P, C)}\n\n")
print(f"({aut_novo(D, P, C)}, {temp_acc_novo(D, P, C)})\n\n")

print("======================================================================")

D, P, C = 30, 50, 500
print("MAX TEMPO ACELERACAO ANTIGO")
# print(f"AUTONOMIA: {aut_ant(D, P, C)} TEMPO ACELERACAO: {temp_acc_ant(D, P, C)}\n\n")
print(f"({aut_ant(D, P, C)}, {temp_acc_ant(D, P, C)})\n\n")


# max tempo aceleracao => max cos(diametro) => diametro = k*pi

D = trig_inteiro["cos"]["max"]
print("MAX TEMPO ACELERACAO NOVO")
# print(f"AUTONOMIA: {aut_ant(D, P, C)} TEMPO ACELERACAO: {temp_acc_ant(D, P, C)}\n\n")
print(f"({aut_novo(D, P, C)}, {temp_acc_novo(D, P, C)})\n\n")