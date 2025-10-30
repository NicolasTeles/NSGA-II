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
    denominador = max(potencia * abs(math.sin(diametro)), 0.1)
    return math.sqrt(capacidade) / denominador

def temp_acc_novo(diametro, potencia, capacidade):
    return math.sqrt(capacidade + abs(math.cos(diametro))) / math.sqrt(potencia)

D, P, C = 30, 200, 100
print("MIN AUTONOMIA ANTIGA")
# print(f"AUTONOMIA: {aut_ant(D, P, C)} TEMPO ACELERACAO: {temp_acc_ant(D, P, C)}\n\n")
print(f"({aut_ant(D, P, C)}, {temp_acc_ant(D, P, C)})\n\n")


# min autonomia => max sen(diametro) => diametro = k*pi + pi/2
D = (3+1/2)*math.pi
print("MIN AUTONOMIA NOVA")
# print(f"AUTONOMIA: {aut_ant(D, P, C)} TEMPO ACELERACAO: {temp_acc_ant(D, P, C)}\n\n")
print(f"({aut_novo(D, P, C)}, {temp_acc_novo(D, P, C)})\n\n")

print("======================================================================")

D, P, C = 10, 50, 500
print("MAX AUTONOMIA ANTIGA")
# print(f"AUTONOMIA: {aut_ant(D, P, C)} TEMPO ACELERACAO: {temp_acc_ant(D, P, C)}\n\n")
print(f"({aut_ant(D, P, C)}, {temp_acc_ant(D, P, C)})\n\n")


# max autonomia => min sen(diametro) => diametro = pi/2+k*pi
D = 3*math.pi
print("MAX AUTONOMIA NOVA")
# print(f"AUTONOMIA: {aut_ant(D, P, C)} TEMPO ACELERACAO: {temp_acc_ant(D, P, C)}\n\n")
print(f"({aut_novo(D, P, C)}, {temp_acc_novo(D, P, C)})\n\n")

print("======================================================================")

D, P, C = 10, 200, 100
print("MIN TEMPO ACELERACAO ANTIGO")
# print(f"AUTONOMIA: {aut_ant(D, P, C)} TEMPO ACELERACAO: {temp_acc_ant(D, P, C)}\n\n")
print(f"({aut_ant(D, P, C)}, {temp_acc_ant(D, P, C)})\n\n")


# min tempo aceleracao => min cos(diametro) => diametro = pi/2+k*pi
D = (3+1/2)*math.pi
print("MIN TEMPO ACELERACAO NOVO")
# print(f"AUTONOMIA: {aut_ant(D, P, C)} TEMPO ACELERACAO: {temp_acc_ant(D, P, C)}\n\n")
print(f"({aut_novo(D, P, C)}, {temp_acc_novo(D, P, C)})\n\n")

print("======================================================================")

D, P, C = 30, 50, 500
print("MAX TEMPO ACELERACAO ANTIGO")
# print(f"AUTONOMIA: {aut_ant(D, P, C)} TEMPO ACELERACAO: {temp_acc_ant(D, P, C)}\n\n")
print(f"({aut_ant(D, P, C)}, {temp_acc_ant(D, P, C)})\n\n")


# max tempo aceleracao => max cos(diametro) => diametro = k*pi

D = 3*math.pi
print("MAX TEMPO ACELERACAO NOVO")
# print(f"AUTONOMIA: {aut_ant(D, P, C)} TEMPO ACELERACAO: {temp_acc_ant(D, P, C)}\n\n")
print(f"({aut_novo(D, P, C)}, {temp_acc_novo(D, P, C)})\n\n")