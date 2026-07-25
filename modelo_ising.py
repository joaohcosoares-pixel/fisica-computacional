import numpy as np

def calcular_energia(malha, J=1.0):
    """
    Calcula a energia total do estado microscópico via vetorização de tensores.
    Aplica condições de contorno periódicas usando np.roll.
    """
    # Deslocamentos matriciais paralelos para localizar vizinhos adjacentes (Cima, Baixo, Esquerda, Direita)
    vizinhos = (np.roll(malha, 1, axis=0) +
                np.roll(malha, -1, axis=0) +
                np.roll(malha, 1, axis=1) +
                np.roll(malha, -1, axis=1))
    
    # Produto de Hadamard (element-wise) e soma global. O fator 0.5 corrige a dupla contagem.
    energia_total = -J * np.sum(malha * vizinhos) * 0.5
    return energia_total

# Inicialização do estado estocástico
np.random.seed(42) # Fixa a semente para reprodutibilidade das medições
malha_spins = np.random.choice([-1, 1], size=(10, 10))

# Processamento do observável físico
energia = calcular_energia(malha_spins)

print(f"Estado do Spin (Matriz 10x10):\n{malha_spins}")
print(f"\nEnergia Total do Sistema: {energia} Joules (ou unidades de J)")