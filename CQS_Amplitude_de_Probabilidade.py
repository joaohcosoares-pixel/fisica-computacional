import pennylane as qml
import numpy as np

N = 3
wires = range(N)
dev = qml.device("default.qubit", wires=wires)

@qml.qnode(dev)
def amplitude_de_probabilidade(features):

    # Codificação de amplitudes 
    qml.AmplitudeEmbedding(features, wires=wires, normalize=True)
    return qml.state() # Retorna o vetor de estado resultante após a aplicação da codificação de amplitudes.
# Definição do vetor clássico de entrada com exatamente 2^3 = 8 elementos
dados_brutos = np.array([1.0, 2.5, 0.0, 4.1, 3.2, 0.5, 0.0, 1.1])

# Execução do circuito quântico
estado_resultante = amplitude_de_probabilidade(dados_brutos)

# Impressão do resultado
print("Vetor de Estado Resultante (Normalizado):")
print(estado_resultante)