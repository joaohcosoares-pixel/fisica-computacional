import pennylane as qml
from pennylane import numpy as np

N = 3 # Define que o sistema terá 3 qubits, resultando em um espaço de Hilbert de dimensão 2^3 = 8.
wires = range(N) # Cria uma sequência de índices para os qubits, que
dev = qml.device("default.qubit", wires=wires) # Inicializa um dispositivo quântico simulado com 3 qubits, usando o backend padrão do PennyLane.

@qml.qnode(dev) # Transforma a função em um nó quântico, permitindo que seja executada no dispositivo quântico simulado.
def codifica_na_base(val_list):
    # codificação por angulo
    qml.AngleEmbedding(val_list, wires=wires) # É a porta quantica (operação) principal do circuito.
    # Medida: valor esperado de Pauli-Z para cada qubit
    return [qml.expval(qml.PauliZ(w)) for w in wires] # instrui o nó quântico a retornar o vetor de estado

# 1. Definir os dados clássicos de entrada como um array do pennylane.numpy
# Os valores representam ângulos em radianos
dados_entrada = np.array([0.5, 1.2, 3.14])

# 2. Executar o circuito quântico
resultados = codifica_na_base(dados_entrada)

# 3. Exibir a saída das medições
print("Valores esperados de Pauli-Z para cada qubit:")
print(resultados)