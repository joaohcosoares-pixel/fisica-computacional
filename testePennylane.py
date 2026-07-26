import pennylane as qml

N = 3 # Define que o sistema terá 3 qubits, resultando em um espaço de Hilbert de dimensão 2^3 = 8.
wires = range(N) # Cria uma sequência de índices para os qubits, que serão usados como "wires" no dispositivo quântico.
dev = qml.device("default.qubit", wires=wires) #Inicializa um dispositivo quântico simulado com 3 qubits, usando o backend padrão do PennyLane.

@qml.qnode(dev) #Transforma a função em um nó quântico, permitindo que seja executada no dispositivo quântico simulado.
def codifica_na_base(b): #define a função que codifica um vetor binário b em um estado quântico usando a base computacional.
    qml.BasisEmbedding(b, wires=wires) #É a porta quantica (operação) principal do circuito. 
    return qml.state() #instrui o nó quântico a retornar o vetor de estado resultante após a aplicação da codificação.