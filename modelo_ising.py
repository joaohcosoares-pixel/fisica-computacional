import numpy as np

# Cria uma matriz 10x10 com valores aleatórios de -1 ou 1 
# Isso representa os spins dos elétrons em uma rede bidimensional
malha_spins = np.random.choice([-1, 1], size=(10, 10))

print("Estado inicial da malha térmica:")
print(malha_spins)