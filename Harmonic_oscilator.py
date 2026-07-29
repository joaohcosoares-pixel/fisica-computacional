import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def harmonic_oscillator(t, estado, m, c, k):

    x1 = estado[0] # Position
    x2 = estado[1] # Velocity

    # Two equations in first order form
    dx1dt = x2
    dx2dt = -(c/m)*x2 - (k/m)*x1

    return np.array([dx1dt, dx2dt])

m = 1.0
k = 1.0
estado_inicial = [0.0, 5.0]  # Initial state: [position, velocity]

casos_c = {
    'subamortecido': 0.1,  
    'criticamente_amortecido': 2.0,
    'superamortecido': 4.0
}

# Tempo de simulação
t_span = (0, 15)
t_eval = np.linspace(t_span[0], t_span[1], 300) #300 time points for evaluation

# Loop through each damping case and solve the ODE
trajetories = {}
for caso, c in casos_c.items():
    sol = solve_ivp(harmonic_oscillator, t_span, estado_inicial, args=(m, c, k), t_eval=t_eval)
    trajetories[caso] = sol 

# Plotting the results
 
fig, ax = plt.subplots(figsize=(10, 6))

for caso, sol in trajetories.items():
    ax.plot(sol.t, sol.y[0], label=f'{caso} (c={casos_c[caso]})')

ax.set_title('Harmonic Oscillator with Different Damping Cases')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position (x)')
ax.legend()

plt.show()