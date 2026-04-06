import matplotlib.pyplot as plt
from epimodels.continuous.models import SIR
import numpy as np

t_eval = np.linspace(0, 160, 1000)

# Criar modelo
model = SIR()

# Rodar simulação (ESSA É A CHAVE)
model(
    inits=[999, 1, 0],          # S, I, R
    trange=[0, 160],            # intervalo de tempo
    totpop=1000,
    params={"beta": 0.3, "gamma": 0.1},
    t_eval=t_eval
)

# Extrair resultados
t = model.traces["time"]
S = model.traces["S"]
I = model.traces["I"]
R = model.traces["R"]

# Plot
plt.figure()
plt.plot(t, S, label="S")
plt.plot(t, I, label="I")
plt.plot(t, R, label="R")

plt.legend()
plt.xlabel("Time")
plt.ylabel("Population")
plt.title("SIR Model Simulation")

plt.figure()
for beta in [0.2, 0.3, 0.5]:
    model(
        inits=[999, 1, 0],
        trange=[0, 160],
        totpop=1000,
        params={"beta": beta, "gamma": 0.1},
        t_eval=t_eval
    )
    plt.plot(model.traces["time"], model.traces["I"], label=rf"$\beta={beta}$")

plt.legend()
plt.title("Effect of Different Beta Values on Infected Population")

plt.show()

# Curve Fitting