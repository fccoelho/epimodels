from epimodels.continuous.models import NeipelHeterogeneousSIR
import matplotlib.pyplot as plt

model = NeipelHeterogeneousSIR()

N = 100000
I0 = 10
alpha = 0.1

model(
    inits=[10, 0],
    trange=[0, 160],
    totpop=N,
    params={
        "beta": 0.3,
        "gamma": 0.1,
        "alpha": alpha,
        "I0": I0,
    },
)

I = model.traces["I"]
tau = model.traces["tau"]
t = model.traces["time"]

S = (N - I0) * (1 + tau / alpha) ** (-alpha)
R = N - S - I

plt.plot(t, S, label="S")
plt.plot(t, I, label="I")
plt.plot(t, R, label="R")
plt.legend()
plt.xlabel("tempo")
plt.ylabel("população")
plt.show()