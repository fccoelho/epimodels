import numpy as np
import matplotlib.pyplot as plt

from epimodels.continuous.models import SIRNonAutonomous


# =============================
# 1. Definição dos parâmetros dependentes do tempo
# =============================

def beta(t):
    """Taxa de transmissão sazonal"""
    A=[0.16, 0.74]
    sigma = 3.2
    sk = -0.25
    return A[0] + A[1]* np.exp(-0.5*((t-12)/sigma+sk*(t-12))**2)


def gamma(t):
    """Taxa de recuperação"""
    return 0.1


def alpha(t):
    """Perda de imunidade (R -> S)"""
    return 0.2


# =============================
# 2. Função de simulação (importante para reutilização)
# =============================

def simulate(params, t_eval):
    model = SIRNonAutonomous()

    model(
        inits=[990, 10, 0],
        trange=[0, 160],
        totpop=1000,
        params=params,
        t_eval=t_eval,
        validate=False  # ⚠️ necessário (funções)
    )

    return model.traces


# =============================
# 3. Execução principal
# =============================

def main():
    t_eval = np.linspace(0, 50, 500)

    params = {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma
    }

    traces = simulate(params, t_eval)

    t = traces["time"]
    S = traces["S"]
    I = traces["I"]
    R = traces["R"]

    # =============================
    # FIGURA 1 — dinâmica completa
    # =============================
    plt.figure()

    plt.plot(t, S, label="S")
    plt.plot(t, I, label="I")
    plt.plot(t, R, label="R")

    plt.title("SIRS não-autônomo")
    plt.xlabel("Tempo")
    plt.ylabel("População")
    plt.legend()

    # =============================
    # FIGURA 2 — efeito de alpha
    # =============================
    plt.figure()

    for a in [0.01, 0.05, 0.1]:

        def alpha_var(t, a=a):
            return a

        params_var = {
            "alpha": alpha_var,
            "beta": beta,
            "gamma": gamma
        }

        traces = simulate(params_var, t_eval)

        plt.plot(
            traces["time"],
            traces["I"],
            label=rf"$\alpha = {a}$"
        )

    plt.title("Impacto da perda de imunidade")
    plt.xlabel("Tempo")
    plt.ylabel("Infectados")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()