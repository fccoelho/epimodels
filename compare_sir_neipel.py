import matplotlib.pyplot as plt
import numpy as np

# Importa os dois modelos da biblioteca
from epimodels.continuous.models import SIR, NeipelHeterogeneousSIR


# =========================
# 1. PARÂMETROS GERAIS
# =========================
# Escolhemos valores simples para comparar os modelos.
# Você pode trocar depois.

N = 100000          # população total
I0 = 10             # infectados iniciais
R0_init = 0         # removidos iniciais no SIR clássico
S0 = N - I0 - R0_init

beta = 0.30         # taxa de transmissão
gamma = 0.10        # taxa de recuperação
t0 = 0              # tempo inicial
tf = 160            # tempo final

# Valores de alpha para testar no modelo heterogêneo do Neipel.
# alpha grande -> comportamento mais próximo do SIR homogêneo
# alpha pequeno -> heterogeneidade mais forte
alphas = [10.0, 1.0, 0.3, 0.1]


# =========================
# 2. RODAR O SIR CLÁSSICO
# =========================
sir = SIR()

sir(
    inits=[S0, I0, R0_init],             # estados iniciais do SIR: S, I, R
    trange=[t0, tf],                     # intervalo de tempo
    totpop=N,                            # população total
    params={
        "beta": beta,
        "gamma": gamma,
    },
)

# Guardamos os resultados
t_sir = np.array(sir.traces["time"])
S_sir = np.array(sir.traces["S"])
I_sir = np.array(sir.traces["I"])
R_sir = np.array(sir.traces["R"])


# =========================
# 3. FUNÇÃO AUXILIAR
# =========================
# No modelo do Neipel, os estados explícitos são I e tau.
# Então reconstruímos S e R usando as fórmulas do artigo:
#
# S(t) = (N - I0) * (1 + tau/alpha)^(-alpha)
# R(t) = N - S(t) - I(t)
#
# Isso nos permite comparar diretamente com o SIR clássico.

def reconstruct_neipel_states(I, tau, N, I0, alpha):
    S = (N - I0) * (1 + tau / alpha) ** (-alpha)
    R = N - S - I
    return S, R


# =========================
# 4. PLOT 1: INFECTADOS
# =========================
plt.figure(figsize=(10, 6))

# Primeiro plota o SIR clássico
plt.plot(t_sir, I_sir, label="SIR clássico", linewidth=2)

# Agora roda o modelo do Neipel para cada alpha
for alpha in alphas:
    neipel = NeipelHeterogeneousSIR()

    neipel(
        inits=[I0, 0],                   # estados iniciais: I, tau
        trange=[t0, tf],
        totpop=N,
        params={
            "beta": beta,
            "gamma": gamma,
            "alpha": alpha,
            "I0": I0,
        },
    )

    t_nei = np.array(neipel.traces["time"])
    I_nei = np.array(neipel.traces["I"])
    tau_nei = np.array(neipel.traces["tau"])

    # Reconstrói S e R, embora aqui só usemos I
    S_nei, R_nei = reconstruct_neipel_states(I_nei, tau_nei, N, I0, alpha)

    plt.plot(t_nei, I_nei, label=f"Neipel alpha={alpha}")

plt.xlabel("Tempo")
plt.ylabel("Infectados I(t)")
plt.title("Comparação entre SIR clássico e Neipel heterogêneo")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# =========================
# 5. PLOT 2: SUSCETÍVEIS
# =========================
plt.figure(figsize=(10, 6))

plt.plot(t_sir, S_sir, label="SIR clássico", linewidth=2)

for alpha in alphas:
    neipel = NeipelHeterogeneousSIR()

    neipel(
        inits=[I0, 0],
        trange=[t0, tf],
        totpop=N,
        params={
            "beta": beta,
            "gamma": gamma,
            "alpha": alpha,
            "I0": I0,
        },
    )

    t_nei = np.array(neipel.traces["time"])
    I_nei = np.array(neipel.traces["I"])
    tau_nei = np.array(neipel.traces["tau"])

    S_nei, R_nei = reconstruct_neipel_states(I_nei, tau_nei, N, I0, alpha)

    plt.plot(t_nei, S_nei, label=f"Neipel alpha={alpha}")

plt.xlabel("Tempo")
plt.ylabel("Suscetíveis S(t)")
plt.title("Suscetíveis: SIR clássico vs Neipel heterogêneo")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# =========================
# 6. PLOT 3: REMOVIDOS
# =========================
plt.figure(figsize=(10, 6))

plt.plot(t_sir, R_sir, label="SIR clássico", linewidth=2)

for alpha in alphas:
    neipel = NeipelHeterogeneousSIR()

    neipel(
        inits=[I0, 0],
        trange=[t0, tf],
        totpop=N,
        params={
            "beta": beta,
            "gamma": gamma,
            "alpha": alpha,
            "I0": I0,
        },
    )

    t_nei = np.array(neipel.traces["time"])
    I_nei = np.array(neipel.traces["I"])
    tau_nei = np.array(neipel.traces["tau"])

    S_nei, R_nei = reconstruct_neipel_states(I_nei, tau_nei, N, I0, alpha)

    plt.plot(t_nei, R_nei, label=f"Neipel alpha={alpha}")

plt.xlabel("Tempo")
plt.ylabel("Removidos R(t)")
plt.title("Removidos: SIR clássico vs Neipel heterogêneo")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# =========================
# 7. RESUMO NUMÉRICO
# =========================
# Aqui mostramos algumas quantidades úteis:
# - pico de infectados
# - tempo do pico
# - total final de removidos
#
# No SIR clássico, R(final) é o total de infectados acumulados ao final.
# No Neipel, como reconstruímos R, vale a mesma ideia.

print("\n===== RESUMO NUMÉRICO =====")

# SIR clássico
peak_idx_sir = np.argmax(I_sir)
print("\nSIR clássico")
print(f"R0 = {beta/gamma:.3f}")
print(f"Pico de infectados: {I_sir[peak_idx_sir]:.2f}")
print(f"Tempo do pico: {t_sir[peak_idx_sir]:.2f}")
print(f"Removidos finais: {R_sir[-1]:.2f}")

# Neipel para vários alpha
for alpha in alphas:
    neipel = NeipelHeterogeneousSIR()

    neipel(
        inits=[I0, 0],
        trange=[t0, tf],
        totpop=N,
        params={
            "beta": beta,
            "gamma": gamma,
            "alpha": alpha,
            "I0": I0,
        },
    )

    t_nei = np.array(neipel.traces["time"])
    I_nei = np.array(neipel.traces["I"])
    tau_nei = np.array(neipel.traces["tau"])

    S_nei, R_nei = reconstruct_neipel_states(I_nei, tau_nei, N, I0, alpha)

    peak_idx_nei = np.argmax(I_nei)

    print(f"\nNeipel heterogêneo (alpha={alpha})")
    print(f"R0 = {beta/gamma:.3f}")
    print(f"Pico de infectados: {I_nei[peak_idx_nei]:.2f}")
    print(f"Tempo do pico: {t_nei[peak_idx_nei]:.2f}")
    print(f"Removidos finais: {R_nei[-1]:.2f}")