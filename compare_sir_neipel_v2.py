import matplotlib.pyplot as plt
import numpy as np
from epimodels.continuous.models import SIR, NeipelHeterogeneousSIR


# ============================================================
# 1. PARÂMETROS GERAIS
# ============================================================
N = 100000          # população total
I0 = 10             # infectados iniciais
R0_init = 0         # removidos iniciais no SIR clássico
S0 = N - I0 - R0_init

beta = 0.30         # taxa de transmissão
gamma = 0.10        # taxa de recuperação
t0 = 0
tf = 160

# Conjunto de alphas para testar
# alpha grande -> aproxima o caso homogêneo
# alpha pequeno -> heterogeneidade forte
alphas = [20.0, 10.0, 3.0, 1.0, 0.5, 0.3, 0.1]

# Pasta/nomes das figuras
fig1_name = "compare_infectados.png"
fig2_name = "compare_suscetiveis.png"
fig3_name = "compare_removidos.png"
fig4_name = "final_size_vs_alpha.png"
fig5_name = "peak_vs_alpha.png"


# ============================================================
# 2. FUNÇÃO AUXILIAR: reconstruir S e R no modelo do Neipel
# ============================================================
# No modelo implementado, os estados explícitos são:
#   y = [I, tau]
#
# Mas podemos reconstruir:
#   S(t) = (N - I0) * (1 + tau/alpha)^(-alpha)
#   R(t) = N - S(t) - I(t)
#
# Isso permite comparar diretamente com o SIR clássico.
def reconstruct_neipel_states(I, tau, N, I0, alpha):
    S = (N - I0) * (1 + tau / alpha) ** (-alpha)
    R = N - S - I
    return S, R


# ============================================================
# 3. RODAR O SIR CLÁSSICO
# ============================================================
sir = SIR()
sir(
    inits=[S0, I0, R0_init],
    trange=[t0, tf],
    totpop=N,
    params={
        "beta": beta,
        "gamma": gamma,
    },
)

t_sir = np.array(sir.traces["time"])
S_sir = np.array(sir.traces["S"])
I_sir = np.array(sir.traces["I"])
R_sir = np.array(sir.traces["R"])

sir_peak = np.max(I_sir)
sir_peak_time = t_sir[np.argmax(I_sir)]
sir_final_size = R_sir[-1]  # no SIR, R final = total acumulado infectado


# ============================================================
# 4. RODAR O MODELO DE NEIPEL PARA CADA ALPHA
# ============================================================
results = []

for alpha in alphas:
    neipel = NeipelHeterogeneousSIR()

    neipel(
        inits=[I0, 0],       # estados iniciais: I e tau
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

    peak = np.max(I_nei)
    peak_time = t_nei[np.argmax(I_nei)]
    final_size = R_nei[-1]

    results.append(
        {
            "alpha": alpha,
            "t": t_nei,
            "S": S_nei,
            "I": I_nei,
            "R": R_nei,
            "peak": peak,
            "peak_time": peak_time,
            "final_size": final_size,
        }
    )


# ============================================================
# 5. FIGURA 1: INFECTADOS
# ============================================================
plt.figure(figsize=(10, 6))
plt.plot(t_sir, I_sir, label="SIR clássico", linewidth=2)

for res in results:
    plt.plot(res["t"], res["I"], label=f'Neipel α={res["alpha"]}')

plt.xlabel("Tempo")
plt.ylabel("Infectados I(t)")
plt.title("Comparação entre SIR clássico e Neipel heterogêneo")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(fig1_name, dpi=300)
plt.show()


# ============================================================
# 6. FIGURA 2: SUSCETÍVEIS
# ============================================================
plt.figure(figsize=(10, 6))
plt.plot(t_sir, S_sir, label="SIR clássico", linewidth=2)

for res in results:
    plt.plot(res["t"], res["S"], label=f'Neipel α={res["alpha"]}')

plt.xlabel("Tempo")
plt.ylabel("Suscetíveis S(t)")
plt.title("Suscetíveis: SIR clássico vs Neipel heterogêneo")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(fig2_name, dpi=300)
plt.show()


# ============================================================
# 7. FIGURA 3: REMOVIDOS
# ============================================================
plt.figure(figsize=(10, 6))
plt.plot(t_sir, R_sir, label="SIR clássico", linewidth=2)

for res in results:
    plt.plot(res["t"], res["R"], label=f'Neipel α={res["alpha"]}')

plt.xlabel("Tempo")
plt.ylabel("Removidos R(t)")
plt.title("Removidos: SIR clássico vs Neipel heterogêneo")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(fig3_name, dpi=300)
plt.show()


# ============================================================
# 8. FIGURA 4: TAMANHO FINAL DA EPIDEMIA VS ALPHA
# ============================================================
alpha_vals = [res["alpha"] for res in results]
final_sizes = [res["final_size"] for res in results]

plt.figure(figsize=(8, 5))
plt.plot(alpha_vals, final_sizes, marker="o", label="Neipel heterogêneo")
plt.axhline(sir_final_size, linestyle="--", label="SIR clássico")

# Escala log em alpha ajuda a visualizar melhor
plt.xscale("log")
plt.xlabel("alpha")
plt.ylabel("Tamanho final da epidemia")
plt.title("Tamanho final da epidemia vs heterogeneidade")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(fig4_name, dpi=300)
plt.show()


# ============================================================
# 9. FIGURA 5: PICO DE INFECTADOS VS ALPHA
# ============================================================
peaks = [res["peak"] for res in results]

plt.figure(figsize=(8, 5))
plt.plot(alpha_vals, peaks, marker="o", label="Neipel heterogêneo")
plt.axhline(sir_peak, linestyle="--", label="SIR clássico")

plt.xscale("log")
plt.xlabel("alpha")
plt.ylabel("Pico de infectados")
plt.title("Pico de infectados vs heterogeneidade")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(fig5_name, dpi=300)
plt.show()


# ============================================================
# 10. RESUMO NUMÉRICO
# ============================================================
print("\n===== RESUMO NUMÉRICO =====")
print("\nSIR clássico")
print(f"R0 = {beta/gamma:.3f}")
print(f"Pico de infectados: {sir_peak:.2f}")
print(f"Tempo do pico: {sir_peak_time:.2f}")
print(f"Tamanho final da epidemia: {sir_final_size:.2f}")

for res in results:
    print(f'\nNeipel heterogêneo (alpha={res["alpha"]})')
    print(f"R0 = {beta/gamma:.3f}")
    print(f'Pico de infectados: {res["peak"]:.2f}')
    print(f'Tempo do pico: {res["peak_time"]:.2f}')
    print(f'Tamanho final da epidemia: {res["final_size"]:.2f}')


# ============================================================
# 11. CHECAGEM DE CONSERVAÇÃO DE MASSA
# ============================================================
# Aqui verificamos se S + I + R ~ N no último alpha,
# só para ter uma checagem rápida de coerência numérica.
last = results[-1]
total_check = last["S"] + last["I"] + last["R"]

print("\nChecagem de conservação de massa (último alpha):")
print(f"Mínimo de S+I+R: {np.min(total_check):.6f}")
print(f"Máximo de S+I+R: {np.max(total_check):.6f}")
print(f"População total N: {N}")