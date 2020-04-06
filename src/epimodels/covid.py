import numpy as np


def seqihr(y,t,*params):
    S,E,Q,I,H,R = y
    chi,phi,beta,rho,delta,alpha = params
    return[
        -beta*S*(I+(1-rho)*H), #dS/dt
        beta*S*(I+(1-rho)*H) - (chi+alpha)*E,#dE/dt
        chi*E -alpha*Q,#dQ/dt
        alpha*E - (phi+delta)*I,#dI/dt
        alpha*Q + phi*I -delta*H,#dH/dt
        delta*I + delta*H,#dR/dt
    ]


def seqiahr(y,t,*params):
    S,E,I,A,H,R,C = y
    chi,phi,beta,rho,delta,alpha,p,q = params
    lamb = beta*S*(I+A+(1-rho)*H)
    chi *= (1+np.tanh(t-q))/2 #Liga a quarentena dia q
    return[
        -lamb*((1-chi)*S), #dS/dt
        lamb*((1-chi)*S) - alpha*E,#dE/dt
        (1-p)*alpha*E - (phi+delta)*I,#dI/dt
        p*alpha*E - delta*A,
        phi*I -delta*H,#dH/dt
        delta*I + delta*H+delta*A ,#dR/dt
        phi*I#(1-p)*alpha*E+ p*alpha*E # Casos acumulados
    ]


def seqiahr2(y,t,*params):
    '''
    Modelo sem transmissão de Hospitalizados.
    '''
    S,E,I,A,H,R,C = y
    chi,phi,beta,rho,delta,alpha,p,q = params
    lamb = beta*S*(I+A+(1-rho)*H)
    chi *= (1+np.tanh(t-q))/2 #Liga a quarentena dia q
    return[
        -lamb*(1-chi)*S, #dS/dt
        lamb*(1-chi)*S - alpha*E,#dE/dt
        (1-p)*alpha*E - delta*I,#dI/dt
        p*alpha*E - delta*A,
        phi*delta*I -delta*H,#dH/dt
        (1-phi)*delta*I + delta*H+delta*A ,#dR/dt
        phi*I#(1-p)*alpha*E+ p*alpha*E # Casos acumulados
    ]



def r0(chi, phi, beta, rho, delta, p, S=1):
    "R0 for seqiahr"
    rzero = ((S * beta * chi - S * beta -
              (S * beta * chi - S * beta) * p) * phi * rho -
             (S * beta * chi - S * beta) * delta -
             (S * beta * chi - S * beta) * phi) / (delta ** 2 + delta * phi)
    return rzero


def r02(chi, phi, beta, rho, delta, p, S=1):
    "R0 for seqiahr2"
    return -(S*beta*chi - S*beta)/delta
