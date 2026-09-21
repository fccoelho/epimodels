# -----------------------------------------------------------------------------
# Name:        gillespie.py
# Project:  Bayesian-Inference
# Purpose:     
#
# Author:      Flávio Codeço Coelho<fccoelho@gmail.com>
#
# Created:     2008-11-26
# Copyright:   (c) 2008 by the Author
# Licence:     GPL
# -----------------------------------------------------------------------------
__docformat__ = "restructuredtext en"

import warnings

warnings.warn(
    "epimodels.stochastic.CTMC.gillespie is a legacy module. "
    "Use epimodels.stochastic.CTMC.models (CTMCModel, SIR, etc.) "
    "and epimodels.stochastic.CTMC.solvers (GillespieSolver) instead.",
    DeprecationWarning,
    stacklevel=2,
)

import copy
import time
from collections.abc import Callable
from multiprocessing import Pool, cpu_count
from typing import Any

import numpy as np
from numpy import arange, array, ceil, log, ndarray, zeros
from numpy.random import multinomial, random

viz: bool = False


def dispatch(model)-> Any:
    '''this function is necessary for paralelization'''
    # ~ model.server = server
    return model.GSSA()


class Model:
    pv0: np.ndarray
    vn: list[str]
    res: np.ndarray
    steps: int
    evseries: list[list[Any]]
    tm: np.ndarray

    def __init__(self, vnames: list[str], rates: list[float], inits: list[int], tmat: np.ndarray, propensity: list[Callable]) -> None:
        '''
        Class representing a Stochastic Differential equation.
        
        :Parameters:
            - `vnames`: list of strings
            - `rates`: list of fixed rate parameters
            - `inits`: list of initial values of variables. Must be integers
            - `tmat`: Transition matrix; numpy array with shape=(len(inits),len(propensity))
            - `propensity`: list of lambda functions of the form: lambda r,ini: some function of rates ans inits.
        '''
        # check types
        inits = [int(i) for i in inits]

        assert tmat.dtype == 'int64'
        self.vn = vnames
        self.rates = tuple(rates)
        self.inits = tuple(inits)
        self.tm = tmat
        self.pv = propensity  # [compile(eq,'errmsg','eval') for eq in propensity]
        self.pvl: int = len(self.pv)  # length of propensity vector
        self.pv0 = zeros(self.pvl, dtype=float)
        self.nvars: int = len(self.inits)  # number of variables
        self.evseries = []  # dictionary with complete time-series for each event type.
        self.steps = 0
        self.viz = False  # if intermediate vizualization should be on

    def getStats(self)-> tuple[np.ndarray, ndarray[Any, np.dtype[Any]], int, list[list[float]]]:
        return self.time, self.series, self.steps, self.evseries

    def run(self, method: str = 'SSA', tmax: int = 10, reps: int = 1, viz: bool = False, serial: bool = False)-> None:
        '''
        Runs the model.
        
        :Parameters:
            - `method`: String specifying the solving algorithm. Currently only 'SSA'
            - `tmax`: duration of the simulation.
            - `reps`: Number of replicates.
            - `viz`: Boolean. Whether to show graph of each replicate during simulation
            - `serial`: Boolean. False to run replicate in parallel when more than one core is a vailable. True to run them serially (easier to debug).

        :Return:
            a numpy array of shape (reps,tmax,nvars)
        '''
        self.tmax: int = tmax
        self.res = zeros((tmax, self.nvars), dtype=float)
        tvec: np.ndarray = arange(tmax, dtype=int)

        if method == 'SSA':
            if not serial:  # Parallel version
                pool = Pool(processes=min(reps, cpu_count()))
                res = pool.map(dispatch, [self] * reps, chunksize=10)
                self.res = array([i[0] for i in res])
                if reps == 0:
                    self.evseries = res[0][1]
                else:
                    self.evseries = [i[1] for i in res]
                pool.close()
                pool.join()
            else:  # Serial
                # res = list(map(dispatch, [self] * reps))
                res = [self.GSSA() for i in range(reps)]
                self.res = array([i[0] for i in res])
                if reps == 0:
                    self.evseries = res[0][1]
                else:
                    self.evseries = [i[1] for i in res]

        elif method == 'SSAct':
            pass

        self.time: np.ndarray = tvec
        self.series = self.res
        # self.steps=steps

    def GSSA(self)-> tuple[np.ndarray, dict[int, list[float]]]:
        '''
        Gillespie Direct algorithm
        '''
        tmax = self.tmax
        ini = array(self.inits)
        r = array(self.rates)
        r = np.where(r < 0, 0, r)
        pvi = self.pv  # propensity functions
        tm = self.tm
        pv = self.pv0  # actual propensity values for each time step
        tc: float = 0.0  # current time
        last_tim = 0  # first time step of results
        evts: dict[int,list[float]] = dict([(i, []) for i in range(len(self.pv))])
        self.steps = 0
        res = copy.deepcopy(self.res)
        res[0, :] = ini
        #        for tim in xrange(1,tmax):
        while tc <= tmax:
            i = 0
            a0 = 0.0
            for p in pvi:
                pv[i] = p(r, ini)
                a0 += pv[i]
                i += 1

            if pv.any():  # no change in state if pv is all zeros
                tau = (-1 / a0) * log(random())
                tc += tau
                tim = int(ceil(tc))
                # try:
                event = multinomial(1, pv / a0) # event which will happen on this iteration
                # except ValueError as exc:
                #     print(min(pv), exc, tc)
                #     continue

                e = event.nonzero()[0][0]
                ini += tm[:, e]
                if tc <= tmax:
                    evts[e].append(tc)
                # print tc, ini
                if tim <= tmax - 1:
                    self.steps += 1
                    #                if a0 == 0: break
                    if tim - last_tim > 1:
                        for j in range(last_tim, tim):
                            res[j, :] = res[last_tim, :]
                    res[tim, :] = ini
                else:
                    for j in range(last_tim, tmax):
                        res[j, :] = res[last_tim, :]
                    break
                last_tim = tim
            # self.evseries = evts
            if a0 == 0:
                break  # breaks when no event has prob above 0

        return res, evts


def p1(r: np.ndarray, ini: np.ndarray) -> float:
    return r[0] * ini[0] * ini[1]


def p2(r:np.ndarray, ini:np.ndarray) -> float:
    return r[1] * ini[1]


def main():
    vnames = ['S', 'I', 'R']
    ini = [500, 1, 0]
    rates = [.001, .1]
    tm = array([[-1, 0], [1, -1], [0, 1]])
    # prop=[lambda r, ini:r[0]*ini[0]*ini[1],lambda r,ini:r[0]*ini[1]]
    M = Model(vnames=vnames, rates=rates, inits=ini, tmat=tm, propensity=[p1, p2])
    t0: float = time.time()
    M.run(tmax=80, reps=1000, viz=False, serial=True)
    print('total time: ', time.time() - t0)
    t, series, steps, evts = M.getStats()
    ser = series.mean(axis=0)
    # print evts, len(evts[0])
    from matplotlib.pyplot import legend, plot, show
    plot(t, ser, '-.')
    legend(M.vn, loc=0)
    show()


if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()',sort=1,filename='gillespie.profile')
    main()
