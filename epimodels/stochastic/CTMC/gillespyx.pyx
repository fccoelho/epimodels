# cython: profile=False
# cython.boundscheck = False
# cython.cdivision = True
# cython.wraparound = False
#

import numpy as np
cimport numpy as cnp
import time
cimport cython

cnp.import_array()

DTYPE = np.double
INT = np.int32
UITYPE = np.uint32
ctypedef cnp.double_t DTYPE_t
ctypedef cnp.int32_t INT_t
ctypedef cnp.uint32_t UITYPE_t

cdef extern from "math.h":
    double log(double)

@cython.profile(False)
cdef inline double clog(double x):
    return log(x)

cdef extern from "stdlib.h":
    double RAND_MAX
    double c_libc_random "random"()
    void c_libc_srandom "srandom"(unsigned int seed)

cdef extern from "/usr/include/gsl/gsl_rng.h":
    ctypedef struct gsl_rng_type
    ctypedef struct gsl_rng
    
    cdef gsl_rng_type *gsl_rng_mt19937
    gsl_rng *gsl_rng_alloc(gsl_rng_type * T) nogil

cdef extern from "/usr/include/gsl/gsl_randist.h":
    void gsl_ran_multinomial "gsl_ran_multinomial"(gsl_rng * r, size_t K,
                                            UITYPE_t N, double p[],
                                            UITYPE_t n[]) nogil

cdef gsl_rng *R = gsl_rng_alloc(gsl_rng_mt19937)

@cython.profile(False)
cdef inline double crandom():
    cdef double rm = RAND_MAX
    return c_libc_random()/rm

cdef inline UITYPE_t gsl_multinomial(cnp.ndarray[DTYPE_t, ndim=1] p, UITYPE_t N):
    cdef:
       size_t K = p.shape[0]
       cnp.ndarray[UITYPE_t, ndim=1] n = np.zeros(p, dtype=UITYPE)


    # void gsl_ran_multinomial (const gsl_rng * r, size_t K, unsigned int N, const double p[], unsigned int n[])
    gsl_ran_multinomial(R, K, N, &p[0], &n[0])

    return n

cdef class Model(object):
    cdef object vn,pv
    cdef cnp.ndarray tm,res,time,series, rates, inits
    cdef UITYPE_t pvl,nvars,steps
    cdef object ts
    def __init__(self,vnames,rates,inits, tmat,propensity):
        '''
         * vnames: list of strings
         * rates: list of fixed rate parameters
         * inits: list of initial values of variables
         * propensity: list of lambda functions of the form: 
            lambda r,ini: some function of rates ans inits.
        '''
        self.vn = vnames
        self.rates = rates
        self.inits = inits
        self.tm = tmat
        self.pv = propensity#[compile(eq,'errmsg','eval') for eq in propensity]
        self.pvl = len(self.pv) #length of propensity vector
        self.nvars = len(self.inits) #number of variables
        self.time = np.zeros(1)
        self.series = np.zeros(1)
        self.steps = 0

    cpdef run(self, method='SSA', int tmax=10, int reps=1):
        cdef cnp.ndarray[INT_t,ndim=3] res = np.zeros((tmax,self.nvars,reps),dtype=INT)
        cdef cnp.ndarray[INT_t,ndim=1] tvec = np.arange(tmax,dtype=INT)
        self.res = res
        cdef int i, steps
        if method =='SSA':
            for i from 0 <= i < reps:
                steps = self.GSSA(tmax,i)
            #print steps,' steps'
        elif method == 'SSAct':
            pass
        self.time=tvec
       # self.series=self.res
        self.steps=steps
    
    def getStats(self):
        return self.time,self.res,self.steps

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline int GSSA(self, unsigned int tmax=50, unsigned int round=0) except *:
        '''
        Gillespie Direct algorithm
        '''
        cdef cnp.ndarray[INT_t, ndim=1] ini = np.array(self.inits, dtype=INT)
        cdef cnp.ndarray[DTYPE_t, ndim=1] r = self.rates
        pvi = self.pv #propensity functions
        cdef int l,steps,i, tim, last_tim
        cdef double tc, tau, a0
        #cdef np.ndarray[INT_t] tvec
        l=self.pvl
        cdef cnp.ndarray[DTYPE_t,ndim=1,mode="c"] pv = np.zeros(l,dtype=DTYPE)

        cdef:
            size_t K = l
            cnp.ndarray[UITYPE_t, ndim=1] n = np.zeros(l, dtype=UITYPE)
        
        tm = self.tm
        #tvec = np.arange(tmax,dtype=int)
        tc = 0
        steps = 0
        self.res[0,:,round]= ini
        a0=1.
        for tim from 1<= tim < tmax:
            while tc < tim:
                a0 = 0.0
                for i from 0<= i < l:
                    pv[i] = pvi[i](r,ini)
                    a0 += pv[i]
                
                if pv.any():
                    tau = (-1/a0)*clog(crandom())
                    pv /= a0
                    #event = gsl_multinomial(probs,1)
                    gsl_ran_multinomial(R, K, 1, &pv[0], &n[0])
                    #event = np.random.multinomial(1,(pv/a0)) # event which will happen on this iteration
                    ini += tm[:,n.nonzero()[0][0]]
                    tc += tau
                steps +=1
                if a0 == 0: break
            self.res[tim,:,round] = ini
            #if a0 == 0: break
#        tvec = tvec[:tim]
#        self.res = res[:tim,:,round]
        return steps

    def CR(self,pv):
        """
        Composition reaction algorithm
        """
        pass

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef inline DTYPE_t l1(cnp.ndarray[DTYPE_t ,ndim=1] r,cnp.ndarray[INT_t,ndim=1] ini):
    return r[0]*ini[0]*ini[1]
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef inline DTYPE_t l2(cnp.ndarray[DTYPE_t,ndim=1] r,cnp.ndarray[INT_t,ndim=1] ini):
    return r[1]*ini[1]

cpdef main():
    vars = ['s','i','r']
    cdef:
        cnp.ndarray[INT_t,ndim=1] ini= np.array([500,1,0], dtype=INT)
        cnp.ndarray[DTYPE_t,ndim=1] rates = np.array([.001,.1],dtype=DTYPE)
        cnp.ndarray[INT_t,ndim=2] tm = np.array([[-1,0],[1,-1],[0,1]],dtype=INT)
    prop = [l1,l2]
    M = Model(vnames = vars,rates = rates,inits=ini, tmat=tm,propensity=prop)
    t0=time.time()
    M.run(tmax=80,reps=1000)
    print('total time: ',time.time()-t0)
    return M.getStats()



