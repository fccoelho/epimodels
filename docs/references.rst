.. _references:

**********
References
**********

This page lists the bibliographic sources describing the methodologies
implemented in **Epimodels**. Citations are grouped by topic. Where
available, a Wikipedia entry is linked as a gentle introduction; the
cited articles and books are the authoritative descriptions of each
method.

Epidemic Models: Foundations
============================

**Kermack, W. O., & McKendrick, A. G. (1927).**
A contribution to the mathematical theory of epidemics.
*Proceedings of the Royal Society of London. Series A*, 115(772), 700–721.
doi:`10.1098/rspa.1927.0118 <https://doi.org/10.1098/rspa.1927.0118>`_
-- The original SIR compartmental model.
See also `Compartmental models in epidemiology (Wikipedia)
<https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology>`_.

**Anderson, R. M., & May, R. M. (1991).**
*Infectious Diseases of Humans: Dynamics and Control.*
Oxford University Press.
-- The standard textbook on compartmental epidemic models (SIR, SIS, SEIR,
SIRS and age/multi-group extensions).

**Diekmann, O., & Heesterbeek, J. A. P. (2000).**
*Mathematical Epidemiology of Infectious Diseases: Model Building,
Analysis and Interpretation.* Wiley.
-- Model building and analysis, including the next-generation method for R0.

**Daley, D. J., & Gani, J. (1999).**
*Epidemic Modelling: An Introduction.* Cambridge University Press.
-- General introduction to deterministic and stochastic epidemic modelling.

**Isham, V., & Medley, G. (Eds.) (1996).**
*Models for Infectious Human Diseases: Their Structure and Relation to
Data.* Cambridge University Press.
-- Structured (heterogeneous) population epidemic models.

**Keeling, M. J., & Rohani, P. (2008).**
*Modeling Infectious Diseases in Humans and Animals.* Princeton
University Press.
-- Comprehensive treatment of compartmental and multi-strain/vector-borne
models. See also `Basic reproduction number (Wikipedia)
<https://en.wikipedia.org/wiki/Basic_reproduction_number>`_.

Discrete-Time (Difference Equation) Models
==========================================

**Finkenstädt, B. F., & Grenfell, B. T. (2000).**
Time series modelling of childhood diseases: a dynamical systems approach.
*Journal of the Royal Statistical Society: Series C (Applied Statistics)*,
49(2), 187–205. doi:`10.1111/1467-9876.00187 <https://doi.org/10.1111/1467-9876.00187>`_
-- The discrete-time (TSIR) formulation with the mixing parameter
:math:`\alpha` used in the :mod:`epimodels.discrete` models.

**Bjørnstad, O. N., Finkenstädt, B. F., & Grenfell, B. T. (2002).**
Dynamics of measles epidemics: estimating scaling of transmission rates
using a time series SIR model. *Ecological Monographs*, 72(2), 169–184.
doi:`10.1890/0012-9615(2002)072[0169:DOMEES]2.0.CO;2 <https://doi.org/10.1890/0012-9615(2002)072[0169:DOMEES]2.0.CO;2>`_
-- Estimation of transmission-rate scaling in discrete-time epidemic models.

**Glass, K., Kappey, J., & Grenfell, B. T. (2004).**
The effect of heterogeneity in measles vaccination on population immunity.
*Epidemiology and Infection*, 132(4), 675–683.
doi:`10.1017/S0950268804002080 <https://doi.org/10.1017/S0950268804002080>`_
-- Waning of vaccine-induced immunity, motivating partial-immunity (SIpR-like)
model variants.

**Longini, I. M., Halloran, M. E., Nizam, A., & Yang, Y. (2004).**
Containing pandemic influenza with antiviral agents.
*American Journal of Epidemiology*, 159(7), 623–633.
doi:`10.1093/aje/kwh092 <https://doi.org/10.1093/aje/kwh092>`_
-- Age-structured influenza transmission models with clinical and
subclinical infections, as in the discrete :class:`~epimodels.discrete.models.Influenza`
model.

COVID-19 / Quarantine Models (SEQIAHR)
======================================

**Nana-Kyere, S., Boateng, F. A., Jonathan, P., Donkor, A., Hoggar, G. K.,
Titus, B. D., Kwarteng, D., & Adu, I. K. (2022).**
Global analysis and optimal control model of COVID-19.
*Computational and Mathematical Methods in Medicine*, 2022, 9491847.
doi:`10.1155/2022/9491847 <https://doi.org/10.1155/2022/9491847>`_
-- A SEQIAHR compartmental model of COVID-19 with quarantine, asymptomatic
and hospitalized compartments. The :class:`~epimodels.continuous.models.SEQIAHR`
implementation follows this class of models, with the quarantine effect
represented by a time-dependent reduction of the force of infection
(:math:`\chi(t)`), switched on at day *q* and off after *r* days via a
smooth tanh transition.

Multi-Strain Models (Dengue, Cross-Immunity)
============================================

**Bartley, L. M., Donnelly, C. A., & Garnett, G. P. (2002).**
The seasonal pattern of dengue in endemic areas: mathematical models of
mechanisms. *Transactions of the Royal Society of Tropical Medicine and
Hygiene*, 96(4), 387–397.
doi:`10.1016/S0035-9203(02)90371-8 <https://doi.org/10.1016/S0035-9203(02)90371-8>`_
-- The sequential multi-serotype dengue model structure (ordered primary,
secondary, ... infections with temporary cross-immunity) used by
:class:`~epimodels.continuous.models.Dengue4Strain`.

**Aguiar, M., Ballesteros, S., Kooi, B. W., & Stollenwerk, N. (2011).**
The role of seasonality and import in a minimalistic multi-strain dengue
model capturing differences between primary and secondary infections:
complex dynamics and its implications for data analysis.
*Journal of Theoretical Biology*, 289, 181–196.
doi:`10.1016/j.jtbi.2011.08.043 <https://doi.org/10.1016/j.jtbi.2011.08.043>`_
-- The two-strain model with temporary cross-immunity, import and seasonality
from which :class:`~epimodels.continuous.models.SIR2Strain` is adapted.

Vector-Borne Models with Climate Forcing (Malaria/Dengue)
=========================================================

**Parham, P. E., & Michael, E. (2010).**
Modeling the effects of weather and climate change on malaria transmission.
*Environmental Health Perspectives*, 118(5), 620–626.
doi:`10.1289/ehp.0901256 <https://doi.org/10.1289/ehp.0901256>`_
-- Climate-driven vector-host model (human SIR × mosquito SEI) with
temperature/rainfall forcing of mosquito demography, the basis of
:class:`~epimodels.continuous.models.SIRSEI` and
:class:`~epimodels.continuous.models.SEIRS_SEI`. A companion book chapter:
Parham & Michael (2010), *Advances in Experimental Medicine and Biology*
673, 184–199.

**Smith, D. L., Battle, K. E., Hay, S. I., Barker, C. M., Scott, T. W., &
McKenzie, F. E. (2012).**
Ross, Macdonald, and a theory for the dynamics and control of
mosquito-transmitted pathogens. *PLoS Pathogens*, 8(4), e1002588.
doi:`10.1371/journal.ppat.1002588 <https://doi.org/10.1371/journal.ppat.1002588>`_
-- The Ross–Macdonald framework underlying R0 for vector-borne models.

Heterogeneous Susceptibility
============================

**Neipel, J., Bauermann, J., Bo, S., Harmon, T., & Jülicher, F. (2020).**
Power-law population heterogeneity governs epidemic waves.
*PLoS ONE*, 15(10), e0239678.
doi:`10.1371/journal.pone.0239678 <https://doi.org/10.1371/journal.pone.0239678>`_
(arXiv:`2008.00471 <https://arxiv.org/abs/2008.00471>`_)
-- SIR generalization with power-law heterogeneous susceptibility
(S(t) = (N−I0)(1+τ/α)^{−α}), implemented as
:class:`~epimodels.continuous.models.NeipelHeterogeneousSIR`.

Ebola Models with Vaccination
=============================

**Legrand, J., Grais, R. F., Boelle, P. Y., Valleron, A. J., & Flahault,
A. (2007).**
Understanding the dynamics of Ebola epidemics.
*Epidemiology and Infection*, 135(4), 610–621.
doi:`10.1017/S0950268806007217 <https://doi.org/10.1017/S0950268806007217>`_
-- The SEIHFR model with community, hospital and funeral transmission,
the basis of :class:`~epimodels.continuous.models.EbolaSEIHFRV`.

**Henao-Restrepo, A. M., et al. (2017).**
Efficacy and effectiveness of an rVSV-vectored vaccine in preventing Ebola
virus disease: final results from the Guinea ring vaccination,
open-label, cluster-randomised trial (Ebola Ça Suffit!).
*The Lancet*, 389(10068), 505–518.
doi:`10.1016/S0140-6736(16)32621-6 <https://doi.org/10.1016/S0140-6736(16)32621-6>`_
-- Evidence base for the ring-vaccination (rVSV-ZEBOV) component of
:class:`~epimodels.continuous.models.EbolaSEIHFRV`.

Stochastic Models (CTMC) and the Gillespie Algorithm
====================================================

**Gillespie, D. T. (1976).**
A general method for numerically simulating the stochastic time evolution
of coupled chemical reactions. *Journal of Computational Physics*, 22(4),
403–434. doi:`10.1016/0021-9991(76)90041-3 <https://doi.org/10.1016/0021-9991(76)90041-3>`_

**Gillespie, D. T. (1977).**
Exact stochastic simulation of coupled chemical reactions.
*The Journal of Physical Chemistry*, 81(25), 2340–2361.
doi:`10.1021/j100540a008 <https://doi.org/10.1021/j100540a008>`_
-- The Direct Method (SSA) implemented in
:class:`~epimodels.stochastic.CTMC.solvers.GillespieSolver`.
See also `Gillespie algorithm (Wikipedia)
<https://en.wikipedia.org/wiki/Gillespie_algorithm>`_.

**Allen, L. J. S. (2017).**
A primer on stochastic epidemic models: formulation, numerical simulation,
and analysis. *Infectious Disease Modelling*, 2(2), 128–142.
doi:`10.1016/j.idm.2017.03.001 <https://doi.org/10.1016/j.idm.2017.03.001>`_
-- CTMC formulations of the SIR/SIS/SEIR/SIRS epidemic models and their
simulation, the methodology of the :mod:`epimodels.stochastic.CTMC` module.

**Andersson, H., & Britton, T. (2000).**
*Stochastic Epidemic Models and Their Statistical Analysis.* Springer.
-- Rigorous treatment of stochastic epidemic processes.

Numerical Solvers (ODE Integration)
===================================

**Dormand, J. R., & Prince, P. J. (1980).**
A family of embedded Runge–Kutta formulae.
*Journal of Computational and Applied Mathematics*, 6(1), 19–26.
doi:`10.1016/0771-050X(80)90013-3 <https://doi.org/10.1016/0771-050X(80)90013-3>`_
-- DOPRI5, the basis of SciPy's ``RK45`` and diffrax's ``Dopri5``.
See also `Runge–Kutta methods (Wikipedia)
<https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods>`_.

**Tsitouras, C. (2011).**
Optimized Runge–Kutta methods for the solution of the orbital problem.
*Applied Numerical Mathematics*, 61(2), 158–169.
doi:`10.1016/j.apnum.2010.09.005 <https://doi.org/10.1016/j.apnum.2010.09.005>`_
-- The Tsit5 solver exposed by :class:`~epimodels.solvers.DiffraxSolver`.

**Hairer, E., Nørsett, S. P., & Wanner, G. (1993).**
*Solving Ordinary Differential Equations I: Nonstiff Problems* (2nd ed.).
Springer.
-- Explicit RK methods including DOP853 (Hairer's 8th-order method).

**Hairer, E., & Wanner, G. (1996).**
*Solving Ordinary Differential Equations II: Stiff and
Differential-Algebraic Problems* (2nd ed.). Springer.
-- Implicit methods for stiff systems: Radau IIA and BDF, exposed by
:class:`~epimodels.solvers.ScipySolver`.

**Petzold, L. (1983).**
Automatic selection of methods for solving stiff and nonstiff systems of
ordinary differential equations. *SIAM Journal on Scientific and
Statistical Computing*, 4(1), 136–159.
doi:`10.1137/0904010 <https://doi.org/10.1137/0904010>`_
-- The LSODA automatic stiffness-switching method.

Parameter Inference (Model Fitting)
===================================

**King, A. A., Ionides, E. L., Bretó, C. M., Ellner, S. P., & Kendall,
B. E. (2015).**
Avoidable errors in the modelling of outbreaks of emerging pathogens, with
special reference to Ebola. *Proceedings of the Royal Society B*,
282(1806), 20150347. doi:`10.1098/rspb.2015.0347 <https://doi.org/10.1098/rspb.2015.0347>`_
-- Best practices for likelihood-based epidemic model fitting (choice of
loss/likelihood functions such as Poisson and negative binomial).

**Huber, P. J. (1964).**
Robust estimation of a location parameter. *The Annals of Mathematical
Statistics*, 35(1), 73–101.
doi:`10.1214/aoms/1177703732 <https://doi.org/10.1214/aoms/1177703732>`_
-- The Huber loss, robust to outliers, implemented as
:class:`~epimodels.fitting.HuberLoss`. See also `Huber loss (Wikipedia)
<https://en.wikipedia.org/wiki/Huber_loss>`_.

**Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995).**
A limited memory algorithm for bound constrained optimization.
*SIAM Journal on Scientific Computing*, 16(5), 1190–1208.
doi:`10.1137/0916069 <https://doi.org/10.1137/0916069>`_
-- L-BFGS-B, used by :class:`~epimodels.fitting.ScipyOptimizer`.

**Nelder, J. A., & Mead, R. (1965).**
A simplex method for function minimization. *The Computer Journal*, 7(4),
308–313. doi:`10.1093/comjnl/7.4.308 <https://doi.org/10.1093/comjnl/7.4.308>`_
-- The Nelder–Mead simplex method.

**Storn, R., & Price, K. (1997).**
Differential evolution – a simple and efficient heuristic for global
optimization over continuous spaces. *Journal of Global Optimization*, 11,
341–359. doi:`10.1023/A:1008202821328 <https://doi.org/10.1023/A:1008202821328>`_
-- Differential evolution, available in :class:`~epimodels.fitting.ScipyOptimizer`.

**Kingma, D. P., & Ba, J. (2015).**
Adam: a method for stochastic optimization.
*ICLR 2015* (arXiv:`1412.6980 <https://arxiv.org/abs/1412.6980>`_)
-- The Adam optimizer exposed by :class:`~epimodels.fitting.JAXOptimizer`.
See also `Stochastic gradient descent (Wikipedia)
<https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam>`_.

**McKay, M. D., Beckman, R. J., & Conover, W. J. (1979).**
A comparison of three methods for selecting values of input variables in
the analysis of output from a computer code. *Technometrics*, 21(2),
239–245. doi:`10.1080/00401706.1979.10489755 <https://doi.org/10.1080/00401706.1979.10489755>`_
-- Latin hypercube sampling used for multi-start initialization.

**Sobol', I. M. (1967).**
On the distribution of points in a cube and the approximate evaluation of
integrals. *USSR Computational Mathematics and Mathematical Physics*, 7(4),
86–112. doi:`10.1016/0041-5553(67)90144-9 <https://doi.org/10.1016/0041-5553(67)90144-9>`_
-- Sobol' sequences used for multi-start initialization.

**Raue, A., Kreutz, C., Maiwald, T., Bachmann, J., Schilling, M.,
Klingmüller, U., & Timmer, J. (2009).**
Structural and practical identifiability analysis of partially observed
dynamical models by exploiting the profile likelihood.
*Bioinformatics*, 25(15), 1923–1929.
doi:`10.1093/bioinformatics/btp358 <https://doi.org/10.1093/bioinformatics/btp358>`_
-- The profile-likelihood method used for confidence intervals by
:meth:`~epimodels.fitting.ModelFitter.profile_likelihood`.

Phase Space Analysis (Nonlinear Time Series)
============================================

**Takens, F. (1981).**
Detecting strange attractors in turbulence.
In *Dynamical Systems and Turbulence*, Lecture Notes in Mathematics 898,
366–381. Springer. doi:`10.1007/BFb0091924 <https://doi.org/10.1007/BFb0091924>`_
-- Takens' embedding theorem, implemented in
:class:`~epimodels.tools.phase.TimeDelayEmbedding`.
See also `Takens' theorem (Wikipedia)
<https://en.wikipedia.org/wiki/Takens%27s_theorem>`_.

**Fraser, A. M., & Swinney, H. L. (1986).**
Independent coordinates for strange attractors from mutual information.
*Physical Review A*, 33(2), 1134–1140.
doi:`10.1103/PhysRevA.33.1134 <https://doi.org/10.1103/PhysRevA.33.1134>`_
-- Selection of the time delay τ from the first local minimum of the
(average) mutual information.

**Cao, L. (1997).**
Practical method for determining the minimum embedding dimension of a
scalar time series. *Physica D: Nonlinear Phenomena*, 110(1–2), 43–50.
doi:`10.1016/S0167-2789(97)00118-8 <https://doi.org/10.1016/S0167-2789(97)00118-8>`_
-- The E1-statistic method implemented in
:meth:`~epimodels.tools.phase.TimeDelayEmbedding.cao_embedding_dimension`.

**Kantz, H., & Schreiber, T. (2004).**
*Nonlinear Time Series Analysis* (2nd ed.). Cambridge University Press.
-- General reference for phase space reconstruction methodology.

R0, Stability and Sensitivity (Symbolic Analysis)
=================================================

**Diekmann, O., Heesterbeek, J. A. P., & Metz, J. A. J. (1990).**
On the definition and the computation of the basic reproduction ratio R0
in the description of infectious disease in heterogeneous populations.
*Journal of Mathematical Biology*, 28(4), 365–382.
doi:`10.1007/BF00178324 <https://doi.org/10.1007/BF00178324>`_
-- The next-generation matrix/operator definition of R0, implemented in
:meth:`epimodels.validation.SymbolicModel.compute_R0_next_generation`.

**van den Driessche, P., & Watmough, J. (2002).**
Reproduction numbers and sub-threshold endemic equilibria for
compartmental models of disease transmission.
*Mathematical Biosciences*, 180(1–2), 29–48.
doi:`10.1016/S0025-5564(02)00108-6 <https://doi.org/10.1016/S0025-5564(02)00108-6>`_
-- The F–V matrix computation of R0 for compartmental ODE models.

**Chitnis, N., Hyman, J. M., & Cushing, J. M. (2008).**
Determining important parameters in the spread of malaria through the
sensitivity analysis of a mathematical model.
*Bulletin of Mathematical Biology*, 70(5), 1272–1296.
doi:`10.1007/s11538-008-9299-0 <https://doi.org/10.1007/s11538-008-9299-0>`_
-- Normalized forward sensitivity indices of R0, implemented in
:meth:`epimodels.validation.SymbolicModel.sensitivity_analysis` and
:meth:`epimodels.validation.SymbolicModel.elasticity_analysis`.

See also `Next-generation matrix (Wikipedia)
<https://en.wikipedia.org/wiki/Next-generation_matrix>`_.
