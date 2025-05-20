**************
Model Overview
**************
Models of disease dynamics can be quite diverse, ranging from caricatures
to very detailed simulations. Traditional models of spread of diseases
are based on the mean field assumption, i.e., that individuals
interact randomly at a certain rate. Important references to the
subject are
\cite{DiekmannOandHeesterbeekJAP2000,DaleyDJandGaniJandCanningsC2001,IshamVandMedleyG1996,AndersonRMandMayRMandAndersonB1992}.
These models are expressed mathematically as difference equations
(discrete time) or differential equations (continuous time). In the
simplest form, these models do not take into consideration either
individual heterogeneity or the local nature of transmission
events. Increased realism is achieved by structuring the population according to age, risk behavior, sex, susceptibility, or other category associated with different risk of getting or transmitting the disease. Within each sub-population, however, the assumption of well mixing must hold. When other species are involved in the transmission process (non-human hosts and vectors), these are also considered as compartments that may be sub-divided as well according to covariates associated with the risk of acquiring or transmitting the disease.

In this context, epidemiological models take the form of multi-compartmental models where each compartment is a well-mixed homogeneous population. The model describes the transition of the individuals in this population through a sequence of disease-related stages. These stages could be *Susceptible*, *Infected*, *Recovered*, for example. And the transitions could be

.. math:: Susceptible \longrightarrow Infected
.. math:: Infected \longrightarrow Recovered

If only these transitions are allowed, then individuals in the recovered class never become susceptible again (lifelong immunity). If, on the other hand, immunity is only temporary (as in pertussis), then another transition should be included:

.. math:: Susceptible \longrightarrow Infected
.. math:: Infected \longrightarrow Recovered
.. math:: Recovered \longrightarrow Susceptible

One way to visualize these models is using state-flow diagrams, where boxes represent states (compartments) and arrows indicate the transitions. It is the state identity together with the transitions allowed that define the type of model in use.

Discrete models
==============

The discrete models implemented in Epimodels, are discrete in time, meaning that they calculate the number of individuals in each compartment, a continuous variable, on time :math:`t+1` as a function the number of individuals in each compartment at time :math:`t`.

In all models presented below, new infections, :math:`L_t` arise from the contact between Susceptibles and Infectious individuals. Mathematically, this is written as:

.. math::  L_{t+1} = \beta S_t \frac{(I_t)^\alpha} {N_t}

where :math:`L_{t+1}` is the number of new cases, :math:`\beta` is the contact rate between Susceptibles and Infectious individuals, :math:`S_t` is the number of susceptibles, :math:`I_t` is the number of infectious individuals, :math:`N_t` is the population size. :math:`\alpha` is a mixing parameter. :math:`\alpha=1` corresponds to homogeneous mixing (Finkenstadt and Grenfell, 2000).


Typology of infectious diseases and corresponding models
--------------------------------------------------------

.. index:: Models;typology

Here we present a brief description of the typology of infectious
diseases models based on the main route of transmission, and type of
immunity resulting from infection. These models correspond to the
types of models built into EpiGrass.

SIR-like models
^^^^^^^^^^^^^^^

The natural history of many directly transmitted infectious diseases can be appropriately described by a SIR-like model.
*SIR* stands for Susceptible :math:`(S)`, Infected :math:`(I)` and Recovered :math:`(R)`. Archetypal *SIRs* are measles
or chickenpox, i.e., diseases that confer lifelong immunity (but see \cite{KGlassandBTGrenfell2004}).
An individual starts his life in the :math:`S` state and may progress to the :math:`I` state. The rate of progression of
individuals from :math:`S` to :math:`I` is called the incidence rate or force of infection :math:`(\lambda)` which is a
function of contact rate, probability of transmission per contact and density of infectious individuals. Individuals
stay in the infectious period for a certain time and then move to the recovered state where they become immune to new
infections. Generally, the removal rate from the infectious class is the inverse of the infectious
period (i.e., it is assumed that the duration of infection is exponentially distributed).

.. _fig:sir:
.. figure:: SIRdiagram.png

    Figure: SIR-like models


Variations of this model allow cases where infected individuals do not acquire immunity after infection, thus returning to the susceptible pool (*SIS* model). Another variation is the inclusion of a latent stage to hold individuals that are infected but not infectious to others yet (incubation period). These are the *SEIR* (with immunity) and *SEIS* (no immunity) models.

Next, we describe in more detail each one of these models in their deterministic and stochastic versions, as used by EpiGrass.

.. tabularcolumns:: |c|l|

+----------------+---------------------------------------------------+
| **Symbol**     | **Meaning**                                       |
+================+===================================================+
| :math:`L_t`    | number of newly infected individuals at time      |
+----------------+---------------------------------------------------+
| :math:`E`      | number of exposed but not infectious individuals  |
|                | at time t                                         |
+----------------+---------------------------------------------------+
| :math:`I`      | number of infectious individuals at time t        |
+----------------+---------------------------------------------------+
| :math:`R`      | number of recovered individuals at time t         |
+----------------+---------------------------------------------------+
| :math:`\beta`  | contact rate (:math:`t^{-1}`)                     |
+----------------+---------------------------------------------------+
| :math:`\theta` | number of infectious visitors                     |
+----------------+---------------------------------------------------+
| :math:`\alpha` | mixing parameter (:math:`\alpha = 1` means        |
|                | homogeneous mixing)                               |
+----------------+---------------------------------------------------+
| :math:`n`      | number of visitors                                |
+----------------+---------------------------------------------------+
| :math:`N`      | population :math:`(S+E+I+R)`                      |
+--------------------------------------------------------------------+
| :math:`B`      | susceptible pool replenishment                    |
+----------------+---------------------------------------------------+
| :math:`r`      | fraction of :math:`I` recovering from infection   |
|                | per unit of time :math:`([0,1])`                  |
+----------------+---------------------------------------------------+
| :math:`e`      | fraction of :math:`E` becoming infectious per     |
|                | unit of time :math:`([0,1])`                      |
+----------------+---------------------------------------------------+
| :math:`\delta` | probability of acquiring immunity :math:`([0,1])` |
+----------------+---------------------------------------------------+
| :math:`w`      | probability of losing immunity :math:`([0,1])`    |
+----------------+---------------------------------------------------+
| :math:`p`      | probability of recovered individual acquiring     |
|                | infection, given exposure :math:`([0,1])`         |
+----------------+---------------------------------------------------+

.. index:: Models;SIR
**SIR models**
    Examples of diseases represented by SIR models are measles, chickenpox. Some diseases that do not confer lifelong immunity may be represented by this model if only short term dynamics is of interest. In the scale of a year, influenza and pertussis, for example, could be described using SIR. The SIR model is implemented in EpiGrass as a system of four difference equations. Besides the three equations describing the dynamics of :math:`S`, :math:`I` and :math:`R`, a fourth equation explicitly defines the number of new cases per time step, :math:`L(t)` (i.e., the incidence). In general, this quantity is embedded in the :math:`I` equation (prevalence), but it is important to keep track of the incidence if one wishes to compare prediction with notification data.

.. math::
    :nowrap:
    :label: E:SIRmodel

    \begin{align}
        L_{t+1} &= \beta S_t \frac{(I_t+\theta)^\alpha} {N_t+n_t}\nonumber \\
        I_{t+1} &= L_{t+1} + (1-r)I_t\nonumber\\
        S_{t+1} &= S_t + B - L_{t+1}\nonumber\\
        R_{t+1} &= N_t-(S_{t+1}+I_{t+1})\nonumber
    \end{align}


This model can be easily extended to include diseases without recovery, for example AIDS, the so called SI models. Basically, the recovery rate is set to 0.

.. index:: Models;SIS

**SIS models**
    In the SIS model, individuals do not acquire immunity after the infection. They return directly to the susceptible class.

    The only difference between $SIS$ and $SIR$ models is the absence of $R$ and the flow of recovered individuals to the susceptible stage:

.. math::
    :nowrap:
    :label: E:SISmodel

    \begin{align}
            L_{t+1} &= \beta S_t \frac{(I_t+\theta)^\alpha} {N_t+n_t} \nonumber\\
            I_{t+1} &= L_{t+1} + (1-r)I_t\nonumber\\
            S_{t+1} &= S_t + B - L_{t+1} + r I_{t+1}\nonumber
    \end{align}

.. index:: Models;SEIR

**SEIR models**
    These models have an extra compartment for those individuals who have acquired the infection but are still not infectious to others. This is the latent period and it is often parameterized as the inverse of the incubation period. Note, however, that for many diseases, initiation of infectiousness does not necessarily coincides with symptoms. In principle, any disease described by the SIR model can also be described by the SEIR model. The decision regarding the use of one or another depends on the magnitude of the latent period in relation to the time frame of other events in the simulation. The model has the form:

.. math::
    :nowrap:
    :label: E:SEIRmodel

    \begin{align}
            L_{t+1} &= \beta S_t \frac{(I_t+\theta)^\alpha} {N_t+n_t}\nonumber \\
        E_{t+1} &= (1-e) E_t + L_{t+1}\nonumber\\
            I_{t+1} &= e E_t + (1-r)I_t\nonumber\\
            S_{t+1} &= S_t + B - L_{t+1}\nonumber\\
            R_{t+1} &= N_t-(S_{t+1}+I_{t+1}+E_{t+1})\nonumber
    \end{align}

.. index:: Models;SEIS

**SEIS models**
    These are SIS models with the inclusion of the latent stage.

.. math::
    :nowrap:
    :label: E:SEISmodel

    \begin{align}
            L_{t+1} &= \beta S_t \frac{(I_t+\theta)^\alpha} {N_t+n_t}\nonumber \\
        E_{t+1} &= (1-e) E_t + L_{t+1}\nonumber\\
            I_{t+1} &= e E_t + (1-r)I_t\nonumber\\
            S_{t+1} &= S_t + B - L_{t+1} + r I_t\nonumber
    \end{align}



SIpR-like models
^^^^^^^^^^^^^^^^
These are *SIR* models with immunity intermediary between full (*SIR*) and null (*SIS*).
Some possibilities arise here: 1) Infection confers full immunity to a fraction of the individuals and the remaining ones return to the susceptible class again, after infection. (*SIpRpS*); 2) Infection provides only partial immunity and recovered individuals are partially susceptible to new infection (*SIpR*); 3) Immunity is full right after infection but wanes with time (*SIRS*). Each model is presented below. Figure :ref:`fig:sipr` illustrates them diagrammatically.

.. _fig:sipr
.. figure:: SIpRdiagram.png

    SIpR-like models.


Related models, that included the latent state $E$ are: \textit{SEIpRpS}, *SEIpR*, *SEIRS*.

.. index:: Models;SIpRpS

**SIpRpS model**
    This model assumes that a fraction $\delta$ of infectious individuals acquire full immunity while the remaining $(1-\delta)$ returns to the susceptible stage. The model is:

.. math::
    :nowrap:
    :label: E:SIpRpSmodel

    \begin{align}
            L_{t+1} &= \beta S_t \frac{(I_t+\theta)^\alpha} {N_t+n_t}\nonumber \\
            I_{t+1} &= L_{t+1} + (1-r)I_t\nonumber\\
            S_{t+1} &= S_t + B - L_{t+1} + (1-\delta) r I_t\nonumber\\
            R_{t+1} &= N_t-(S_{t+1}+I_{t+1})\nonumber
    \end{align}


**SIpR model**
    This model assumes that immunity is only partial and recovered individuals may acquire infection again (at a  lower rate :math:`p \lambda`, where :math:`0\leq p \leq 1`). Two equations calculate the number of new infections. :math:`L_S`  calculates the number of susceptibles that become infected at :math:`t+1`. :math:`L_R`  calculates the number of recovered that become infected at :math:`t+1`. The latter are less susceptible to the disease when compared to susceptibles. The model is:

.. math::
    :nowrap:
    :label: E:SIpRmodel

    \begin{align}
        L_{S,t+1} &= \beta S_t \frac{(I_t+\theta)^\alpha} {N_t+n_t}\nonumber \\
        L_{R,t+1} &= p \beta R_t \frac{(I_t+\theta)^\alpha} {N_t+n_t}\nonumber\\
        I_{t+1} &= L_{S,t+1} + L_{R,t+1} + (1-r)I_t\nonumber\\
        S_{t+1} &= S_t + B - L_{S,t+1} \nonumber\\
        R_{t+1} &= N_t-(S_{t+1}+I_{t+1}) \nonumber
        \end{align}

.. index:: Models;SIRS

**SIRS model**
    Here, the immunity acquired by infection wanes with time. Pertussis is an example of this dynamic.

.. math::
    :nowrap:
    :label: E:SIRSmodel

    \begin{align} \label{}
            L_{S,t+1} &= \beta S_t \frac{(I_t+\theta)^\alpha} {N_t+n_t}\nonumber \\
        I_{t+1} &= L_{S,t+1} + L_{R,t+1} + (1-r)I_t\nonumber\\
            S_{t+1} &= S_t + B - L_{S,t+1} + w R_t\nonumber\\
            R_{t+1} &= N_t-(S_{t+1}+I_{t+1}) \nonumber
    \end{align}





SnInRn-like Models
^^^^^^^^^^^^^^^^^^
These are models with more than one compartment for susceptibles,
infected and recovered stages. They are used when infection involves
more than one distinct populations. Vector borne diseases are
classical examples where a SIR model is used to describe infection in
humans and another SIR-like model is used to describe infection in the
vector (and/or the reservoir(s)). Dengue fever and yellow fever are
examples. Sexually transmitted diseases may also be modeled with
SnInRn models if male and female populations are distinguished. These
models can be define by the user as a custom model.