"""
This module offers tools for phase space representation of
models' dynamics.


Embedding tool is based of the following papers:

1. Basharat, A., & Shah, M. (2009, September). Time series prediction by chaotic modeling of nonlinear dynamical systems. In Computer Vision, 2009 IEEE 12th International Conference on (pp. 1941-1948). IEEE.
2. Cao, L. (1997). Practical method for determining the minimum embedding dimension of a scalar time series. Physica D: Nonlinear Phenomena, 110(1-2), 43-50.

Created on 29/11/18
by fccoelho
license: GPL V3 or Later
"""
# import numpy
# # from pyitlib import discrete_random_variable as drv
# import pylab as P
#
# class TimeDelayEmbedding:
#     def __init__(self, traces):
#         self.traces = traces
#
#     def mutual_information(self, data, tau_max=100):
#         mis = []
#
#         for tau in range(1, tau_max):
#             unlagged = data[:-tau]
#             lagged = numpy.roll(data, -tau)[:-tau]
#             joint = numpy.hstack((unlagged, lagged))
#             mis.append(drv.information_multi(joint))
#             if len(mis) > 1 and mis[-2] < mis[-1]:  # return first local minima
#                 tau -= 1
#         return tau, mis
#
#     def _find_embedding_dimension(self, x, tau):
#         d = 2
#         while True:
#             ai = max([])
#
#     def show_mutual_information(self, mis):
#         P.plot(mis)
#         P.show()
