from pkg_resources import get_distribution, DistributionNotFound
import logging

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound

from matplotlib import pyplot as P

class BaseModel:
    """
    Base class for all models
    """
    def __init__(self):
        self.name = None
        self.model_type = None
        self.state_variables = {}
        self.parameters = {}
        self.traces = {}

    def plot_traces(self):
        for series, data in self.traces.items():
            if series == 'time':
                continue
            P.plot(self.traces['time'], data, label=series)
        P.legend(loc=0)
        P.title("{} model".format(self.model_type))
