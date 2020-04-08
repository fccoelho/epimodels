
# Epimodels



This library a simple interface to simulate mathematical epidemic models.
 


## Getting started


Simple SIR simulation

```python
from epimodels.continuous.models import SIR
model = SIR()
model([1000, 1, 0], [0, 50], 1001, {'beta': 2, 'gamma': .1})
model.plot_traces()
```


### Related libraries

For stochastic epidemic models check [this](https://github.com/fccoelho/EpiStochModels).