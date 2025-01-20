import torch
import numpy as np

torch.set_default_dtype(torch.double)

class Integrator(object):
    '''
    Base class for integrator
    '''

    def __init__(self, model):
        self.model = model

    def forward(self, x0, ts, **params):
        if (not torch.is_tensor(x0)):
            x0 = torch.tensor(x0).double()
        if (not torch.is_tensor(ts)):
            ts = torch.tensor(ts).double().squeeze()
        Nt = ts.size(0)
        p = x0.size(0)
        Xs = torch.empty(Nt, p).double()
        Xs[0,:] = x0
        for i in range(Nt-1):
            xt = Xs[i,:]
            dt = ts[i+1] - ts[i]
            Xs[(i+1),:] = self._step_func(xt, dt, **params)
        return (Xs)

    def _step_func(self):
        pass

class Euler(Integrator):
    '''
        Implementation of Euler method
    '''

    def __init__(self, model):
        super().__init__(model)

    def _step_func(self, x, dt, step = 10):
        ss = dt / step # step size
        for j in range(step):
            x = x + ss * self.model(x)
        return (x.detach())

class RungeKutta(Integrator):
    '''
        Implementation of Runge-Kutta method
    '''
    def __init__(self, model):
        super().__init__(model)

    def _step_func(self, x, dt):
        k1 = self.model(x)
        k2 = self.model((x+k1*dt/2.))
        k3 = self.model((x+k2*dt/2.))
        k4 = self.model((x+k3*dt))
        x = x + dt * (k1 + 2. * k2 + 2. * k3 + k4) / 6.
        return (x.detach())
