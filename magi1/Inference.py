import torch
import gpytorch
import numpy as np
import torch.fft as ft
import time

from .Kernal import MaternKernel
from . import integrator
from .grid_interpolation import GridInterpolationKernel
from .FT_helper import FourierGP, Fourier_vetor, EigenGP

torch.set_default_dtype(torch.double)

class KISSGPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, grid, interpolation_orders):
        super(KISSGPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            GridInterpolationKernel(
                MaternKernel(), grid=grid, interpolation_orders=interpolation_orders
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class FTMAGI(object):
    def __init__(self, ys, dynamic, grid_size=201, interpolation_orders=3):
        self.grid_size = grid_size
        self.comp_size = len(ys)
        for i in range(self.comp_size):
            if (not torch.is_tensor(ys[i])):
                ys[i] = torch.tensor(ys[i]).double().squeeze()
        self.ys = ys
        self.fOde = dynamic
        self._kiss_gp_initialization(interpolation_orders=interpolation_orders)

    def _kiss_gp_initialization(self, interpolation_orders=3, training_iterations=100):
        tmin = self.ys[0][:,0].min()
        tmax = self.ys[0][:,0].max()
        for i in range(1, self.comp_size):
            tmin = torch.min(tmin, self.ys[i][:,0].min())
            tmax = torch.max(tmax, self.ys[i][:,0].max())
        spacing = (tmax - tmin) / (self.grid_size - 1)
        padding = int((interpolation_orders + 1) / 2)
        grid_bounds = (tmin - padding * spacing, tmax + padding * spacing)
        self.grid = torch.linspace(grid_bounds[0], grid_bounds[1], self.grid_size+2*padding)
        self.gp_models = []
        for i in range(self.comp_size):
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = KISSGPRegressionModel(self.ys[i][:,0], self.ys[i][:,1], 
                likelihood, self.grid, interpolation_orders)
            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model) # loss function
            for j in range(training_iterations):
                optimizer.zero_grad()
                output = model(self.ys[i][:,0])
                loss = -mll(output, self.ys[i][:,1])
                loss.backward()
                optimizer.step()
            model.eval()
            likelihood.eval()
            self.gp_models.append(model)
        self.grid = self.grid[padding:-padding] # remove extended grid points

    def map(self, max_epoch=1000, 
            learning_rate=1e-3,verbose=False,
            returnX=False,Truncated=False, k = None,z_t = None):
        gpmat = []
        u = torch.empty(self.grid_size, self.comp_size).double()
        x = torch.empty(self.grid_size, self.comp_size).double()
        z = torch.empty(z_t, self.comp_size).double()
        dxdtGP = torch.empty(self.grid_size, self.comp_size).double()

        if Truncated:
            if k is None: # default k
                if self.grid_size % 2 == 1:
                    k = (self.grid_size + 1) // 2
                else:
                    k = self.grid_size // 2
            else: 
                k = k

        with torch.no_grad():
            for i in range(self.comp_size):
                ti = self.ys[i][:,0]
                model = self.gp_models[i]
                mean = model.mean_module.constant.item()
                outputscale = model.covar_module.outputscale.item()
                noisescale = model.likelihood.noise.item()
                nugget = noisescale / outputscale
                grid_kernel = model.covar_module.base_kernel
                base_kernel = grid_kernel.base_kernel
                # compute mean for grid points
                xi = model(self.grid).mean
                LC = base_kernel(self.grid,self.grid).add_jitter(1e-6)._cholesky()
                LCinv = LC.inverse()
                ui = LCinv.matmul(xi-mean) / np.sqrt(outputscale)
                # compute eigen decomposition 
                C_d = base_kernel(self.grid, self.grid).add_jitter(1e-6).evaluate()
                eval_sqrt, evec_tilde = EigenGP(C_d, z_t)
                K = evec_tilde.matmul(eval_sqrt)
                Kinv = eval_sqrt.inverse().matmul(evec_tilde.t())
                zi = Kinv.matmul(xi - mean) / np.sqrt(outputscale) # 161x1
                # compute uq for the grid points
                q = grid_kernel(ti,ti).add_jitter(nugget)._cholesky().inverse().matmul(grid_kernel(ti,self.grid))
                LU = (base_kernel(self.grid,self.grid)-q.t().matmul(q)).add_jitter(1e-6)._cholesky().mul(np.sqrt(outputscale))
                # compute gradient for grid points
                m = LCinv.matmul(base_kernel.dCdx2(self.grid,self.grid)).t()
                dxi = m.matmul(ui) * np.sqrt(outputscale)
                LK = (base_kernel.d2Cdx1dx2(self.grid,self.grid)-m.matmul(m.t())).to_dense()
                LKnew = FourierGP(LK, k)
                LKinv = LKnew.add_jitter(1e-6)._cholesky().inverse()
                m = m.matmul(LCinv)
                m_tilde = m.matmul(K)
                # compute covariance for x|grid
                s = LCinv.matmul(grid_kernel(self.grid,ti))
                LQinv = (grid_kernel(ti,ti).add_jitter(nugget) - s.t().matmul(s)).add_jitter(1e-6)._cholesky().inverse()
                s = s.t().matmul(LCinv)
                s_tilde = s.matmul(K)
                # store information
                u[:,i] = ui
                x[:,i] = xi
                z[:,i] = zi
                dxdtGP[:,i] = dxi
                gpmat.append({'LC':LC,'LCinv':LCinv,'m':m,'LKinv':LKinv,'s':s,'LQinv':LQinv,'LU':LU,'K':K,'Kinv':Kinv,'m_tilde':m_tilde,'s_tilde':s_tilde})      

        # optimizer for u and theta
        state_optimizer = torch.optim.Adam([z], lr=learning_rate)
        theta_optimizer = torch.optim.Adam(self.fOde.parameters(), lr=learning_rate)

        # optimize initial theta
        # attach gradient for theta
        for param in self.fOde.parameters():
            param.requires_grad_(True)
        for tt in range(200):
            xr = x.clone()
            dxrdtOde = self.fOde(xr)
            theta_optimizer.zero_grad()
            lkh = torch.zeros(self.comp_size)
            for i in range(self.comp_size):
                mean = self.gp_models[i].mean_module.constant.item()
                outputscale = self.gp_models[i].covar_module.outputscale.item()
                xr_new = gpmat[i]['m_tilde'].matmul(np.sqrt(outputscale) * z[:,i])
                xr_ft, dxrdtOde_ft = Fourier_vetor(xr_new, k), Fourier_vetor(dxrdtOde[:,i], k)
                dxrdtError = gpmat[i]['LKinv'].matmul(dxrdtOde_ft-xr_ft)
                lkh[i] = -0.5/outputscale * dxrdtError.square().mean()
            theta_loss = -torch.sum(lkh)
            theta_loss.backward(retain_graph=True)
            theta_optimizer.step()

        # detach theta gradient
        for param in self.fOde.parameters():
            param.requires_grad_(False)

        s_time = time.time()

        for epoch in range(max_epoch):
            # optimize u (x after Cholesky decomposition)
            z.requires_grad_(True)
            for st in range(1):
                state_optimizer.zero_grad()
                # reconstruct x
                x = torch.empty_like(u).double()
                for i in range(self.comp_size):
                    mean = self.gp_models[i].mean_module.constant.item()
                    outputscale = self.gp_models[i].covar_module.outputscale.item()
                    x[:,i] = mean + np.sqrt(outputscale) *  gpmat[i]['K'].matmul(z[:,i])
                dxdtOde = self.fOde(x)
                lkh = torch.zeros((self.comp_size, 3))
                for i in range(self.comp_size):
                    mean = self.gp_models[i].mean_module.constant.item()
                    outputscale = self.gp_models[i].covar_module.outputscale.item()
                    # p(X[I] = x[I]) = P(U[I] = u[I])
                    lkh[i,0] = -0.5 * z[:,i].square().sum()
                    # p(Y[I] = y[I] | X[I] = x[I])
                    yiError = gpmat[i]['LQinv'].matmul(self.ys[i][:,1]-(mean+gpmat[i]['s_tilde'].matmul(np.sqrt(outputscale) * z[:,i])))
                    lkh[i,1] = -0.5/outputscale * yiError.square().sum()
                    # p(X'[I]=f(x[I],theta)|X(I)=x(I))
                    dxdtOde_ft = Fourier_vetor(dxdtOde[:,i], k)
                    x_new = gpmat[i]['m_tilde'].matmul(np.sqrt(outputscale) * z[:,i])
                    x_ft = Fourier_vetor(x_new, k)
                    dxidtError = gpmat[i]['LKinv'].matmul(dxdtOde_ft-x_ft)
                    lkh[i,2] = -0.5/outputscale * dxidtError.square().sum() / self.grid_size * yiError.size(0)
                state_loss = -torch.sum(lkh)  / self.grid_size
                state_loss.backward()
                state_optimizer.step()
            # detach gradient information
            z.requires_grad_(False)

            if (verbose and (epoch==0 or (epoch+1) % int(max_epoch/5) == 0)):
                print('%d/%d iteration: %.6f' %(epoch+1,max_epoch,state_loss.item()))

            # reconstruct x
            x = torch.empty_like(u).double()
            for i in range(self.comp_size):
                mean = self.gp_models[i].mean_module.constant.item()
                outputscale = self.gp_models[i].covar_module.outputscale.item()
                x[:,i] = mean + np.sqrt(outputscale) * gpmat[i]['K'].matmul(z[:,i])

            if ((epoch+1) < max_epoch):
                for param in self.fOde.parameters():
                    param.requires_grad_(True)
                for tt in range(1):
                    xr = x.clone()
                    dxrdtOde = self.fOde(xr)
                    theta_optimizer.zero_grad()
                    lkh = torch.zeros(self.comp_size)
                    for i in range(self.comp_size):
                        mean = self.gp_models[i].mean_module.constant.item()
                        outputscale = self.gp_models[i].covar_module.outputscale.item()
                        xr_new = gpmat[i]['m_tilde'].matmul(np.sqrt(outputscale) * z[:,i])
                        xr_ft, dxrdtOde_ft = Fourier_vetor(xr_new, k), Fourier_vetor(dxrdtOde[:,i], k)
                        dxrdtError = gpmat[i]['LKinv'].matmul(dxrdtOde_ft-xr_ft)
                        lkh[i] = -0.5/outputscale * dxrdtError.square().mean()
                    theta_loss = -torch.sum(lkh)
                    theta_loss.backward()
                    theta_optimizer.step()
                # detach theta gradient
                for param in self.fOde.parameters():
                    param.requires_grad_(False)

        e_time = time.time()
        r_time = e_time - s_time
        print('Time for optimization: %.2f' % r_time)

        if (returnX):
            return (self.grid.numpy(), x.numpy(), r_time)

    def predict(self, x0, ts, **params):
        # obtain prediction by numerical integration
        itg = integrator.RungeKutta(self.fOde)
        ts = torch.tensor(ts).double().squeeze()
        xs = itg.forward(x0, ts, **params)
        return (xs.numpy())