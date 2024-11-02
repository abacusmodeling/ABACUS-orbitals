'''this module defines the minimizer of Spillage

What should be provided is the functor that takes the variables and returns the
spillage value.

'''
from scipy.optimize import minimize, basinhopping
import torch_optimizer as topt
from torch import Tensor
import unittest
import numpy as np

def _torchopt(f, x0, minimizer, **kwargs):
    '''minimize the function with SWATS optimizer
    
    Parameters
    ----------
    f: callable
        the function to minimize
    x0: list
        the initial guess
    minimizer: str
        the minimizer to use, can ONLY be 'swats' now
    **kwargs: dict
        the keyword arguments for the optimizer

    Returns
    -------
    list, float
        the optimized variables, the minimized value
    '''
    raise NotImplementedError('The pytorch optimizer is not implemented yet')

    x0 = Tensor(x0)
    x0.requires_grad = True

    nstep = kwargs.get('maxiter', 3000)
    disp = kwargs.get('disp', False)
    ndisp = kwargs.get('ndisp', 50)
    learning_rate = kwargs.get('learning_rate', 0.001)

    optimizer = None
    if minimizer == 'swats':
        optimizer = topt.SWATS([x0], lr=learning_rate)
    else:
        raise ValueError(f'{minimizer} is not supported')

    loss = None
    for i in range(nstep):
        optimizer.zero_grad()
        loss = f(x0)
        loss.backward()
        optimizer.step()
        # print each nstep/ndisp steps
        if disp and i % (nstep//ndisp) == 0:
            print(f'step {i:8d}, loss {loss.item():.8e}')

    return x0.tolist(), loss.item()

def _scipyopt(f, x0, minimizer, **kwargs):
    '''minimize the function with scipy optimizer
    
    Parameters
    ----------
    f: callable
        the function to minimize
    x0: list
        the initial guess
    minimizer: str
        the minimizer to use, can ONLY be 'l-bfgs-b' and 'basinhopping' now
    **kwargs: dict
        the keyword arguments for the optimizer

    Returns
    -------
    list, float
        the optimized variables, the minimized value
    '''
    
    
    disp = kwargs.get('disp', False)
    ftol = kwargs.get('ftol', 0)
    gtol = kwargs.get('gtol', 1e-6)
    maxiter = kwargs.get('maxiter', 1000)

    x0 = np.array(x0)
    bounds = kwargs.get('bounds', [(-1.0, 1.0) for _ in x0])

    if minimizer == 'L-BFGS-B':
        options = {'disp': disp, 'ftol': ftol, 'gtol': gtol, 'maxiter': maxiter}
        res = minimize(f, x0, jac=True, method='L-BFGS-B', 
                       bounds=bounds, options=options)
    
    elif minimizer == 'basinhopping':
        minimizer_kwargs = {'method': 'L-BFGS-B', 'jac': True, 'bounds': bounds}
        res = basinhopping(f, x0, minimizer_kwargs=minimizer_kwargs, 
                           niter=maxiter, disp=disp)
    
    else:
        raise ValueError(f'{minimizer} is not supported')

    return res.x.tolist(), res.fun

def run(f, x0, minimizer, **kwargs):
    '''run the minimizer
    
    Parameters
    ----------
    f: callable
        the function to minimize
    x0: list
        the initial guess
    minimizer: str
        the minimizer to use, can be 'swats', 'l-bfgs-b' and 'basinhopping'
    **kwargs: dict
        the keyword arguments for the optimizer
    
    Returns
    -------
    list, float
        the optimized variables, the minimized value
    '''
    if minimizer == 'swats':
        return _torchopt(f, x0, minimizer, **kwargs)
    else:
        return _scipyopt(f, x0, minimizer, **kwargs)


class TestMinimizer(unittest.TestCase):
    '''test the minimizer'''
    def test_swats(self):
        '''test the pytorch_swats'''
        def f(x):
            '''the function to minimize'''
            return x**2
        x0 = [1.0]
        x, y = _torchopt(f, x0, 'swats', nstep=1000, disp=False, learning_rate=0.01)
        self.assertAlmostEqual(x[0], 0.0)
        self.assertAlmostEqual(y, 0.0)

    def test_adam(self):
        '''test the pytorch_adam'''
        def f(x):
            '''the function to minimize'''
            return x**2
        x0 = [1.0]
        with self.assertRaises(ValueError): # adam is not supported
            x = _torchopt(f, x0, 'adam', nstep=1000, disp=False, learning_rate=0.01)

    def test_l_bfgs_b(self):
        '''test the scipy_l_bfgs_b'''
        def f(x):
            '''the function to minimize'''
            return np.sum(x**2)
        x0 = [1.0]
        x, y = _scipyopt(f, x0, 'L-BFGS-B', disp=False)
        self.assertAlmostEqual(x[0], 0.0)
        self.assertAlmostEqual(y, 0.0)
    
    def test_basinhopping(self):
        '''test the scipy_basinhopping'''
        def f(x):
            '''the function to minimize'''
            return np.sum(x**2)
        x0 = [1.0]
        x, y = _scipyopt(f, x0, 'basinhopping', disp=False)
        self.assertAlmostEqual(x[0], 0.0)
        self.assertAlmostEqual(y, 0.0)

if __name__ == '__main__':
    unittest.main()