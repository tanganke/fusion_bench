from abc import abstractmethod
from typing import Tuple

import cvxopt
import cvxpy as cp
import numpy as np
import torch
from torch import Tensor

"""Implementation of Pareto HyperNetworks with:
1. Linear scalarization
3. EPO


EPO code from: https://github.com/dbmptr/EPOSearch
"""


class Solver:
    def __init__(self, n_tasks):
        super().__init__()
        self.n_tasks = n_tasks

    @abstractmethod
    def get_weighted_loss(self, losses, ray, parameters=None, **kwargs):
        pass

    def __call__(self, losses, ray, parameters, **kwargs):
        return self.get_weighted_loss(losses, ray, parameters, **kwargs)


class LinearScalarizationSolver(Solver):
    """For LS we use the preference ray to weigh the losses"""

    def __init__(self, n_tasks):
        super().__init__(n_tasks)

    def get_weighted_loss(self, losses, ray, parameters=None, **kwargs):
        return (losses * ray).sum()


class EPOSolver(Solver):
    """Wrapper over EPO"""

    def __init__(self, n_tasks, n_params):
        super().__init__(n_tasks)
        self.solver = EPO(n_tasks=n_tasks, n_params=n_params)

    def get_weighted_loss(
        self, losses, ray, parameters: Tuple[Tensor] = None, **kwargs
    ):
        assert parameters is not None
        return self.solver.get_weighted_loss(losses, ray, parameters)


class EPO:
    def __init__(self, n_tasks, n_params):
        self.n_tasks = n_tasks
        self.n_params = n_params

    def __call__(self, losses, ray, parameters):
        return self.get_weighted_loss(losses, ray, parameters)

    @staticmethod
    def _flattening(grad: Tuple[Tensor]) -> Tensor:
        return torch.cat(tuple(g.flatten() for g in grad), dim=0)

    def get_weighted_loss(
        self,
        losses: Tuple[Tensor],
        ray: Tensor,
        parameters: Tuple[Tensor],
    ):
        lp = ExactParetoLP(m=self.n_tasks, n=self.n_params, r=ray.cpu().numpy())

        grads = []
        for i, loss in enumerate(losses):
            g = torch.autograd.grad(
                loss, parameters, retain_graph=True, allow_unused=True
            )
            g = tuple(filter(lambda x: x is not None, g))
            flat_grad = self._flattening(g)
            grads.append(flat_grad.data)

        G = torch.stack(grads)
        GG_T = G @ G.T
        GG_T = GG_T.detach().cpu().numpy()

        numpy_losses = losses.detach().cpu().numpy()

        try:
            alpha = lp.get_alpha(numpy_losses, G=GG_T, C=True)
        except Exception as excep:
            print(excep)
            alpha = None

        if alpha is None:  # A patch for the issue in cvxpy
            alpha = (ray / ray.sum()).cpu().numpy()

        alpha *= self.n_tasks
        alpha = torch.from_numpy(alpha).to(losses.device)

        weighted_loss = torch.sum(losses * alpha)
        return weighted_loss


class ExactParetoLP(object):
    """modifications of the code in https://github.com/dbmptr/EPOSearch"""

    def __init__(self, m, n, r, eps=1e-4):
        cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"
        self.m = m
        self.n = n
        self.r = r
        self.eps = eps
        self.last_move = None
        self.a = cp.Parameter(m)  # Adjustments
        self.C = cp.Parameter((m, m))  # C: Gradient inner products, G^T G
        self.Ca = cp.Parameter(m)  # d_bal^TG
        self.rhs = cp.Parameter(m)  # RHS of constraints for balancing

        self.alpha = cp.Variable(m)  # Variable to optimize

        obj_bal = cp.Maximize(self.alpha @ self.Ca)  # objective for balance
        constraints_bal = [
            self.alpha >= 0,
            cp.sum(self.alpha) == 1,  # Simplex
            self.C @ self.alpha >= self.rhs,
        ]
        self.prob_bal = cp.Problem(obj_bal, constraints_bal)  # LP balance

        obj_dom = cp.Maximize(cp.sum(self.alpha @ self.C))  # obj for descent
        constraints_res = [
            self.alpha >= 0,
            cp.sum(self.alpha) == 1,  # Restrict
            self.alpha @ self.Ca >= -cp.neg(cp.max(self.Ca)),
            self.C @ self.alpha >= 0,
        ]
        constraints_rel = [
            self.alpha >= 0,
            cp.sum(self.alpha) == 1,  # Relaxed
            self.C @ self.alpha >= 0,
        ]
        self.prob_dom = cp.Problem(obj_dom, constraints_res)  # LP dominance
        self.prob_rel = cp.Problem(obj_dom, constraints_rel)  # LP dominance

        self.gamma = 0  # Stores the latest Optimum value of the LP problem
        self.mu_rl = 0  # Stores the latest non-uniformity

    def get_alpha(self, loss_terms, G: Tensor, r=None, C: bool = False, relax=False):
        r = self.r if r is None else r
        assert len(loss_terms) == len(G) == len(r) == self.m, "length != m"
        rl, self.mu_rl, self.a.value = adjustments(loss_terms, r)
        self.C.value = G if C else G @ G.T
        self.Ca.value = self.C.value @ self.a.value

        if self.mu_rl > self.eps:
            J = self.Ca.value > 0
            if len(np.where(J)[0]) > 0:
                J_star_idx = np.where(rl == np.max(rl))[0]
                self.rhs.value = self.Ca.value.copy()
                self.rhs.value[J] = -np.inf  # Not efficient; but works.
                self.rhs.value[J_star_idx] = 0
            else:
                self.rhs.value = np.zeros_like(self.Ca.value)
            self.gamma = self.prob_bal.solve(solver=cp.GLPK, verbose=False)
            self.last_move = "bal"
        else:
            if relax:
                self.gamma = self.prob_rel.solve(solver=cp.GLPK, verbose=False)
            else:
                self.gamma = self.prob_dom.solve(solver=cp.GLPK, verbose=False)
            self.last_move = "dom"

        return self.alpha.value


def mu(rl, normed=False):
    if len(np.where(rl < 0)[0]):
        raise ValueError(f"rl<0 \n rl={rl}")
        # return None
    m = len(rl)
    l_hat = rl if normed else rl / rl.sum()
    eps = np.finfo(rl.dtype).eps
    l_hat = l_hat[l_hat > eps]
    return np.sum(l_hat * np.log(l_hat * m))


def adjustments(l, r=1):
    m = len(l)
    rl = r * l
    l_hat = rl / rl.sum()
    mu_rl = mu(l_hat, normed=True)
    a = r * (np.log(l_hat * m) - mu_rl)
    return rl, mu_rl, a
