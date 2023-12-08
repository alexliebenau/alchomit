from base import optimizer
from search_based import random_search
from scipy.optimize import minimize


class bfgs(optimizer):
    def __init__(self, use_constraints: bool = True, method: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.scipy:
            print(f'WARNING: Only SciPy is supported for this kind of optimisation.')
            self.scipy = True

        self.use_constraints = use_constraints
        self.method = method if method is not None else 'L-BFGS-B'

    def objective_function(self, x):
        if True: # self.use_constraints:
            if self.framework.check_constraints(x):
                return self.model.predict(x)
            return 0
        # else:
        #     return self.model.predict(x)

    def run_scipy(self):
        random_x = random_search(iterations=1, objective_model=self.model, framework=self.framework).generate_x()

        res = minimize(
            self.objective_function,
            x0=random_x,
            options={'maxiter': self.n_iter},
            bounds=self.bounds,
            constraints=self.framework.constraints if not self.use_constraints else None,
            method=self.method
        )
        opt_x = res.x
        opt_x_t = self.framework.round_indices(opt_x)
        y = self.model.predict(opt_x)
        return opt_x_t, y