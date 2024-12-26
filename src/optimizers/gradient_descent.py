import numpy as np
from tqdm import tqdm

class GradientDescent:
    def __init__(self, lr=1e-4, tol=1e-6, max_iter=10000) -> None:
        self.lr = lr
        self.tol = tol
        self.max_iter = int(max_iter)

    def minimize(self, loss_grad, w_init):
        w = w_init
        with tqdm(total=self.max_iter) as pbar:
            for n_iter in range(self.max_iter):
                w_new = w - self.lr * loss_grad(w)
                
                if np.linalg.norm(w_new - w) < self.tol:
                    print(f"Converged after {n_iter} iterations")
                    break
                w = w_new
                pbar.update(1)
        return w
        