# ssa.py
import numpy as np

class SalpSwarmAlgorithm:
    def __init__(self, obj_func, dim, num_salp=30, max_iter=100, lb=-10, ub=10):
        self.obj_func = obj_func
        self.dim = dim
        self.num_salp = num_salp
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        self.food_position = np.zeros(dim)
        self.food_fitness = float('inf')

    def optimize(self):
        salp_positions = np.random.uniform(self.lb, self.ub, (self.num_salp, self.dim))
        for t in range(self.max_iter):
            c1 = 2 * np.exp(-(4*t/self.max_iter)**2)
            for i in range(self.num_salp):
                if i == 0:
                    self.food_position = np.clip(salp_positions[0] + c1*np.random.uniform(-1,1,self.dim), self.lb, self.ub)
                else:
                    salp_positions[i] = 0.5 * (salp_positions[i] + salp_positions[i-1])
            for i in range(self.num_salp):
                fitness = self.obj_func(salp_positions[i])
                if fitness < self.food_fitness:
                    self.food_fitness = fitness
                    self.food_position = salp_positions[i].copy()
        return self.food_position, self.food_fitness

# Example usage
if __name__ == "__main__":
    def sphere(x):
        return sum(x**2)
    
    ssa = SalpSwarmAlgorithm(sphere, dim=5)
    best_pos, best_fit = ssa.optimize()
    print("Best position:", best_pos)
    print("Best fitness:", best_fit)
