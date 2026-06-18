import numpy as np
import matplotlib.pyplot as plt

# ---------- Objective function ----------

def sphere(x):
    return np.sum(x**2)

bounds = (-5, 5)
low, high = bounds

# ---------- Instrumented algorithms: return trajectory of best-so-far ----------

def pio_traj(f, dim=2, n_pigeons=30, iters=50, bounds=(-5, 5)):
    low, high = bounds
    switch_iter = iters // 2

    X = np.random.uniform(low, high, size=(n_pigeons, dim))
    V = np.zeros_like(X)

    best_pos = X[0].copy()
    best_fit = f(best_pos)
    traj = []

    for t in range(iters):
        fitness = np.array([f(x) for x in X])
        idx_best = np.argmin(fitness)
        if fitness[idx_best] < best_fit:
            best_fit = fitness[idx_best]
            best_pos = X[idx_best].copy()

        traj.append(best_pos.copy())

        if t < switch_iter:
            # Map & compass stage
            R = np.random.rand(*X.shape)
            V = V + R * (best_pos - X)
            X = X + V
        else:
            # Landmark stage
            order = np.argsort(fitness)
            survivors = order[: len(order) // 2]
            X = X[survivors]
            center = np.mean(X, axis=0)
            R = np.random.rand(*X.shape)
            X = X + R * (center - X)

        X = np.clip(X, low, high)

    return np.array(traj)


def ssa_traj(f, dim=2, n_salp=30, iters=50, bounds=(-5, 5)):
    import math
    low, high = bounds

    X = np.random.uniform(low, high, size=(n_salp, dim))
    fitness = np.array([f(x) for x in X])
    best_idx = np.argmin(fitness)
    food = X[best_idx].copy()
    food_fit = fitness[best_idx]

    traj = []

    for t in range(1, iters + 1):
        c1 = 2 * math.exp(-((4 * t / iters) ** 2))

        for i in range(n_salp):
            if i == 0:
                # Leader
                c2 = np.random.rand(dim)
                c3 = np.random.rand(dim)
                direction = np.where(c3 >= 0.5, 1, -1)
                X[i] = food + direction * c1 * ((high - low) * c2 + low)
            else:
                # Followers
                X[i] = (X[i] + X[i - 1]) / 2.0

        X = np.clip(X, low, high)

        fitness = np.array([f(x) for x in X])
        idx = np.argmin(fitness)
        if fitness[idx] < food_fit:
            food_fit = fitness[idx]
            food = X[idx].copy()

        traj.append(food.copy())

    return np.array(traj)


def abc_traj(f, dim=2, n_food=20, iters=50, bounds=(-5, 5), limit=10):
    import random
    low, high = bounds

    foods = np.random.uniform(low, high, size=(n_food, dim))
    fitness = np.array([f(x) for x in foods])
    trials = np.zeros(n_food)
    traj = []

    def random_neighbor(x):
        k = random.randint(0, dim - 1)
        v = x.copy()
        phi = random.uniform(-1, 1)
        v[k] += phi * (high - low)
        return np.clip(v, low, high)

    for _ in range(iters):
        # Employed phase
        for i in range(n_food):
            v = random_neighbor(foods[i])
            fv = f(v)
            if fv < fitness[i]:
                foods[i], fitness[i] = v, fv
                trials[i] = 0
            else:
                trials[i] += 1

        # Onlooker phase
        inv_fit = 1.0 / (1.0 + fitness)
        probs = inv_fit / np.sum(inv_fit)
        for _ in range(n_food):
            i = np.random.choice(np.arange(n_food), p=probs)
            v = random_neighbor(foods[i])
            fv = f(v)
            if fv < fitness[i]:
                foods[i], fitness[i] = v, fv
                trials[i] = 0
            else:
                trials[i] += 1

        # Scout phase
        for i in range(n_food):
            if trials[i] > limit:
                foods[i] = np.random.uniform(low, high, size=dim)
                fitness[i] = f(foods[i])
                trials[i] = 0

        best_idx = np.argmin(fitness)
        traj.append(foods[best_idx].copy())

    return np.array(traj)


def aco_traj(f, dim=2, n_ants=20, iters=50, bounds=(-5, 5), evaporation=0.5):
    low, high = bounds
    pheromone = np.ones(dim)
    best_pos = np.random.uniform(low, high, size=dim)
    best_val = f(best_pos)

    traj = []

    for _ in range(iters):
        for _ in range(n_ants):
            sigma = (high - low) / (1 + pheromone)
            pos = np.random.normal(0, sigma, size=dim)
            pos = np.clip(pos, low, high)
            v = f(pos)
            if v < best_val:
                best_val = v
                best_pos = pos.copy()

        pheromone *= (1 - evaporation)
        pheromone += 1.0 / (1.0 + best_val)

        traj.append(best_pos.copy())

    return np.array(traj)


def pso_traj(f, dim=2, n_particles=30, iters=50, bounds=(-5, 5),
             w=0.7, c1=1.5, c2=1.5):
    low, high = bounds

    X = np.random.uniform(low, high, size=(n_particles, dim))
    V = np.random.uniform(-abs(high - low), abs(high - low),
                          size=(n_particles, dim))

    pbest = X.copy()
    pbest_val = np.array([f(x) for x in X])

    gbest_idx = np.argmin(pbest_val)
    gbest = pbest[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]

    traj = []

    for _ in range(iters):
        r1 = np.random.rand(n_particles, dim)
        r2 = np.random.rand(n_particles, dim)

        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = X + V
        X = np.clip(X, low, high)

        vals = np.array([f(x) for x in X])
        improved = vals < pbest_val
        pbest[improved] = X[improved]
        pbest_val[improved] = vals[improved]

        gbest_idx = np.argmin(pbest_val)
        if pbest_val[gbest_idx] < gbest_val:
            gbest_val = pbest_val[gbest_idx]
            gbest = pbest[gbest_idx].copy()

        traj.append(gbest.copy())

    return np.array(traj)


def ga_traj(f, dim=2, pop_size=30, iters=50, bounds=(-5, 5),
            crossover_rate=0.8, mutation_rate=0.1):
    low, high = bounds
    pop = np.random.uniform(low, high, size=(pop_size, dim))

    def fitness(x):
        return -f(x)

    traj = []

    for _ in range(iters):
        fit_vals = np.array([fitness(ind) for ind in pop])
        best_idx = np.argmax(fit_vals)
        traj.append(pop[best_idx].copy())

        def select():
            a, b = np.random.randint(0, pop_size, size=2)
            return pop[a] if fit_vals[a] > fit_vals[b] else pop[b]

        new_pop = []
        while len(new_pop) < pop_size:
            parent1 = select()
            parent2 = select()

            # Crossover
            if np.random.rand() < crossover_rate:
                alpha = np.random.rand(dim)
                child1 = alpha * parent1 + (1 - alpha) * parent2
                child2 = alpha * parent2 + (1 - alpha) * parent1
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Mutation
            for child in (child1, child2):
                for d in range(dim):
                    if np.random.rand() < mutation_rate:
                        child[d] += np.random.normal(0, 0.1 * (high - low))
                child[:] = np.clip(child, low, high)
                new_pop.append(child)
                if len(new_pop) == pop_size:
                    break

        pop = np.array(new_pop)

    return np.array(traj)

# ---------- Generate trajectories ----------

traj_pio = pio_traj(sphere)
traj_ssa = ssa_traj(sphere)
traj_abc = abc_traj(sphere)
traj_aco = aco_traj(sphere)
traj_pso = pso_traj(sphere)
traj_ga  = ga_traj(sphere)

# ---------- Plot on contour of the sphere function ----------

x = np.linspace(low, high, 200)
y = np.linspace(low, high, 200)
Xg, Yg = np.meshgrid(x, y)
Z = Xg**2 + Yg**2

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
algos = [("PIO", traj_pio), ("SSA", traj_ssa), ("ABC", traj_abc),
         ("ACO", traj_aco), ("PSO", traj_pso), ("GA", traj_ga)]

for ax, (name, traj) in zip(axes.ravel(), algos):
    cs = ax.contour(Xg, Yg, Z, levels=20, cmap='viridis')
    ax.plot(traj[:, 0], traj[:, 1],
            marker='o', markersize=2, linewidth=1, color='red')
    ax.set_title(name)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)

plt.tight_layout()
plt.show()