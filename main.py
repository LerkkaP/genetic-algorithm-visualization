import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def rastrigin_2D(X, A=10):
    return A*2 + np.sum(X**2 - A*np.cos(2*np.pi*X), axis=-1)

def fitness(X):
    return -rastrigin_2D(X)

x1 = np.linspace(-5.12, 5.12, 400)
x2 = np.linspace(-5.12, 5.12, 400)
X, Y = np.meshgrid(x1, x2)
data = np.vstack([X.ravel(), Y.ravel()]).T
Z = rastrigin_2D(data).reshape(X.shape)

# Hyperparams
N = 100
max_generations = 200
mutation_strength = 0.2

population = np.random.uniform(-5.12, 5.12, size=(N, 2))
best_solution_over_time = []
populations_over_time = []

for gen in range(max_generations):
    fitness_values = fitness(population)
    
    # Track best individual
    best_index = np.argmax(fitness_values)
    best_solution_over_time.append(population[best_index].copy())
    populations_over_time.append(population.copy())
    
    new_population = np.zeros_like(population)
    
    for i in range(N):
        # Tournament selection
        indices1 = np.random.choice(N, 3, replace=False)
        parent1 = population[indices1[np.argmax(fitness_values[indices1])]]
        
        indices2 = np.random.choice(N, 3, replace=False)
        parent2 = population[indices2[np.argmax(fitness_values[indices2])]]
        
        # Crossover
        alpha = np.random.rand()
        child = alpha * parent1 + (1 - alpha) * parent2
        
        # Mutation
        child += np.random.normal(0, mutation_strength, size=2)
        new_population[i] = child
    
    population = new_population.copy()

final_fitness_values = fitness(population)
best_index = np.argmax(final_fitness_values)
best_individual = population[best_index]
print("Best individual (x1, x2):", best_individual)
print("Negated fitness:", final_fitness_values[best_index])
print("Rastrigin value:", rastrigin_2D(best_individual.reshape(1,2))[0])

fig, ax = plt.subplots(figsize=(8,6))
contour = ax.contourf(X, Y, Z, levels=30, cmap="viridis")
scatter = ax.scatter([], [], color="red", s=50, alpha=0.6)
best_dot = ax.scatter([], [], color="yellow", s=100, edgecolor="black")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
fig.colorbar(contour, ax=ax, label="Rastrigin value")

def init():
    scatter.set_offsets(np.empty((0,2)))
    best_dot.set_offsets(np.empty((0,2)))
    return scatter, best_dot

def update(frame):
    pop = populations_over_time[frame]
    best = best_solution_over_time[frame]
    scatter.set_offsets(pop)
    best_dot.set_offsets(best.reshape(1,2))
    ax.set_title(f"Generation {frame}")
    return scatter, best_dot

ani = animation.FuncAnimation(
    fig, update, frames=len(populations_over_time),
    init_func=init, blit=True, interval=100
)

ani.save("population_evolution.mp4", writer="ffmpeg", fps=10)
plt.close(fig)
print("Animation saved as population_evolution.mp4")
