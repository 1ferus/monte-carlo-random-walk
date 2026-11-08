"""
Модуль для стохастического моделирования методом Монте-Карло.
Реализует случайное блуждание ансамбля частиц и расчет статистических характеристик.
"""
import numpy as np

def simulate_random_walk(N, steps, seed=None):
    """Моделирует случайное блуждание для N частиц."""
    if seed is not None:
        np.random.seed(seed)
        
    trajectories = np.zeros((steps + 1, N, 2), dtype=np.int32)
    random_moves = np.random.randint(0, 4, size=(steps, N))
    
    dx = np.where(random_moves == 0, 1, np.where(random_moves == 1, -1, 0))
    dy = np.where(random_moves == 2, 1, np.where(random_moves == 3, -1, 0))
    
    for step in range(1, steps + 1):
        trajectories[step, :, 0] = trajectories[step - 1, :, 0] + dx[step - 1]
        trajectories[step, :, 1] = trajectories[step - 1, :, 1] + dy[step - 1]
        
    return trajectories

def calculate_msd(trajectories):
    """Вычисляет среднеквадратичное смещение."""
    squared_distances = trajectories[:, :, 0]**2 + trajectories[:, :, 1]**2
    return np.mean(squared_distances, axis=1)

def get_theoretical_gaussian(x_edges, y_edges, steps, step_size=1.0):
    """Возвращает теоретическое распределение Гаусса."""
    D = (step_size**2) / 2.0
    variance = 2 * D * steps
    sigma = np.sqrt(variance)
    
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers)
    
    Z = (1.0 / (2 * np.pi * variance)) * np.exp(-(X**2 + Y**2) / (2 * variance))
    return Z