import os
import numpy as np
import matplotlib.pyplot as plt
from model import simulate_random_walk

# Настройки шрифта, как в main.py
plt.rcParams['font.family'] = 'Times New Roman'
font_title = {'family': 'Times New Roman', 'style': 'italic', 'weight': 'bold', 'size': 12}

# Параметры симуляции
N = 10000
steps = 300
traj = simulate_random_walk(N, steps, seed=42)

# Берем каждую 10-ю частицу, чтобы графики не были слишком тяжелыми
traj_light = traj[:, ::10, :]
max_val = np.max(np.abs(traj)) + 5

# Выбираем шаги для раскадровки
snapshots = [0, 100, 200, 300]
labels = ["(а) Шаг 0", "(б) Шаг 100", "(в) Шаг 200", "(г) Шаг 300"]

# Создаем сетку 2x2
fig, axes = plt.subplots(2, 2, figsize=(10, 10), facecolor='#0f0f0f')
axes = axes.flatten()

for i, (step, label) in enumerate(zip(snapshots, labels)):
    ax = axes[i]
    ax.set_facecolor('#0f0f0f')
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_aspect('equal')
    ax.axis('off') # Отключаем оси для чистоты
    
    pos = traj_light[step]
    distances = np.linalg.norm(pos, axis=1)
    
    # Рисуем точки
    ax.scatter(pos[:, 0], pos[:, 1], s=8, alpha=0.8, c=distances,
               cmap='plasma', marker='o', edgecolors='none')
    
    # Рисуем теоретический круг (кроме шага 0, где он равен 0)
    if step > 0:
        sigma = np.sqrt(2 * 0.25 * step)
        radius = 2 * sigma
        circle = plt.Circle((0, 0), radius, color='white', fill=False, 
                            linestyle='-', alpha=0.6, linewidth=1.5)
        ax.add_patch(circle)
        
    # Подпись к кадру
    ax.set_title(label, fontdict=font_title, color='white', pad=10)

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/evolution_storyboard.png", dpi=200, facecolor='#0f0f0f', bbox_inches='tight')
plt.close()
print("Раскадровка сохранена в results/evolution_storyboard.png")