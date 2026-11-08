import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from model import simulate_random_walk, calculate_msd, get_theoretical_gaussian

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'

font_title = {'family': 'Times New Roman', 'style': 'italic', 'weight': 'bold', 'size': 16}
font_labels = {'family': 'Times New Roman', 'style': 'italic', 'weight': 'bold', 'size': 14}

def plot_msd(msd, steps, save_path):
    plt.close('all')
    t = np.arange(steps + 1)
    D = 0.25
    msd_theory = 4 * D * t
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, msd, color='#0077be', linewidth=3, label="Модель Монте-Карло")
    ax.plot(t, msd_theory, color='#d62728', linestyle='--', linewidth=2, label="Теория (4Dt)")
    
    ax.set_xlabel("Количество шагов (t)", fontdict=font_labels)
    ax.set_ylabel("Среднеквадратичное смещение <r²>", fontdict=font_labels)
    ax.set_title("Закон диффузии Эйнштейна-Смолуховского", fontdict=font_title, pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=14, prop={'family': 'Times New Roman', 'style': 'italic'})
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_histogram(trajectories, steps, save_path):
    plt.close('all')
    final_positions = trajectories[-1]
    x, y = final_positions[:, 0], final_positions[:, 1]
    
    max_val = int(np.max(np.abs(final_positions))) + 5
    bins = np.arange(-max_val, max_val + 1)
    
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=[bins, bins], density=True)
    Z_theory = get_theoretical_gaussian(x_edges, y_edges, steps)
    
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='#1e1e1e', constrained_layout=True)
    
    ax1 = axes[0]
    ax1.set_facecolor('#2d2d2d')
    pc1 = ax1.pcolormesh(X, Y, hist.T, cmap='magma', shading='auto')
    ax1.set_title("Модель Монте-Карло", fontdict=font_title, color='white', pad=10)
    
    ax2 = axes[1]
    ax2.set_facecolor('#2d2d2d')
    pc2 = ax2.pcolormesh(X, Y, Z_theory.T, cmap='magma', shading='auto')
    ax2.contour(X, Y, Z_theory.T, levels=5, colors='#00e5ff', linewidths=1.5)
    ax2.set_title("Теоретический Гаусс (2D)", fontdict=font_title, color='white', pad=10)
    
    for ax in axes:
        ax.set_xlabel("Координата x", fontdict=font_labels, color='white')
        ax.set_ylabel("Координата y", fontdict=font_labels, color='white')
        ax.tick_params(colors='white', labelsize=12)
        ax.set_aspect('equal')
        
    cbar = fig.colorbar(pc1, ax=axes, pad=0.02)
    cbar.set_label("Плотность вероятности", fontdict=font_labels, color='white')
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
    
    fig.savefig(save_path, dpi=200, facecolor='#1e1e1e', bbox_inches='tight')
    plt.close(fig)

def create_gif(trajectories, save_path):
    plt.close('all')
    # Берем каждую 4-ю частицу (около 2500 штук) для идеального баланса плотности и скорости
    traj_light = trajectories[:, ::4, :] 
    
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#0f0f0f')
    ax.set_facecolor('#0f0f0f')
    
    max_val = np.max(np.abs(trajectories)) + 5
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_aspect('equal')
    ax.axis('off')
    
    step_indices = range(0, len(traj_light), 3)
    
    def update(frame_idx):
        ax.clear()
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
        ax.set_aspect('equal')
        ax.axis('off')
        
        step = step_indices[frame_idx]
        pos = traj_light[step]
        distances = np.linalg.norm(pos, axis=1)
        
        # один слой четких точек с градиентом
        ax.scatter(pos[:, 0], pos[:, 1], s=15, alpha=0.9, c=distances,
                   cmap='plasma', marker='o', edgecolors='none')
        
        if step > 0:
            sigma = np.sqrt(2 * 0.25 * step)
            radius = 2 * sigma
            # Белая граница
            circle = plt.Circle((0, 0), radius, color='white', fill=False, 
                                linestyle='-', alpha=0.7, linewidth=1.5)
            ax.add_patch(circle)
            
            ax.text(max_val*0.5, -max_val*0.8, f"Граница 2σ: {radius:.1f}", 
                    color='white', alpha=0.9,
                    fontdict={'family': 'Times New Roman', 'size': 14, 'style': 'italic', 'weight': 'bold'})
        
        ax.set_title(f"Случайное блуждание (Шаг {step})", fontdict=font_title, color='white', pad=20)
        return ax,
    
    print("Генерация четкой анимации...", flush=True)
    anim = FuncAnimation(fig, update, frames=len(step_indices), interval=100, blit=False)
    anim.save(save_path, writer=PillowWriter(fps=15))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Стохастическое моделирование (Random Walk)")
    parser.add_argument("--N", type=int, default=10000, help="Количество частиц")
    parser.add_argument("--steps", type=int, default=300, help="Количество шагов")
    parser.add_argument("--out", type=str, default="results", help="Папка для результатов")
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    
    print(f"Запуск: N={args.N}, шагов={args.steps}")
    
    traj = simulate_random_walk(args.N, args.steps, seed=42)
    msd = calculate_msd(traj)
    
    plot_msd(msd, args.steps, f"{args.out}/msd_plot.png")
    plot_histogram(traj, args.steps, f"{args.out}/hist_2d.png")
    
    print("Генерация анимации...")
    create_gif(traj, f"{args.out}/brownian_motion.gif")
    
    print(f"Готово. Результаты в папке {args.out}")