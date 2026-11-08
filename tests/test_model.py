import numpy as np
import pytest
from model import simulate_random_walk, calculate_msd

def test_shapes():
    """Проверяет, что массивы имеют правильную форму."""
    N = 100
    steps = 50
    traj = simulate_random_walk(N, steps)
    assert traj.shape == (steps + 1, N, 2)
    
    msd = calculate_msd(traj)
    assert msd.shape == (steps + 1,)

def test_start_position():
    """Проверяет, что все частицы стартуют из начала координат (0,0)."""
    traj = simulate_random_walk(N=10, steps=10)
    start_positions = traj[0]
    assert np.all(start_positions == 0)

def test_seed_reproducibility():
    """Проверяет, что при одинаковом seed результаты совпадают."""
    traj1 = simulate_random_walk(N=100, steps=10, seed=42)
    traj2 = simulate_random_walk(N=100, steps=10, seed=42)
    assert np.array_equal(traj1, traj2)

def test_msd_growth():
    """Проверяет, что MSD растет примерно линейно (закон диффузии)."""
    traj = simulate_random_walk(N=10000, steps=100, seed=1)
    msd = calculate_msd(traj)
    
    assert 90 <= msd[100] <= 110
    assert 40 <= msd[50] <= 60

def test_step_displacement():
    """Проверяет, что частица не стоит на месте."""
    traj = simulate_random_walk(N=1, steps=10, seed=42)
    assert not np.array_equal(traj[0], traj[-1])