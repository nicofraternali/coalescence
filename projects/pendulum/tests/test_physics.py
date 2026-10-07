import math

from pendulum_physics import Pendulum, energy, rk4_step

DT = 0.002


def simulate(p, state, seconds):
    states = [state]
    for _ in range(round(seconds / DT)):
        state = rk4_step(p, state, DT)
        states.append(state)
    return states


def test_undamped_energy_is_conserved():
    # A chaotic release (both arms well above horizontal) for 5 s.
    p = Pendulum(m1=2.0, m2=3.0, l1=1.2, l2=0.8, damping=0.0)
    states = simulate(p, (2.5, 3.0, 0.0, 0.0), 5.0)
    e0 = energy(p, states[0])
    drift = max(abs(energy(p, s) - e0) for s in states)
    # Scale by the energy range the pendulum can move through.
    scale = (p.m1 + p.m2) * 9.81 * (p.l1 + p.l2)
    assert drift / scale < 1e-6


def test_damped_energy_decreases():
    p = Pendulum(m1=2.0, m2=3.0, l1=1.2, l2=0.8, damping=0.5)
    states = simulate(p, (2.5, 3.0, 0.0, 0.0), 5.0)
    energies = [energy(p, s) for s in states]
    assert energies[-1] < energies[0]
    # Friction only removes energy: no step may gain more than integration noise.
    assert max(b - a for a, b in zip(energies, energies[1:])) < 1e-9


def test_small_swing_matches_simple_pendulum_period():
    # With a negligible second bob and a small angle, the first arm is a
    # simple pendulum: T = 2*pi*sqrt(l/g).
    p = Pendulum(m1=1.0, m2=1e-9, l1=1.0, l2=0.5, damping=0.0)
    states = simulate(p, (0.05, 0.05, 0.0, 0.0), 6.0)
    # Times where theta1 crosses zero going downward-to-upward (sign change).
    crossings = [
        i * DT for i in range(1, len(states))
        if states[i - 1][0] > 0 >= states[i][0]
    ]
    period = crossings[1] - crossings[0]
    assert abs(period - 2 * math.pi * math.sqrt(1.0 / 9.81)) < 2 * DT


def test_same_inputs_give_same_trajectory():
    p = Pendulum(m1=1.5, m2=4.0, l1=0.6, l2=1.9, damping=0.1)
    assert simulate(p, (1.0, -2.0, 0.0, 0.0), 1.0) == simulate(p, (1.0, -2.0, 0.0, 0.0), 1.0)
