"""
Double pendulum physics in SI units, independent of py5.

State is (theta1, theta2, omega1, omega2): angles in radians measured from
the downward vertical, angular velocities in rad/s. Positions in meters
have y pointing down (matching screen coordinates).

Equations of motion come from the Lagrangian in mass-matrix form,

    M(theta) * alpha = f(theta, omega) + Q(omega)

solved as a 2x2 linear system at every evaluation. Q holds the generalized
forces from viscous friction at both joints:

    pivot: torque -c * omega1
    elbow: torque -c * (omega2 - omega1)   (relative rotation of the joint)

so c = 0 is the ideal, energy-conserving pendulum. Integration is classic
fourth-order Runge-Kutta (RK4) with a fixed step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

G = 9.81  # m/s^2

State = tuple[float, float, float, float]


@dataclass(frozen=True)
class Pendulum:
    m1: float       # kg
    m2: float       # kg
    l1: float       # m
    l2: float       # m
    damping: float  # joint friction coefficient c, N*m*s/rad


def accelerations(p: Pendulum, state: State) -> tuple[float, float]:
    """Angular accelerations (alpha1, alpha2) in rad/s^2."""
    th1, th2, w1, w2 = state
    d = th1 - th2
    cos_d, sin_d = math.cos(d), math.sin(d)

    # Mass matrix.
    m11 = (p.m1 + p.m2) * p.l1 * p.l1
    m12 = p.m2 * p.l1 * p.l2 * cos_d
    m22 = p.m2 * p.l2 * p.l2

    # Joint friction as generalized forces.
    q1 = -p.damping * w1 + p.damping * (w2 - w1)
    q2 = -p.damping * (w2 - w1)

    # Velocity (centripetal) and gravity terms.
    f1 = -p.m2 * p.l1 * p.l2 * w2 * w2 * sin_d - (p.m1 + p.m2) * G * p.l1 * math.sin(th1) + q1
    f2 = p.m2 * p.l1 * p.l2 * w1 * w1 * sin_d - p.m2 * G * p.l2 * math.sin(th2) + q2

    # Solve the 2x2 system (Cramer's rule; det > 0 for positive masses/lengths).
    det = m11 * m22 - m12 * m12
    return (f1 * m22 - m12 * f2) / det, (m11 * f2 - m12 * f1) / det


def _derivative(p: Pendulum, state: State) -> State:
    a1, a2 = accelerations(p, state)
    return state[2], state[3], a1, a2


def rk4_step(p: Pendulum, state: State, h: float) -> State:
    """Advance the state by h seconds with one RK4 step."""
    def shifted(k: State, scale: float) -> State:
        return tuple(s + scale * dk for s, dk in zip(state, k))  # type: ignore[return-value]

    k1 = _derivative(p, state)
    k2 = _derivative(p, shifted(k1, h / 2))
    k3 = _derivative(p, shifted(k2, h / 2))
    k4 = _derivative(p, shifted(k3, h))
    return tuple(  # type: ignore[return-value]
        s + h / 6 * (a + 2 * b + 2 * c + d)
        for s, a, b, c, d in zip(state, k1, k2, k3, k4)
    )


def energy(p: Pendulum, state: State) -> float:
    """Total mechanical energy in joules (zero potential at the pivot)."""
    th1, th2, w1, w2 = state
    kinetic = (
        0.5 * (p.m1 + p.m2) * p.l1 ** 2 * w1 ** 2
        + 0.5 * p.m2 * p.l2 ** 2 * w2 ** 2
        + p.m2 * p.l1 * p.l2 * w1 * w2 * math.cos(th1 - th2)
    )
    potential = -(p.m1 + p.m2) * G * p.l1 * math.cos(th1) - p.m2 * G * p.l2 * math.cos(th2)
    return kinetic + potential


def bob_positions(p: Pendulum, state: State) -> tuple[float, float, float, float]:
    """(x1, y1, x2, y2) of both bobs in meters, pivot at the origin, y down."""
    th1, th2 = state[0], state[1]
    x1 = p.l1 * math.sin(th1)
    y1 = p.l1 * math.cos(th1)
    return x1, y1, x1 + p.l2 * math.sin(th2), y1 + p.l2 * math.cos(th2)
