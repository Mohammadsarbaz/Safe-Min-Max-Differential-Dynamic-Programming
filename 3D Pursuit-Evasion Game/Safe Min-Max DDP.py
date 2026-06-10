import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 3D Pursuit–Evasion (single integrator) with:
#   - Min–Max DDP (u = pursuer minimizes, v = evader maximizes)
#   - CBF-QP safety filter (projection to halfspaces) for BOTH agents
#   - Iteration cost plot: J = sum stage_cost + terminal_cost (per DDP iteration)
#   - Time cost plot: running distance^2 along executed trajectory
#
# State: x = [pp(3); pe(3)]  -> nx = 6
# Controls: u (3), v (3)     -> nu = nv = 3
#
# Dynamics:
#   pp_{k+1} = pp_k + dt * u_k
#   pe_{k+1} = pe_k + dt * v_k
#
# Game cost (min_u max_v):
#   l_k = w_sep*||pp-pe||^2 + Ru*||u||^2 - Rv*||v||^2
#   lf  = wT*||pp-pe||^2
# ============================================================

# ==============================
# Projection to halfspaces Au >= b (simple, fast)
# ==============================
def _is_feasible(u, A, b, tol=1e-9):
    return A.size == 0 or np.all(A @ u >= b - tol)

def project_to_halfspaces(u0, A, b, n_iter=60):
    """
    Approximate projection onto {u | A u >= b} via sequential projection
    onto the most violated constraint.
    """
    if _is_feasible(u0, A, b):
        return u0
    u = u0.copy()
    for _ in range(n_iter):
        viol = b - A @ u
        idx = int(np.argmax(viol))
        if viol[idx] <= 0:
            break
        a = A[idx]
        u += (viol[idx] / (a @ a + 1e-12)) * a
    return u

# ==============================
# CBF constraints (spherical obstacles)
# ==============================
def obstacle_halfspaces(p, obstacles, R_safe, gamma):
    """
    For each obstacle:
      h(p)=||p-c||^2 - R^2 >= 0
    Single-integrator p_dot = u, CBF condition:
      2(p-c)^T u + gamma h >= 0  =>  A u >= b
      A = 2(p-c),  b = -gamma h
    """
    A, b = [], []
    for obs in obstacles:
        c = obs["center"]
        R = obs["radius"] + R_safe
        d = p - c
        h = float(d @ d - R * R)
        A.append(2.0 * d)
        b.append(-gamma * h)
    if len(A) == 0:
        return np.zeros((0, 3)), np.zeros((0,))
    return np.array(A, dtype=float), np.array(b, dtype=float)

def control_limits(u_max):
    # |u_i| <= u_max  ->  u_i >= -u_max and -u_i >= -u_max
    A, b = [], []
    for i in range(3):
        e = np.zeros(3); e[i] = 1.0
        A.append(e);    b.append(-u_max)
        A.append(-e);   b.append(-u_max)
    return np.array(A, dtype=float), np.array(b, dtype=float)

def safety_filter(p, u_nom, obstacles, R_safe, gamma, u_max):
    A1, b1 = obstacle_halfspaces(p, obstacles, R_safe, gamma)
    A2, b2 = control_limits(u_max)

    if A1.size == 0:
        A, b = A2, b2
    else:
        A = np.vstack([A1, A2])
        b = np.hstack([b1, b2])

    return project_to_halfspaces(u_nom, A, b)

# ==============================
# Dynamics
# ==============================
def step_dynamics(pp, pe, u, v, dt):
    return pp + dt * u, pe + dt * v

# ==============================
# Game costs (min_u max_v)
# ==============================
def stage_cost(pp, pe, u, v, w_sep, Ru, Rv):
    d = pp - pe
    return w_sep * (d @ d) + Ru * (u @ u) - Rv * (v @ v)

def terminal_cost(pp, pe, wT):
    d = pp - pe
    return wT * (d @ d)

# ==============================
# Min–Max DDP
# ==============================
def minmax_ddp(
    pp0, pe0,
    dt, T,
    obstacles, R_safe, gamma, umax, vmax,
    w_sep=10.0, Ru=0.05, Rv=0.5, wT=50.0,
    n_iter=30,
    alpha_list=(1.0, 0.5, 0.25, 0.1, 0.05),
    reg0=1e-6,
    bounds=(-1.0, 8.0),
    verbose=True,
):
    """
    Returns:
      U_nom: (T,3) nominal pursuer controls (before safety)
      V_nom: (T,3) nominal evader controls (before safety)
      X_exec: (T+1,6) executed safe trajectory (after safety)
      J_hist: (n_iter+1,) total planned cost per iteration (running + terminal)
      J_time: (T+1,) distance^2 over time along executed trajectory
    """
    # initial nominal controls
    U = np.zeros((T, 3), dtype=float)
    V = np.zeros((T, 3), dtype=float)

    # seed: pursuer heads toward evader, evader runs away (helps avoid "do nothing")
    pp = pp0.copy()
    pe = pe0.copy()
    for k in range(T):
        d = pe - pp
        U[k] = 0.0 * d
        V[k] = 0.0 * d

    def rollout(U_in, V_in):
        """Rollout using safety-filtered controls; also compute J = sum l + lf."""
        X = np.zeros((T+1, 6), dtype=float)
        pp = pp0.copy()
        pe = pe0.copy()
        X[0, :3] = pp
        X[0, 3:] = pe

        J = 0.0
        for k in range(T):
            u = safety_filter(pp, U_in[k], obstacles, R_safe, gamma, umax)
            v = safety_filter(pe, V_in[k], obstacles, R_safe, gamma, vmax)

            # stage cost uses the ACTUAL applied controls (after safety)
            J += stage_cost(pp, pe, u, v, w_sep, Ru, Rv)

            pp, pe = step_dynamics(pp, pe, u, v, dt)

            # workspace clamp (hard)
            pp = np.clip(pp, bounds[0] + 1e-3, bounds[1] - 1e-3)
            pe = np.clip(pe, bounds[0] + 1e-3, bounds[1] - 1e-3)

            X[k+1, :3] = pp
            X[k+1, 3:] = pe

        J += terminal_cost(X[T, :3], X[T, 3:], wT)
        return X, float(J)

    # initial rollout
    X, Jbest = rollout(U, V)
    J_hist = [Jbest]
    reg = float(reg0)

    # constant Jacobians for single-integrator
    # x=[pp;pe], u affects pp, v affects pe
    Fx = np.eye(6)
    Fu = np.zeros((6, 3)); Fu[0:3, :] = dt * np.eye(3)
    Fv = np.zeros((6, 3)); Fv[3:6, :] = dt * np.eye(3)

    for it in range(n_iter):
        # terminal value function derivatives
        ppT = X[T, 0:3]
        peT = X[T, 3:6]
        dT = ppT - peT

        Vx = np.zeros(6)
        Vx[0:3] =  2.0 * wT * dT
        Vx[3:6] = -2.0 * wT * dT

        Vxx = np.zeros((6, 6))
        Vxx[0:3, 0:3] =  2.0 * wT * np.eye(3)
        Vxx[3:6, 3:6] =  2.0 * wT * np.eye(3)
        Vxx[0:3, 3:6] = -2.0 * wT * np.eye(3)
        Vxx[3:6, 0:3] = -2.0 * wT * np.eye(3)

        K_u = np.zeros((T, 3, 6))
        K_v = np.zeros((T, 3, 6))
        k_u = np.zeros((T, 3))
        k_v = np.zeros((T, 3))

        diverged = False

        for k in reversed(range(T)):
            pp = X[k, 0:3]
            pe = X[k, 3:6]
            d = pp - pe

            # Stage derivatives (quadratic, exact)
            lx = np.zeros(6)
            lx[0:3] =  2.0 * w_sep * d
            lx[3:6] = -2.0 * w_sep * d

            lu =  2.0 * Ru * U[k]
            lv = -2.0 * Rv * V[k]  # because stage has -Rv||v||^2

            lxx = np.zeros((6, 6))
            lxx[0:3, 0:3] =  2.0 * w_sep * np.eye(3)
            lxx[3:6, 3:6] =  2.0 * w_sep * np.eye(3)
            lxx[0:3, 3:6] = -2.0 * w_sep * np.eye(3)
            lxx[3:6, 0:3] = -2.0 * w_sep * np.eye(3)

            luu = 2.0 * Ru * np.eye(3)
            lvv = -2.0 * Rv * np.eye(3)  # negative definite (maximizer)

            lxu = np.zeros((6, 3))
            lxv = np.zeros((6, 3))
            luv = np.zeros((3, 3))

            # Q-function derivatives
            Qx  = lx  + Fx.T @ Vx
            Qu  = lu  + Fu.T @ Vx
            Qv  = lv  + Fv.T @ Vx

            Qxx = lxx + Fx.T @ Vxx @ Fx
            Quu = luu + Fu.T @ Vxx @ Fu
            Qvv = lvv + Fv.T @ Vxx @ Fv
            Qux = lxu.T + Fu.T @ Vxx @ Fx    # (3x6)
            Qvx = lxv.T + Fv.T @ Vxx @ Fx    # (3x6)
            Quv = luv   + Fu.T @ Vxx @ Fv    # (3x3)
            Qvu = Quv.T

            # Regularization:
            # - make Quu more PD for the minimizer
            # - make -Qvv more PD for the maximizer saddle solve
            Quu_reg = Quu + reg * np.eye(3)
            Qvv_reg = Qvv - reg * np.eye(3)  # pushes it "more negative"

            # Solve saddle system:
            # [ Quu  Quv ] [du] = -[Qu]
            # [ Qvu  Qvv ] [dv]   -[Qv]
            # and similarly for gains vs x:
            # [ Quu  Quv ] [Ku] = -[Qux]
            # [ Qvu  Qvv ] [Kv]   -[Qvx]
            M = np.block([[Quu_reg, Quv],
                          [Qvu,     Qvv_reg]])

            rhs_ff = -np.hstack([Qu, Qv])                 # (6,)
            rhs_K  = -np.vstack([Qux, Qvx])               # (6x6)

            try:
                sol_ff = np.linalg.solve(M, rhs_ff)       # (6,)
                sol_K  = np.linalg.solve(M, rhs_K)        # (6x6)
            except np.linalg.LinAlgError:
                diverged = True
                break

            du = sol_ff[0:3]
            dv = sol_ff[3:6]
            Ku = sol_K[0:3, :]
            Kv = sol_K[3:6, :]

            k_u[k] = du
            k_v[k] = dv
            K_u[k] = Ku
            K_v[k] = Kv

            # Value function update (standard for saddle iLQG/DDP):
            # Use compact form:
            z = np.hstack([du, dv])           # (6,)
            Kz = np.vstack([Ku, Kv])          # (6x6)
            Qz = np.hstack([Qu, Qv])          # (6,)
            Qzx = np.vstack([Qux, Qvx])       # (6x6)

            Vx  = Qx  + Qzx.T @ z + Kz.T @ Qz + Kz.T @ M @ z
            Vxx = Qxx + Qzx.T @ Kz + Kz.T @ Qzx + Kz.T @ M @ Kz

            Vxx = 0.5 * (Vxx + Vxx.T)

        if diverged:
            reg *= 10.0
            if verbose:
                print(f"MinMax-DDP iter {it:02d}: backward failed, reg -> {reg:.2e}")
            J_hist.append(Jbest)
            continue

        # Forward line search
        accepted = False
        U_best = U.copy()
        V_best = V.copy()
        X_best = X.copy()

        for alpha in alpha_list:
            pp = pp0.copy()
            pe = pe0.copy()
            X_try = np.zeros((T+1, 6), dtype=float)
            X_try[0, :3] = pp
            X_try[0, 3:] = pe
            J_try = 0.0

            for k in range(T):
                xk = X_try[k]
                dx = xk - X[k]

                u_nom = U[k] + alpha * k_u[k] + K_u[k] @ dx
                v_nom = V[k] + alpha * k_v[k] + K_v[k] @ dx

                # safety at execution
                u = safety_filter(pp, u_nom, obstacles, R_safe, gamma, umax)
                v = safety_filter(pe, v_nom, obstacles, R_safe, gamma, vmax)

                J_try += stage_cost(pp, pe, u, v, w_sep, Ru, Rv)

                pp, pe = step_dynamics(pp, pe, u, v, dt)
                pp = np.clip(pp, bounds[0] + 1e-3, bounds[1] - 1e-3)
                pe = np.clip(pe, bounds[0] + 1e-3, bounds[1] - 1e-3)

                X_try[k+1, :3] = pp
                X_try[k+1, 3:] = pe

            J_try += terminal_cost(X_try[T, :3], X_try[T, 3:], wT)

            # minimizer acceptance
            if J_try < Jbest - 1e-9:
                accepted = True
                Jbest = float(J_try)
                X_best = X_try

                # update NOMINAL sequences (before safety)
                for k in range(T):
                    dx = X_try[k] - X[k]
                    U_best[k] = U[k] + alpha * k_u[k] + K_u[k] @ dx
                    V_best[k] = V[k] + alpha * k_v[k] + K_v[k] @ dx
                break

        if accepted:
            U, V, X = U_best, V_best, X_best
            reg = max(reg0, reg / 2.0)
            if verbose:
                dist0 = np.linalg.norm(X[0, :3] - X[0, 3:])
                distT = np.linalg.norm(X[T, :3] - X[T, 3:])
                print(f"MinMax-DDP iter {it:02d}: J={Jbest:.3f}, reg={reg:.2e}, dist {dist0:.2f}->{distT:.2f}")
        else:
            reg *= 10.0
            if verbose:
                print(f"MinMax-DDP iter {it:02d}: no improvement, reg -> {reg:.2e}")

        J_hist.append(Jbest)

    # Build executed safe trajectory using final nominal U,V (for clean plotting)
    X_exec, _ = rollout(U, V)

    # time distance^2 (not the DDP objective; just a diagnostic)
    J_time = np.zeros(T+1)
    for k in range(T+1):
        d = X_exec[k, :3] - X_exec[k, 3:]
        J_time[k] = d @ d

    return U, V, X_exec, np.array(J_hist, dtype=float), J_time

# ==============================
# Plot helpers
# ==============================
def plot_scene_3d(PP, PE, obstacles, R_safe, bounds, title):
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(PP[:,0], PP[:,1], PP[:,2], lw=3, label="Pursuer")
    ax.plot(PE[:,0], PE[:,1], PE[:,2], lw=3, label="Evader")

    # start / end markers
    ax.scatter(*PP[0],  c="black",   s=90, marker="o")
    ax.scatter(*PP[-1], c="black",   s=90, marker="s")
    ax.scatter(*PE[0],  c="black", s=90, marker="^")
    ax.scatter(*PE[-1], c="black", s=90, marker="s")

    # obstacles
    uu = np.linspace(0, 2*np.pi, 30)
    vv = np.linspace(0, np.pi, 20)
    for obs in obstacles:
        c = obs["center"]
        R = obs["radius"] + R_safe
        x = c[0] + R*np.outer(np.cos(uu), np.sin(vv))
        y = c[1] + R*np.outer(np.sin(uu), np.sin(vv))
        z = c[2] + R*np.outer(np.ones_like(uu), np.cos(vv))
        ax.plot_surface(x, y, z, alpha=0.3)

    pad = 0.5
    ax.set_xlim(bounds[0]-pad, bounds[1]+pad)
    ax.set_ylim(bounds[0]-pad, bounds[1]+pad)
    ax.set_zlim(bounds[0]-pad, bounds[1]+pad)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="blue", lw=3, label="Pursuer"),
        Line2D([0], [0], color="orange",   lw=3, label="Evader"),
        Line2D([0], [0], marker="o", color="k", linestyle="None",
            markersize=8, label="Start"),
        Line2D([0], [0], marker="^", color="k", linestyle="None",
            markersize=8, label="Start"),
        Line2D([0], [0], marker="s", color="k", linestyle="None",
            markersize=8, label="End"),
    ]

    ax.legend(handles=legend_elements, loc="upper left")

    ax.set_title(title)
    plt.tight_layout()
    #plt.savefig("pic18.png", dpi=600, bbox_inches="tight")
    plt.show()

# ==============================
# Main
# ==============================
def main():
    dt = 0.05
    T  = 100
    bounds = (-1.0, 8.0)

    obstacles = [
        {"center": np.array([3.2, 3.0, 1.3]), "radius": 1.2},
        {"center": np.array([5.2, 4.7, 1.6]), "radius": 1.0},
    ]

    # Safety settings
    R_safe = 0.3
    gamma  = 6.0
    umax   = 4.0
    vmax   = 4.0

    # Initial conditions
    pp0 = np.array([1.5, 1.5, 1.0])
    pe0 = np.array([7.0, 6.2, 0.5])

    # Game weights (these matter for "evader escaping")
    # If evader still doesn't escape, reduce Rv more, or increase w_sep and wT.
    w_sep = 10.0
    Ru    = 0.01
    Rv    = 0.1   # small => evader willing to move more
    wT    = 80.0    # big => evader cares about being far at the end too

    U_nom, V_nom, X_exec, J_iter, J_time = minmax_ddp(
        pp0=pp0, pe0=pe0,
        dt=dt, T=T,
        obstacles=obstacles, R_safe=R_safe, gamma=gamma,
        umax=umax, vmax=vmax,
        w_sep=w_sep, Ru=Ru, Rv=Rv, wT=wT,
        n_iter=30,
        bounds=bounds,
        verbose=True,
    )

    PP = X_exec[:, 0:3]
    PE = X_exec[:, 3:6]

    # 3D trajectory plot (Z axis goes to 8 like you wanted)
    plot_scene_3d(
        PP, PE, obstacles, R_safe, bounds,
        title="3D CC-GT-DDP: Pursuit–Evasion"
    )

    # Cost over DDP iterations: THIS is running + terminal per iteration
    plt.figure()
    plt.plot(J_iter, lw=3)
    plt.grid(True)
    plt.xlabel("DDP iteration")
    plt.ylabel("Total cost  Σ l_k + l_f")
    plt.title("Min–Max DDP iteration cost")
    plt.show()

    # Distance^2 over time along the executed safe trajectory (diagnostic)
    plt.figure()
    plt.plot(J_time, lw=3)
    plt.grid(True)
    plt.xlabel("Iterations")
    plt.ylabel("Total cost")
    plt.title("CC-GT-DDP Iteration Cost: Pursuit-Evasion")
    plt.show()

if __name__ == "__main__":
    main()
