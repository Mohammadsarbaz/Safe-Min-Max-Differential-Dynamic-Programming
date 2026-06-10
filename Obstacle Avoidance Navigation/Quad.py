import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.stats import norm
import numpy as np

##############################################################################
#                   GLOBAL PARAMETERS
##############################################################################

HORIZON           = 80      # single‐shot DDP horizon
Initial_STEPS_MPC = 200     # total control‐step budget
APPLY_STEPS       = 10      # how many controls to apply per MPC update
STEPS_MPC         = math.ceil(Initial_STEPS_MPC / APPLY_STEPS)
MAX_ITER_DDP      = 80      # DDP inner iterations
DT                = 0.01    # time step
SIGMA_W           = 1e-5 * np.eye(12)  # plant disturbance covariance
GOAL_THRESH       = 0.1     # [m] goal‐tolerance

##############################################################################
#                   Quadcopter Dynamics
##############################################################################

def quad_dynamics_nominal(x, u_net, dt=DT, g=9.81, m=1.0,
                          Ix=0.02, Iy=0.02, Iz=0.04):
    px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r = x
    F, Mx, My, Mz = u_net

    # translational accel in body axes
    ax = (-F/m)*(np.sin(phi)*np.sin(theta))
    ay = ( F/m)*(np.sin(phi)*np.cos(theta))
    az = ( F/m)*(np.cos(phi)*np.cos(theta)) - g

    # Euler rates
    phi_dot   = p + q*np.sin(phi)*np.tan(theta) + r*np.cos(phi)*np.tan(theta)
    theta_dot = q*np.cos(phi) - r*np.sin(phi)
    psi_dot   = (q*np.sin(phi) + r*np.cos(phi))/max(np.cos(theta),1e-6)

    # rotational accel
    p_dot = Mx / Ix
    q_dot = My / Iy
    r_dot = Mz / Iz

    # integrate
    px_n    = px   + dt*vx
    py_n    = py   + dt*vy
    pz_n    = pz   + dt*vz
    vx_n    = vx   + dt*ax
    vy_n    = vy   + dt*ay
    vz_n    = vz   + dt*az
    phi_n   = phi  + dt*phi_dot
    theta_n = theta+ dt*theta_dot
    psi_n   = psi  + dt*psi_dot
    p_n     = p    + dt*p_dot
    q_n     = q    + dt*q_dot
    r_n     = r    + dt*r_dot

    return np.array([px_n, py_n, pz_n,
                     vx_n, vy_n, vz_n,
                     phi_n, theta_n, psi_n,
                     p_n, q_n, r_n], dtype=float)

def quad_dynamics_noisy(x, u_net):
    """Apply nominal step + add Gaussian noise."""
    x_nom = quad_dynamics_nominal(x, u_net)
    noise = 0.1*np.random.multivariate_normal(np.zeros(12), SIGMA_W * DT)
    return x_nom + noise

def quad_linearize_12(x, u_net):
    """Numerical linearization of the nominal dynamics."""
    n_state = 12
    n_ctrl  = 4
    eps     = 1e-5
    fx = quad_dynamics_nominal(x, u_net)
    A  = np.zeros((n_state, n_state))
    B  = np.zeros((n_state, n_ctrl))
    for i in range(n_state):
        xp     = x.copy(); xp[i] += eps
        A[:,i] = (quad_dynamics_nominal(xp, u_net) - fx) / eps
    for j in range(n_ctrl):
        up     = u_net.copy(); up[j] += eps
        B[:,j] = (quad_dynamics_nominal(x, up) - fx) / eps
    return A, B

##############################################################################
#                   Min-Max Cost
##############################################################################

def running_cost_minmax(x, u_min, v_max, x_goal, Q, Ru, Rv):
    e = x - x_goal
    return e.T@Q@e + u_min.T@Ru@u_min - v_max.T@Rv@v_max

def terminal_cost_minmax(x, x_goal, Qf):
    e = x - x_goal
    return e.T@Qf@e

##############################################################################
#                Chance Constraints
##############################################################################

def sphere_chance_constraint(x, c, r, Σ, β=0.995):
    d    = x[:3] - c
    dist = max(np.linalg.norm(d), 1e-12)
    grad = np.zeros(12); grad[:3] = d/dist
    z    = norm.ppf(β)
    m    = z*np.sqrt(grad@Σ@grad)
    return (r + m) - dist

def multiple_obstacles_penalty(x, Σ, obs_list):
    α = 1e9
    g_sum, H_sum = np.zeros(12), np.zeros((12,12))
    for obs in obs_list:
        g = sphere_chance_constraint(x, obs["center"], obs["radius"], Σ)
        if g > -1e-4:
            d = x[:3] - obs["center"]
            dist = max(np.linalg.norm(d),1e-12)
            gd = np.zeros(12); gd[:3] = d/dist
            g_sum += α * g * (-gd)
            H_sum += α * np.outer(-gd, -gd)
    return g_sum, H_sum

def closed_loop_covariance(Ku, Kv, Σw, X, U, V):
    Σs = [np.zeros((12,12))]
    Σ  = Σs[0]
    for k in range(HORIZON):
        A,B = quad_linearize_12(X[k], U[k]+V[k])
        K   = Ku[k] + Kv[k]
        Σ   = (A+B@K)@Σ@(A+B@K).T + Σw
        Σs.append(Σ)
    return Σs

##############################################################################
#   Single‐Shot Min‐Max DDP
##############################################################################

def single_shot_forced_minmax_ddp(x0, x_goal, obstacles, Σw):
    n_ctrl = 4
    Q  = np.diag([1,1,1]+[0.1]*9)
    Ru = np.diag([0.01]*4); Rv = np.diag([10]*4)
    Qf = np.diag([5,5,5]+[1]*9)

    # initialize feedforward/disturbance
    U = np.zeros((HORIZON, n_ctrl))
    V = np.zeros((HORIZON, n_ctrl))

    # nominal rollout helper
    def rollout_nom(x, U_, V_):
        traj = [x.copy()]
        for k in range(HORIZON):
            x = quad_dynamics_nominal(x, U_[k]+V_[k])
            traj.append(x)
        return np.array(traj)

    X  = rollout_nom(x0, U, V)
    Ku = [np.zeros((n_ctrl,12)) for _ in range(HORIZON)]
    Kv = [np.zeros((n_ctrl,12)) for _ in range(HORIZON)]

    for _ in range(MAX_ITER_DDP):
        Σs = closed_loop_covariance(Ku, Kv, Σw, X, U, V)
        Vx  = 2*Qf@(X[-1]-x_goal)
        Vxx = 2*Qf.copy()

        new_Ku, new_du = [None]*HORIZON, [None]*HORIZON
        new_Kv, new_dv = [None]*HORIZON, [None]*HORIZON

        # backward
        for k in reversed(range(HORIZON)):
            A,B = quad_linearize_12(X[k], U[k]+V[k])
            lx,lu,lv = 2*Q@(X[k]-x_goal), 2*Ru@U[k], -2*Rv@V[k]
            lxx,luu,lvv = 2*Q,2*Ru,-2*Rv
            pg,Ph        = multiple_obstacles_penalty(X[k], Σs[k], obstacles)
            lx += pg; lxx += Ph

            Qx  = lx + A.T@Vx
            Qu  = lu + B.T@Vx
            Qv  = lv + B.T@Vx
            Qxx = lxx + A.T@Vxx@A
            Quu = luu + B.T@Vxx@B
            Qvv = lvv + B.T@Vxx@B
            Qux = B.T@Vxx@A
            Qvx = B.T@Vxx@A

            Quu_r = Quu + 1e-9*np.eye(n_ctrl)
            Qvv_r = Qvv - 1e-9*np.eye(n_ctrl)

            du = -np.linalg.solve(Quu_r, Qu)
            dv = -np.linalg.solve(Qvv_r, Qv)
            Ku_k = -np.linalg.solve(Quu_r, Qux)
            Kv_k = -np.linalg.solve(Qvv_r, Qvx)

            Vx  = Qx + Ku_k.T@(Quu@du+Qu)+Qux.T@du + Kv_k.T@(Qvv@dv+Qv)+Qvx.T@dv
            Vxx = Qxx + Ku_k.T@Quu@Ku_k + Ku_k.T@Qux + Qux.T@Ku_k \
                    + Kv_k.T@Qvv@Kv_k + Kv_k.T@Qvx + Qvx.T@Kv_k

            new_Ku[k], new_du[k] = Ku_k, du
            new_Kv[k], new_dv[k] = Kv_k, dv

        # line‐search forward
        best = None
        for α in [1.0,0.5,0.2,0.1]:
            x    = x0.copy()
            cost = 0.0
            traj = [x]; Us, Vs = [], []
            for k in range(HORIZON):
                du = α*new_du[k] + new_Ku[k]@(x - X[k])
                dv = α*new_dv[k] + new_Kv[k]@(x - X[k])
                u_hat = U[k] + du
                v_hat = V[k] + dv
                x     = quad_dynamics_nominal(x, u_hat+v_hat)
                traj.append(x)
                cost += running_cost_minmax(x, u_hat, v_hat, x_goal, Q, Ru, Rv)
                Us.append(u_hat); Vs.append(v_hat)
            cost += terminal_cost_minmax(x, x_goal, Qf)
            if best is None or cost<best[0]:
                best=(cost,np.array(traj),np.array(Us),np.array(Vs))

        _,X,U,V = best

    return X, U+V, best[0]

##############################################################################
#   MPC: Receding Horizon
##############################################################################

def mpc_safe_minmax_ddp_forced(x0, x_goal, obstacles):
    real_states = []
    x_current   = x0.copy()
    for t in range(STEPS_MPC):
        print(f"  MPC step {t+1}/{STEPS_MPC}: pos=({x_current[0]:.3f},{x_current[1]:.3f},{x_current[2]:.3f})")
        Xddp, net, _ = single_shot_forced_minmax_ddp(x_current, x_goal, obstacles, Σw=SIGMA_W)
        for i in range(APPLY_STEPS):
            x_current = quad_dynamics_noisy(x_current, net[i])
            real_states.append(x_current.copy())
        if np.linalg.norm(x_current[:3]-x_goal[:3])<GOAL_THRESH:
            print(f"    → reached goal at MPC step {t+1}")
            break
    return np.vstack([x0, np.array(real_states)]), None

##############################################################################
#                               MAIN
##############################################################################

def main():
    x0     = np.zeros(12)
    x_goal = np.array([2.0,2.0,2.0]+[0]*9)
    obstacles = [
        {"center": np.array([0.75,0.75,1.0]),"radius":0.3},
        {"center": np.array([1.0,1.0,1.5]), "radius":0.3},
        {"center": np.array([0.5,0.5,0.5]), "radius":0.3},
        {"center": np.array([1.5,1.5,1.75]),"radius":0.3},
    ]

    # Monte Carlo
    num_mc=20
    all_T=[]
    for run in range(num_mc):
        print(f"\n=== Monte Carlo run {run+1}/{num_mc} ===")
        traj, _ = mpc_safe_minmax_ddp_forced(x0, x_goal, obstacles)
        all_T.append(traj[:,:3])
    all_T = np.stack(all_T,axis=0)
    
    
        # ----------------------------------------------------------------------
    # Compute Monte Carlo Metrics
    # ----------------------------------------------------------------------
    reach_thresh  = 0.2      # e.g. 0.1 m
    safety_margin = 0.05             # extra clearance from obstacle surface

    num_mc = all_T.shape[0]
    safety_count, reach_count, success_count = 0, 0, 0
    rmsd_vals = np.zeros(num_mc)

    # Loop over each MC trajectory
    for j in range(num_mc):
        traj_xyz = all_T[j]       # shape (T,3)

        # 1) Safety: all pts >= (radius + margin) from every obstacle
        is_safe = True
        for pt in traj_xyz:
            for obs in obstacles:
                if np.linalg.norm(pt - obs["center"]) < obs["radius"] + safety_margin:
                    is_safe = False
                    break
            if not is_safe:
                break

        # 2) Reachability: final pt within reach_thresh of goal
        final_dist = np.linalg.norm(traj_xyz[-1] - x_goal[:3])
        is_reach   = (final_dist <= reach_thresh)

        # 3) Success = safe AND reach
        safety_count  += is_safe
        reach_count   += is_reach
        success_count += (is_safe and is_reach)

        # 4) RMSD: root‐mean‐square distance to goal over full traj
        dists       = np.linalg.norm(traj_xyz - x_goal[:3], axis=1)
        rmsd_vals[j] = np.sqrt(np.mean(dists**2))

    # 5) Total state variance (positions only)
    all_pts = all_T.reshape(-1, 3)    # (num_mc * T, 3)
    total_state_variance = np.sum(np.var(all_pts, axis=0))

    # percentages & average
    safety_rate   = safety_count  / num_mc * 100
    reach_rate    = reach_count   / num_mc * 100
    success_rate  = success_count / num_mc * 100
    mean_RMSD     = np.mean(rmsd_vals)

    print("\n=== Monte Carlo Metrics ===")
    print(f"Safety:               {safety_rate:5.1f}%")
    print(f"Reachability:         {reach_rate:5.1f}%")
    print(f"Success:              {success_rate:5.1f}%")
    print(f"Mean RMSD (m):        {mean_RMSD:.4f}")
    print(f"Total State Variance: {total_state_variance:.4f}\n")


    mean_T = all_T.mean(axis=0)
    std_T  = all_T.std(axis=0)
    T = mean_T.shape[0]

    # Plot mean ±σ
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot mean trajectory
    ax.plot(mean_T[:,0], mean_T[:,1], mean_T[:,2], 'b.-', label='Mean Trajectory')

    # Plot ±σ error bars
    for k in range(T):
        ax.plot([mean_T[k,0]-std_T[k,0], mean_T[k,0]+std_T[k,0]], [mean_T[k,1]]*2, [mean_T[k,2]]*2, 'r-', alpha=0.6)
        ax.plot([mean_T[k,0]]*2, [mean_T[k,1]-std_T[k,1], mean_T[k,1]+std_T[k,1]], [mean_T[k,2]]*2, 'r-', alpha=0.6)
        ax.plot([mean_T[k,0]]*2, [mean_T[k,1]]*2, [mean_T[k,2]-std_T[k,2], mean_T[k,2]+std_T[k,2]], 'r-', alpha=0.6)

    # Start and Goal points
    ax.scatter(mean_T[0,0],  mean_T[0,1],  mean_T[0,2],  color='green', s=100, label='Start')
    ax.scatter(mean_T[-1,0], mean_T[-1,1], mean_T[-1,2], color='red',   s=100, label='Goal')

    # Plot obstacles
    θ = np.linspace(0,2*np.pi,30)
    φ = np.linspace(0,np.pi,15)
    for obs in obstacles:
        X = obs["center"][0] + obs["radius"]*np.outer(np.cos(θ), np.sin(φ))
        Y = obs["center"][1] + obs["radius"]*np.outer(np.sin(θ), np.sin(φ))
        Z = obs["center"][2] + obs["radius"]*np.outer(np.ones_like(θ), np.cos(φ))
        ax.plot_surface(X, Y, Z, color='r', alpha=0.3)

    ax.set_box_aspect((1,1,1))
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("3D Monte Carlo Mean and Standard Variance ±σ")
    ax.legend()
    plt.show(block=True)

if __name__=="__main__":
    main()
