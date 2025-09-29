import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

# --- Params (tweak these) ---
SHIFT_FRACTION = 0.75   # 0.0 = easy min at origin-side; 1.0 = align easy min at the hard center projection
MODE = "schedule"       # "random", "fixed", or "schedule"
ETA = 0.12
DECAY = 0.999
STEPS = 800
ALPHA_FIXED = 0.5

# --- Original analytic landscape ---
init = np.array([-1, 1])
xg, yg = 2.5, 2.0  # hard basin center (for L_hard)

d = np.array([xg, yg])
d_norm = d/np.linalg.norm(d)
perp = np.array([-d_norm[1], d_norm[0]])

a_valley, b_valley = 1.0, 0.05  # across- vs along-valley curvature
k, sigma_r, A, B, C = 6.0, 1.5, 1.2, 0.02, 0.15

# --- Easy valley center: move toward hard basin along the valley direction ---
s_g = np.linalg.norm(d)           # valley coordinate of hard basin
s0  = SHIFT_FRACTION * s_g        # valley center for the easy basin (closer to hard as SHIFT_FRACTION -> 1)

def rotate_to_valley(x, y):
    p = np.stack([x, y], axis=-1)
    s = np.tensordot(p, d_norm, axes=([p.ndim-1],[0]))  # along-valley coord
    t = np.tensordot(p, perp,    axes=([p.ndim-1],[0]))  # perpendicular coord
    return s, t

def L_easy(x, y):
    s, t = rotate_to_valley(x, y)
    return a_valley * (t**2) + b_valley * ((s - s0)**2)

def grad_easy(w):
    x, y = w
    s, t = rotate_to_valley(x, y)
    return 2*a_valley*t*perp + 2*b_valley*(s - s0)*d_norm

def L_hard(x, y):
    r2 = x**2 + y**2
    ripples  = A * np.exp(-r2/(2*sigma_r**2)) * (1 - np.cos(k*x)*np.cos(k*y))
    bowl_far = C * ((x - xg)**2 + (y - yg)**2)
    confine  = B * r2
    return ripples + bowl_far + confine

def grad_hard(w):
    x, y = w
    r2 = x**2 + y**2
    E = np.exp(-r2/(2*sigma_r**2))
    cosx, sinx = np.cos(k*x), np.sin(k*x)
    cosy, siny = np.cos(k*y), np.sin(k*y)
    term = (1 - cosx*cosy)
    dE_dx = E * (-x / (sigma_r**2))
    dE_dy = E * (-y / (sigma_r**2))
    dterm_dx = k*sinx*cosy
    dterm_dy = k*siny*cosx
    dripple_dx = A*(dE_dx*term + E*dterm_dx)
    dripple_dy = A*(dE_dy*term + E*dterm_dy)
    dbowl_dx = 2*C*(x - xg)
    dbowl_dy = 2*C*(y - yg)
    dconf_dx = 2*B*x
    dconf_dy = 2*B*y
    return np.array([dripple_dx + dbowl_dx + dconf_dx,
                     dripple_dy + dbowl_dy + dconf_dy])

def grad_mix(w, alpha=0.5):
    return (1-alpha)*grad_easy(w) + alpha*grad_hard(w)

def L_mix(x, y, alpha=0.5):
    return (1-alpha)*L_easy(x, y) + alpha*L_hard(x, y)

# --- Global min finder for a given alpha (grid search) ---
def find_global_min(alpha=0.5, grid=(-1.5, 3.5), N=401):
    xs = np.linspace(grid[0], grid[1], N)
    ys = np.linspace(grid[0], grid[1], N)
    X, Y = np.meshgrid(xs, ys)
    Z = L_mix(X, Y, alpha)
    idx = np.unravel_index(np.argmin(Z), Z.shape)
    return X[idx], Y[idx], Z[idx]

# --- SGD with alpha strategies ---
rng = np.random.default_rng(0)

def best_alpha(g_e, g_h, d):
    # Search over two coefficients to maximize dot product with d
    alphas = np.linspace(0, 1, 100)  # Search range for both coefficients
    best_a, best_b, best_dot = 0.0, 0.0, -1e9
    
    for a in alphas:
        for b in alphas:
            g = a*g_e + b*g_h  # General linear combination
            dot = np.dot(g, d)  # Maximize dot product directly
            if dot > best_dot:
                best_dot, best_a, best_b = dot, a, b
    
    # Return both coefficients directly
    return best_a, best_b

def run_sgd_best_alpha(w_star, steps=200, eta=0.08, decay=0.999):
    w = init.copy()
    traj, alphas_used = [w.copy()], []
    lr = eta
    for t in range(steps):
        g_e = grad_easy(w)
        g_h = grad_hard(w)
        d = w_star - w
        # Get coefficients for linear combination
        a_e, a_h = best_alpha(g_e, g_h, -d)
        # Use both coefficients for the linear combination
        g = a_e*g_e + a_h*g_h
        w = w - lr * g
        lr *= decay
        traj.append(w.copy())
        alphas_used.append([a_e, a_h])  # Store both coefficients
    return np.array(traj), np.array(alphas_used)


def run_sgd(steps=600, eta=0.08, decay=0.999, mode="schedule", alpha_fixed=0.5):
    w = init.copy()
    traj = [w.copy()]
    lr = eta
    for t in range(steps):
        if mode == "random":
            alpha = rng.uniform(0, 1)
        elif mode == "fixed":
            alpha = alpha_fixed
        elif mode == "schedule":
            alpha = (t / steps)**2  # linear 0 -> 1
            print(alpha)
        else:
            raise ValueError("Unknown mode")
        g = grad_mix(w, alpha)
        w = w - lr * g
        lr *= decay
        traj.append(w.copy())
    return np.array(traj)

# --- Run & plot ---
alpha_ref = 0.5
xmin, ymin, zmin = find_global_min(alpha_ref)
print(f"Easy valley center s0={s0:.3f} (SHIFT_FRACTION={SHIFT_FRACTION}); global min at α={alpha_ref}: ({xmin:.3f}, {ymin:.3f}), loss={zmin:.4f}")

MODE = "random"
ETA = 0.2
DECAY = 1
STEPS = 20
ALPHA_FIXED = 0.5

traj_random = run_sgd(steps=STEPS, eta=ETA, decay=DECAY, mode=MODE, alpha_fixed=ALPHA_FIXED)

x_star, y_star, z_star = find_global_min(alpha=0.5)
w_star = np.array([x_star, y_star])

traj_guided, alphas = run_sgd_best_alpha(w_star, steps=20, eta=0.15)

grid = np.linspace(-1.5, 3.5, 240)
X, Y = np.meshgrid(grid, grid)
Z_mid = L_mix(X, Y, alpha_ref)

plt.figure(figsize=(6.7,5.6))
cs = plt.contour(X, Y, Z_mid, levels=50, cmap="Set2", linewidths=1)
plt.clabel(cs, inline=1, fontsize=7, fmt="%.1f")
#Plot trajectory segments with colors based on which gradient dominates
plt.plot(traj_guided[:,0], traj_guided[:,1], 'o-', c='black', ms=3)#, label=f"SGD (mode={MODE})")
for i in range(len(traj_guided)-1):
    # Get the weights for this step
    a_e, a_h = alphas[i]
    # Blue for easy gradient dominance, red for hard gradient dominance
    color = 'blue' if a_e > a_h else 'red'
    plt.plot(traj_guided[i:i+2,0], traj_guided[i:i+2,1], '-', c=color, linewidth=1.5)
#Plot points on top
plt.plot(traj_random[:,0], traj_random[:,1], 'o-', c='black', ms=3)#, label=f"SGD (mode={MODE})")
plt.scatter([init[0]], [init[1]], c='black', marker='o', s=3)#, label="Init")
#plt.scatter([xmin],[ymin], c='red', marker='*', s=150, label="Global min (α=0.5)")
# Mark the easy basin center point (project s0 back to xy)
easy_center_xy = d_norm * s0
#plt.scatter([easy_center_xy[0]],[easy_center_xy[1]], c='cyan', edgecolor='k', marker='D', s=70, label="Easy basin center")
plt.xlabel("w1"); plt.ylabel("w2")
plt.title("Shifted Easy Basin Toward Hard Basin")
plt.legend(); plt.grid(False); plt.tight_layout()
plt.show()
