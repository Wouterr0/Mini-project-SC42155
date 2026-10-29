import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from dataclasses import dataclass
import scipy.io
mat = scipy.io.loadmat('matrices.mat')

@dataclass
class Params:
    # heating params
    C_e: float  = 0.0   # Capacity (Coulombs) (not used in this model)
    C_w: float  = 50.0 # Thermal capacity of the water (Joules per degree Celsius)
    T_e: float  = 10.0  # Environment temperature (Celsius)
    R_e: float  = 1.0  # Resistance (Ohms)
    R_w: float  = 1.1   # Thermal resistance of the water to the environment (degree Celsius per Watt)
    T_ref: float = 80.0  # Reference temperature (Celsius)
    u_h: float  = 12.0  # Heater voltage (Volts)
    epsilon: float = 4  # Hysteresis band (Celsius)

    # moving params
    Ax = mat['Ax']
    Bx = mat['Bx']
    Ay = mat['Ay']
    By = mat['By']
    dx = .2
    dy = .2
    u_m = .1 # moving voltage (both left and right, neg. voltage for ccw turning)
    meps = 0.001

def f(x, u_m, q, p):
    """Vector field dx = f(x, u) of the active mode q.

    Inputs:
        x: state vector
        u: input vector
        q[0]: heating active mode (1 for heater on, 0 for heater off)
        q[1]: moving active mode (0 for right, 1 for left, 2 for up, 3 for down, 4 for idle)
        p: parameters (instance of Params)

    Outputs:
        dx: time derivative of the state vector
    """

    # heating part
    T_w = x[0]
    if q[0] == 1:  # Heater is on
        dT_w = p.u_h**2 /(p.R_e * p.C_w)  - (T_w - p.T_e) / (p.C_w * p.R_w)
    else:  # Heater is off
        dT_w = - (T_w - p.T_e) / (p.C_w * p.R_w)

    # moving part
    x_x = x[1:4]
    x_y = x[4:7]
    match q[1]:
        case 0: u_m = [1, 1]      # right
        case 1: u_m = [-1, -1]    # left
        case 2: u_m = [1, -1]     # up
        case 3: u_m = [-1, 1]     # down
        case 4: u_m = [0, 0]      # not moving
    u_m = p.u_m*np.array(u_m)
    dx_x = p.Ax @ x_x + p.Bx @ u_m
    dx_y = p.Ay @ x_y + p.By @ u_m
    return np.array([dT_w, *dx_x, *dx_y])

def make_events(q, u, p):
    """Guard functions of mode q, in the form solve_ivp expects: g(t, x).
    Each has .terminal (stop integration) and .direction (which sign change
    counts). Built per mode, because the active guard depends on q.

    Inputs:
        q: active mode (1 for heater on, 0 for heater off)
        u: input vector
        p: parameters (instance of Params)

    Outputs:
        events: list of guard functions g(t, x) that return a scalar value.
            Each guard function represents a condition that, when met, will trigger a mode change.
            Properties of each guard function:
                .terminal: If True, the integration will stop when this event occurs.
                .direction: Specifies the direction of the zero crossing that will trigger the event.
    """
    gs = []

    # heating modes
    if q[0] == 1:  # Heater is on
        # Trigger guard when the water is warm enough (T_w crosses T_ref from below)
        def g(t, x):
            T_w = x[0]
            return T_w - (p.T_ref + p.epsilon)
        g.direction = 1  # only count upward crossings
        g.terminal = True

    else:  # Heater is off
        # Trigger guard when the water is too becoming too cold (T_w crosses T_ref from above)
        def g(t, x):
            T_w = x[0]
            return T_w - (p.T_ref - p.epsilon)
        g.direction = -1  # only count downward crossings
        g.terminal = True

    gs.append(g)

    if 1:
        g = lambda t,x: x[0]
        g.direction = -1
        g.terminal = True
        gs.append(g)

        g = lambda t,x: x[0] - p.dx
        g.direction = 1
        g.terminal = True
        gs.append(g)

        g = lambda t,x: x[4]
        g.direction = -1
        g.terminal = True
        gs.append(g)

        g = lambda t,x: x[4] - p.dy
        g.direction = 1
        g.terminal = True
        gs.append(g)

    return gs

def jump(x, q, u, p):
    """Reset map of mode q.

    Inputs:
        x: state vector
        q: active mode (1 for heater on, 0 for heater off)
        u: input vector
        p: parameters (instance of Params)

    Outputs:
        x: updated state vector
    """
    # No state reset in this model, so just return the state unchanged
    return x

def initial_mode(x0, u0, p):
    """Determine the initial mode based on the initial state and input.

    Inputs:
        x0: initial state vector
        u0: initial input vector
        p: parameters (instance of Params)

    Outputs:
        q: initial mode (1 for heater on, 0 for heater off)
    """
    q = [0, 4] # stay still until heated up

    T_w = x0[0]
    if T_w < p.T_ref:
        q[0] = 1  # Heater on
    else:
        q[1] = 0  # Heater off

    q[1] = 4 # stay still
    return q

def simulate_hybrid_system(x0, u, p, t_span, max_events=1000):
    # teval = np.linspace(t_span[0], t_span[1], 1000)  # Time points for evaluation
    t_start, t_end = t_span
    t = [t_start]
    x = [x0]
    q_t = [t_start]
    q = [initial_mode(x0, u[0], p)]

    mode_idx = 0

    while t[-1] < t_end:
        current_q = q[-1]
        current_x = x[-1]
        # print(f'current t,q,x: {t[-1]:.2f}', current_q, current_x)
        events = make_events(current_q, u, p)   # each must have .terminal = True

        sol = solve_ivp(lambda t,x: f(x, u, current_q, p),
                        [t[-1], min(t_end, np.ceil(t[-1]+0.01))], # end sim every sec so a new input can be chosen
                        current_x,
                        events=events,
                        dense_output=True)#,
                        # t_eval=teval[teval >= t[-1]])  # Evaluate only at future time points

        # Check all event functions, not only the first
        hit = [i for i, te in enumerate(sol.t_events) if te.size > 0]

        if hit:
            t_event = sol.t_events[hit[0]][0]
            x_event = sol.y_events[hit[0]][0]
            new_x = jump(x_event, current_q, u, p) # < --- Apply the jump map to get the new state after the event

            new_q = [1 - current_q[0], current_q[1]]

            # Deliberate duplicate time: pre-jump state is already stored,
            # now store the post-jump state at the same time (shows the discontinuity)
            t.extend(sol.t[sol.t < t_event])
            x.extend(sol.y.T[sol.t < t_event])
            t.append(t_event)
            x.append(new_x)
            q.append(new_q)
            q_t.append(t_event)

            if len(q) >= max_events:
                print("Maximum number of events reached (possible Zeno behavior).")
                break
        else:
            # every 1s the following is run
            t.extend(sol.t[1:])
            x.extend(sol.y.T[1:])
            new_x = x[-1]
            new_mov = q[-1][1] # old moving active mode
            if q[-1][1] == 4: # if currently idle
                new_mov = 4 # stay idle
                if mode_idx < len(u) and x[-1][0] > (p.T_ref - p.epsilon):
                    mode_idx += 1
                    new_mov = u[mode_idx]
            else:
                mode_idx += 1
                if mode_idx < len(u):
                    new_mov = u[mode_idx]
                else:
                    new_mov = 4
            print(mode_idx)
            new_q = [q[-1][0], # stay heating / cooling
                     new_mov]
            t.append(t[-1])
            x.append(new_x)
            q.append(new_q)
            q_t.append(t_end)

            if np.abs(t[-1] - t_end) < 0.01:
                break

    return np.array(t), np.array(x), np.array(q), np.array(q_t)

# run the simulation
p   = Params()
x0  = np.array([p.T_e - 5.0,   # Initial temperature of the water
                p.dx/2, 0, 0,  # init x, x_dot, x_dot_dot
                p.dy/2, 0, 0]) # init y, y_dot, y_dot_dot
u   = np.array([4,0,3,0,0,2,2,2,2,1,2,1,2,1,3,1,3,1,3,3,3,3,0,0,2])  # No external input in this model
t_span = (0, 100)  # Time span for the simulation

simulation_results = simulate_hybrid_system(x0, u, p, t_span)

# Plot the results
t, x, q, q_t = simulation_results

plt.plot(x[:, 1], x[:, 4], lw=1)
plt.xlabel('x displacement (m)')
plt.ylabel('y displacement (m)')
# plt.plot(t, x[:, 0], lw=1)
plt.tight_layout()
plt.savefig('house.pdf')
plt.show()

if 0:
    fig, (ax1, ax2) = plt.subplots(figsize=(10, 5), nrows=2, ncols=1, sharex=True)
    # Plot temperature
    ax1.plot(t, x[:, 0], label='Water Temperature (T_w)', color='blue')
    ax1.axhline(p.T_ref, color='red', linestyle='--', label='Reference Temperature (T_ref)')
    ax1.axhline(p.T_ref + p.epsilon, color='orange', linestyle='--', label='Upper Hysteresis Bound')
    ax1.axhline(p.T_ref - p.epsilon, color='orange', linestyle='--', label='Lower Hysteresis Bound')
    ax1.set_title('Water Temperature Over Time')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Temperature (°C)')
    ax1.grid()
    ax1.legend()
    # Plot mode
    ax2.step(q_t, q, where='post', label='Heater Mode (q)', color='green')
    ax2.set_title('Heater Mode Over Time')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Mode (1=On, 0=Off)')
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1], ['Off', 'On'])
    ax2.set_xticks(np.round(t))
    # Change ticks orientation to vertical for better readability
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.legend()
    ax2.grid()
    ax2.set_xlim(t_span)
    plt.tight_layout()
    plt.show()
