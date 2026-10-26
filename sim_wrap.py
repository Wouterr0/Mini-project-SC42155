import numpy as np
import time
import matplotlib.pyplot as plt

T_n = 200
T_b = 60
T_e = 20

rho = 1240
c_p = 1900
k = 0.15
alpha = k/(rho*c_p)

d = .4e-3
h_air = 10
h_bed = 80

# v = 50e-3
v = 5e-3

l = 0.1
n_i = 0
dl = .1e-3
N = int(l//dl)+1
T = np.full(N, np.nan)
t = 0
dt = 0.01
end_time = 5

print('CFL:', alpha*dt/dl**2)

f = lambda T: 3*h_air/(rho*c_p*d)*(T_e - T) + h_bed/(rho*c_p*d)*(T_b - T)

def time_step():
    global T, l, n_i
    # print(n_i, N)

    # update the nodes
    for i in range(1, n_i):
        dT = alpha * (T[i-1] - 2*T[i] + T[i+1])/dl**2 + f(T[i])
        T[i] += dT*dt

    # set boundary condition at nozzle T(l,t) = T_n if nozzle is dispensing
    # otherwise T(l,t) = T(0,t)
    n_i_new = int(v*t/dl)
    # print(n_i, n_i_new)
    # print(n_i_new)
    if n_i_new < N:
        # create newly deposited filament (hot)
        for i in range(n_i, n_i_new+1):
            T[i] = T_n
        n_i = n_i_new

        # set boundary condition at x=0; T(0,t) = T(1,t)
        T[0] = T[1]
    else:
        # make laplacian wrap-around
        dT = alpha * (T[N-1] - 2*T[0] + T[1])/dl**2 + f(T[0])
        T[0] = T[N-1] = T[N-1] + dT*dt



    # print(T)
    # print()

dT_save = 0.1
T_save = np.full((int(50//dT_save)+1, N), np.nan)
n_save = 0

while t < 50:
    if t//dT_save >= n_save:
        # print(t, n_save)
        T_save[n_save, :] = T.copy()
        n_save += 1

    time_step()
    t += dt

    # print(t)
    # time.sleep(dt)

fig = plt.figure(figsize=(6,4))
e = (0, (N-1)*dl*1e3, t, 0)
plt.imshow(T_save, cmap='plasma', extent=e, aspect='auto')
cbar = plt.colorbar()
cbar.set_label('temperature (°C)')
cs = plt.contour(T_save, origin='upper', extent=e, colors='k', linewidth=.4)
cbar.add_lines(cs)
plt.xlabel('x dimension (mm)')
plt.ylabel('time (s)')
plt.tight_layout()
plt.savefig('sim_wrap.pdf')
plt.show()
