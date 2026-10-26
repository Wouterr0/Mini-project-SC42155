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
v = 50e-3

l = 0.1
n_i = 0
dl = .1e-2
N = int(l//dl)+1
T = np.full(N, np.nan)
t = 0
dt = 0.1
end_time = 5

print(alpha*dt/dl**2)

def time_step():
    global T, l, n_i

    # update the nodes
    for i in range(1, n_i-0):
        f = 3*h_air/(rho*c_p*d)*(T_e - T[i]) + h_bed/(rho*c_p*d)*(T_b - T[i])
        dT = alpha * (T[i-1] - 2*T[i] + T[i+1])/dl**2 + f
        T[i] += dT*dt

    # set boundary condition at x=0; T(0,t) = T(1,t)
    T[0] = T[1]

    # set boundary condition at nozzle T(l,t) = T_n if nozzle is dispensing
    # otherwise T(l,t) = T(l-1,t)
    n_i_new = int(v*t/dl)
    # print(n_i_new)
    if n_i_new < N:
        # print(v*t/dl)
        for i in range(n_i, n_i_new+1):
            T[i] = T_n
        n_i = n_i_new
    else:
        T[N-1] = T[N-2]

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
e = (0, (N-1)*dl*1e3, t, 0)
plt.imshow(T_save, cmap='plasma', extent=e, aspect='auto')
cbar = plt.colorbar()
cbar.set_label('temperature (°C)')
plt.xlabel('x dimension (mm)')
plt.ylabel('time (s)')
plt.show()
