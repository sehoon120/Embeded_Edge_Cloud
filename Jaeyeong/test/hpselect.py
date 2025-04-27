from __future__ import print_function
#Python 3 스타일의 print 함수 사용가능.
import numpy as np
import matplotlib.pyplot as plt
thresh = 0.15 # neuronal threshold (V)
delta_t = 2**(-10) #s
tau1, tau2 = 25e-3, 2.5e-3 #s *********** CM
const = tau1 / (tau1 - tau2)
decay1 = np.exp(-delta_t/tau1)
decay2 = np.exp(-delta_t/tau2)

def act_fun(thresh, mem):
	if mem > thresh:
		return 1
	else:
		return 0

def mem_update(x, mem1, mem2, spike):
    mem1 = mem1 * decay1 * (1. - spike) + const * 0.1 * x
    mem2 = mem2 * decay2 * (1. - spike) + const * 0.1 * x
    mem = mem1 - mem2
    spike = act_fun(thresh, mem)
    return mem, mem1, mem2, spike

import time
import sys

mem1 = 0
mem2 = 0.01
mem_list = list([])
spike = 0
x_list = list([])
spike_list = list([])    


T = 100
for step in range(T):
  if step % 3 == 0:
    x = 1
  elif (step % 2 == 0) and (step > 60):
    x = 1
  else:
    x = 0
  mem, mem1, mem2, spike = mem_update(x, mem1, mem2, spike)
  print(f"x: {x}, mem: {mem}, mem1: {mem1}, mem2: {mem2} spike: {spike}")
  mem_list.append(mem)
  #y_list.append(y_select(spike))
  x_list.append(x)
  spike_list.append(spike)


plt.subplot(3, 1, 1)
plt.title("SRM Model", loc = 'left')
plt.ylabel("Membrane Potential")
plt.xlabel("Time")
plt.axhline(y=float(thresh), color='g', linestyle='--')
plt.plot(mem_list, marker = 'o', ms = 2)

plt.subplot(3, 1, 2)
plt.ylim(0.9, 1.1)
plt.ylabel("Input Spike")
plt.xlabel("Time")
plt.plot(x_list, '|', color='k')

plt.subplot(3, 1, 3)
plt.ylim(0.9, 1.1)
plt.ylabel("Output Spike")
plt.xlabel("Time")
plt.plot(spike_list, '|', color='k')

plt.show()

mem_list.clear()
