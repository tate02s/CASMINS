import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rates = [rate for rate in pd.read_csv('DGS10.csv')['DGS10'] if rate != '.']

class VasicekModel:
  def __init__(self, rates):
    self.rates = rates


  def VasicekNextRate(self, r, kappa, theta, sigma, dt=1/252):
    # Implements above closed form solution
    val1 = np.exp(-1*kappa*dt)
    val2 = (sigma**2)*(1-val1**2) / (2*kappa)
    out = r*val1 + theta*(1-val1) + (np.sqrt(val2))*np.random.normal()
    return out
  # Vasicek simulation short rate

  def VasicekSim(self, N, r0, kappa, theta, sigma, dt = 1/252):
    short_r = [0]*N # Create array to store rates
    short_r[0] = r0 # Initialize rates at $r_0$
    for i in range(1,N):
      short_r[i]=self.VasicekNextRate(short_r[i-1],kappa, theta, sigma, dt)
    return short_r

  # Vasicek multi-simulation
  def VasicekMultiSim(self, M, N, r0, kappa, theta, sigma, dt = 1/252):
    sim_arr = np.ndarray((N, M))
    for i in range(0,M):
      sim_arr[:, i] = self.VasicekSim(N, r0, kappa, theta, sigma, dt)

    forcastedRate = np.mean([rate[-1] for rate in sim_arr])
    return [sim_arr, forcastedRate]

  # Maximum Likelihood Estimation to calibrate parameters
  def VasicekCalibration(self, rates, dt=1/252):
    n = len(rates)
    Ax = sum(rates[0:(n-1)])
    Ay = sum(rates[1:n])
    Axx = np.dot(rates[0:(n-1)], rates[0:(n-1)])
    Axy = np.dot(rates[0:(n-1)], rates[1:n])
    Ayy = np.dot(rates[1:n], rates[1:n])
    theta = (Ay * Axx - Ax * Axy) / (n * (Axx - Axy) - (Ax**2 - Ax*Ay))
    kappa = -np.log((Axy - theta * Ax - theta * Ay + n * theta**2) / (Axx - 2*theta*Ax + n*theta**2)) / dt
    a = np.exp(-kappa * dt)

    print(f'mean reversion coefficient: {kappa}, equilibrium rate: {a}, theta: {theta}')

    sigmah2 = (Ayy - 2*a*Axy + a**2 * Axx - 2*theta*(1-a)*(Ay - a*Ax) + n*theta**2 * (1-a)**2) / n
    sigma = np.sqrt(sigmah2*2*kappa / (1-a**2))
    r0 = rates[n-1]
    return [kappa, theta, sigma, r0]

VM = VasicekModel(rates)

params = VM.VasicekCalibration([float(r)/100 for r in rates])
kappa = params[0]
theta = params[1]
sigma = params[2]
r0 = params[3]
years = 1
N = years * 252
t = np.arange(0,N)/252


test_sim = VM.VasicekSim(N, r0, kappa, theta, sigma, 1/252)
plt.figure(figsize=(10,5))
plt.plot(t,test_sim, color='r')
plt.show()


M = 100000

# rates_arr = VasicekMultiSim(M, N, r0, kappa, theta, sigma)
# plt.figure(figsize=(10,5))
# plt.plot(t,rates_arr)
# plt.hlines(y=theta, xmin = -100, xmax=100, zorder=10, linestyles = 'dashed', label='Theta')
# plt.annotate('Theta', xy=(1.0, theta+0.0005))
# plt.xlim(-0.05, 1.05)
# plt.ylabel("Rate")
# plt.xlabel("Time (year)")
# plt.title('Long run (mean reversion level) yield spread, theta = 0.0033%')
# plt.show()

# rates_arr = VasicekMultiSim(M, N, -0.01, kappa, theta, sigma)
# plt.figure(figsize=(10,5))
# plt.plot(t,rates_arr)
# plt.hlines(y=theta, xmin = -100, xmax=100, zorder=10, linestyles = 'dashed', label='Theta')
# plt.annotate('Theta', xy=(1.0, theta+0.0005))
# plt.xlim(-0.05, 1.05)
# plt.ylabel("Rate")
# plt.xlabel("Time (year)")
# plt.title('Last observed value, r0, is further away from theta')
# plt.show()

# rates_arr = VasicekMultiSim(M, N, -0.01, kappa*100, theta, sigma)
# plt.figure(figsize=(10,5))
# plt.plot(t,rates_arr)
# plt.hlines(y=theta, xmin = -100, xmax=100, zorder=10, linestyles = 'dashed', label='Theta')
# plt.annotate('Theta', xy=(1.0, theta+0.0005))
# plt.xlim(-0.05, 1.05)
# plt.ylabel("Rate")
# plt.xlabel("Time (year)")
# plt.title("Kappa (mean reversion speed) scaled up 10 times")
# plt.show()

rates_arr = VM.VasicekMultiSim(M, N, 0.0355, kappa*10, theta, sigma)
print(f'expected rate {N} days from now: {rates_arr[1]}')

plt.figure(figsize=(10,5))
plt.plot(t,rates_arr[0])
plt.hlines(y=theta, xmin = -100, xmax=100, zorder=10, linestyles = 'dashed', label='Theta')
plt.annotate('Theta', xy=(1.0, theta+0.0005))
plt.xlim(-0.05, 1.05)
plt.ylabel("Rate")
plt.xlabel("Time (year)")
plt.title("Sigma scaled up 10 times")
plt.show()