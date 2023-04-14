import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class VasicekModel:
  def __init__(self, rates, days=252):
    self.rates = rates
    self.dt = 1/days


  def VasicekNextRate(self, r, kappa, theta, sigma):
    # Implements above closed form solution
    val1 = np.exp(-kappa*self.dt)
    val2 = (sigma**2)*(1-val1**2) / (2*kappa)
    out = r*val1 + theta*(1-val1) + (np.sqrt(val2))*np.random.normal()
    return out
  # Vasicek simulation short rate

  def VasicekSim(self, N, r0, kappa, theta, sigma):
    short_r = [0]*N # Create array to store rates
    short_r[0] = r0 # Initialize rates at $r_0$
    for i in range(1,N):
      short_r[i]=self.VasicekNextRate(short_r[i-1],kappa, theta, sigma)
    return short_r

  # Vasicek multi-simulation
  def VasicekMultiSim(self, M, N, r0, kappa, theta, sigma):
    sim_arr = np.ndarray((N, M))
    for i in range(0,M):
      sim_arr[:, i] = self.VasicekSim(N, r0, kappa, theta, sigma)

    forcastedRate = np.mean([rate[-1] for rate in sim_arr])
    return [sim_arr, forcastedRate]

  # Maximum Likelihood Estimation to calibrate parameters
  def VasicekCalibration(self, rates):
    n = len(rates)
    Ax = sum(rates[0:(n-1)])
    Ay = sum(rates[1:n])
    Axx = np.dot(rates[0:(n-1)], rates[0:(n-1)])
    Axy = np.dot(rates[0:(n-1)], rates[1:n])
    Ayy = np.dot(rates[1:n], rates[1:n])
    theta = (Ay * Axx - Ax * Axy) / (n * (Axx - Axy) - (Ax**2 - Ax*Ay))
    kappa = -np.log((Axy - theta * Ax - theta * Ay + n * theta**2) / (Axx - 2*theta*Ax + n*theta**2)) / self.dt
    a = np.exp(-kappa * self.dt)

    print(f'mean reversion coefficient: {kappa}, equilibrium rate: {a}, theta: {theta}')

    sigmah2 = (Ayy - 2*a*Axy + a**2 * Axx - 2*theta*(1-a)*(Ay - a*Ax) + n*theta**2 * (1-a)**2) / n
    sigma = np.sqrt(sigmah2*2*kappa / (1-a**2))
    r0 = rates[n-1]
    return [kappa, theta, sigma, r0]
  
  def computeSimulatedRateChange(self, multiSimRates):
    multiRateReturn = []

    for rates in multiSimRates:
      ratesSer = pd.Series(rates)
      rateSimRound = pd.Series(np.log(ratesSer/ratesSer.shift(1))).dropna()
  
      multiRateReturn.append(rateSimRound)

    return multiRateReturn
  


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