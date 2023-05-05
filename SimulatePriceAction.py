import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from VasicekModelf1 import VasicekModel 
from yr10TBondCorrelation import SecurityData

#Set up the model that finds the correlation between the 10 year treasury bond ans the specified stock

start = '2020-01-01'
end = '2022-01-01'

interestData = pd.read_csv('DGS10.csv')

interestData = interestData[start <= interestData['DATE']]
interestData = interestData[end >= interestData['DATE']]

securitiesData = SecurityData(['WFC'], interestData, start, end)
wfcCorrAndLastPrice = securitiesData.findCorr()

R_squared = wfcCorrAndLastPrice[0]
wfc_last_price = wfcCorrAndLastPrice[1]
wfc_10yr_Tbond_cov = wfcCorrAndLastPrice[2]

print(f'Covariance dict :{wfc_10yr_Tbond_cov}') 

#Set up the Vasicek model to simulate the interest rate for a given time interval.

rates = [rate for rate in pd.read_csv('DGS10.csv')['DGS10'] if rate != '.']

VM = VasicekModel(rates, 252)

params = VM.VasicekCalibration([float(r)/100 for r in rates])
kappa = params[0]
theta = params[1]
sigma = params[2]
r0 = params[3]
years = 1
N = years * 252
t = np.arange(0,N)/252


test_sim = VM.VasicekSim(N, r0, kappa, theta, sigma)

print('HERE!!!')
plt.figure(figsize=(10,5))
plt.title("One Simulated 10 yr Treasury for 255 Days")
plt.xlabel('Days')
plt.ylabel('Interest rate')
plt.plot(t,test_sim, color='r')
plt.show()

M = 1000

rates_arr = VM.VasicekMultiSim(M, N, 0.0355, kappa, theta, sigma)

print(f'expected rate {N} days from now: {rates_arr[1]}')

SimulatedRates = rates_arr[0]

# print(f'SimulatedRates: {[rateArr[-1] for rateArr in SimulatedRates]}')


simulatedRateReturns = VM.computeSimulatedRateChange(SimulatedRates)

plt.figure(figsize=(10,5))
plt.plot(t,SimulatedRates)
plt.hlines(y=theta, xmin = -100, xmax=100, zorder=10, linestyles = 'dashed', label='Theta')
plt.annotate('Theta', xy=(1.0, theta+0.0005))
plt.xlim(-0.05, 1.05)
plt.ylabel("Rate")
plt.xlabel("Time (year)")
plt.title("1000 simulations using Vasicek 1 factor model (market risk)")
plt.show()

#Perform a Monte Carlo simulation based on the influence of the interest rate 

class SimSharePrice:
    def __init__(self, lastPrice, cov, corrCoeff, simulatedRates, simulatedRateReturns):
        self.lastPrice = float(lastPrice)
        self.cov = float(cov)
        self.corrCoeff = float(corrCoeff)
        self.simulatedRates = simulatedRates
        self.simulatedRateReturns = simulatedRateReturns
    

    def simulatePriceAction(self):
        print(f'Last price: {self.lastPrice}, covariance: {self.cov}, r^2: {self.corrCoeff}')
        rateSimulation = []
        expectedPrice = []

        for returnRound in self.simulatedRateReturns:
            
            #Simulate each day's share price, by * prior price by the r^2 coefficient and multiplying it by (1 + change_in_interest_rate) * covariance + random distribution * (1 - r^2 coefficient)
            simulatedStockPrice = np.ones(len(returnRound))

            for i, rateReturn in enumerate(returnRound): 
                if i == 0:
                    simulatedStockPrice[i] = self.lastPrice + (self.lastPrice * self.corrCoeff) * (rateReturn * self.cov)
                else:
                    simulatedStockPrice[i] = simulatedStockPrice[i-1] + (simulatedStockPrice[i-1] * self.corrCoeff) * (rateReturn * self.cov) 
                    
            
            rateSimulation.append(simulatedStockPrice)
            expectedPrice.append(simulatedStockPrice[-1])

        
        expectedPrice = np.average(expectedPrice)
        print(f'Initial Price: {self.lastPrice}, Expected Price: {expectedPrice}')

        plt.plot(rateSimulation)
        plt.title('Simulated 10 Year Treasury Influence on WFC Share Price')
        plt.xlabel('Simulated days from last true WFC price')
        plt.ylabel('WFC Share Price')
        plt.hlines(y=expectedPrice, xmin = -100, xmax=100, zorder=10, linestyles = 'dashed', label='Expected Price')
        plt.show()

        return rateSimulation

WFC_Simulated = SimSharePrice(wfc_last_price, wfc_10yr_Tbond_cov['rates cov'][0], R_squared, SimulatedRates, simulatedRateReturns)
WFC_Simulated.simulatePriceAction()

print(wfc_10yr_Tbond_cov['rates cov'])
