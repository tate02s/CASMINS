import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

start = '2020-01-01'
end = '2022-01-01'

interestData = pd.read_csv('DGS10.csv')
RATE = 'DGS10'

interestData = interestData[start <= interestData['DATE']]
interestData = interestData[end >= interestData['DATE']]
print(interestData)

class SecurityData:
    def __init__(self, tickers, interestRates, start, end, period='1d'):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.period = period
        self.data = [interestRates]

    def _pairData(self):
        """Make sure the dates for the interest rates and stock data match, by converting each into sets, then taking the intersection of the two."""
        data = self._getMarketData()

        stockDates = set([str(date).split(' ')[0] for date in data[1].index])
        interestRateDates = set([date for i, date in enumerate(data[0]['DATE']) if data[0][RATE][data[0].index[i]] != '.'])
        mutualDates = set.intersection(stockDates, interestRateDates)

        mutualCloses = [float(s) for i, s in enumerate(data[1]['Close']) if str(data[1].index[i]).split(' ')[0] in mutualDates]
        lastPrice = mutualCloses[-1]

        mutualRates = [float(rate) for i, rate in enumerate(data[0][RATE]) if data[0]['DATE'][data[0].index[i]] in mutualDates]

        pairs = [pair for pair in zip(mutualRates, mutualCloses)]

        #Find the covariance
        cov = self._findCOV(mutualRates, mutualCloses)

        print(f"The covariance is: {cov}")

        return (pairs, lastPrice, cov)
    
    def _findCOV(self, rates, stockPrices):
        rates = np.array(rates)
        stockPrices = np.array(stockPrices)
        cov = np.cov(rates, stockPrices)

        return {'rates cov': cov[0], 'stock price cov': cov[1]}

    def _getMarketData(self):
        for ticker in self.tickers:
            pa = yf.Ticker(ticker).history(start=self.start, end=self.end, period=self.period)
            self.data.append(pa)
        
        return self.data
    
    def findCorr(self):
        data = self._pairData()
        pairedData = data[0]
        lastPrice = data[1]
        cov = data[2]

        #Prep the data for the Linear Regression model
        X, y = np.array([p[0] for p in pairedData]).reshape(-1, 1), [p[1] for p in pairedData]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        #Train the Linear Regression model with the testing data
        model = LR()
        model.fit(X=X_train, y=y_train)
        y_pred = model.predict(X_test)

        xValues = [v[0] for v in pairedData]
        trendline = [float(model.coef_ * x + model.intercept_) for x in xValues]
        print(f'Trendline points: {trendline}')

        plt.scatter(x=[v[0] for v in pairedData], y=[v[1] for v in pairedData])
        plt.scatter(x=xValues, y=trendline)
        plt.title('Correlation Between 10 yr Treasury Yields and Wells Fargo Share Price')
        plt.xlabel('10 yr Treasury yield (2020-2022)')
        plt.ylabel('Wells Fargo Share Price')

        plt.show()

        return (r2_score(y_test, y_pred), lastPrice, cov)
