# Python QSTrader Implementation
# ==============================

# Since QSTrader handles the position tracking, portfolio management, data ingestion and order management the only code we need to write involves the Strategy object itself.

# The Strategy communicates with the PortfolioHandler via the event queue, making use of SignalEvent objects to do so. In addition we must import the base abstract strategy class, 
# AbstractStrategy.

# Note that in the current alpha version of QSTrader we must also import the PriceParser class. This is used to multiply all prices on input by a large multiple (108108) and 
# perform integer arithmetic when tracking positions. This avoids floating point rounding issues that can accumulate over the long period of a backtest. 
# We must divide all the prices by PriceParser.PRICE_MULTIPLIER to obtain the correct values:

from math import floor

import numpy as np

from qstrader.price_parser import PriceParser
from qstrader.event import (SignalEvent, EventType)
from qstrader.strategy.base import AbstractStrategy

# The next step is to create the KalmanPairsTradingStrategy class. The job of this class is to determine when to create SignalEvent objects based on 
# received BarEvents from the daily OHLCV bars of TLT and IEI from Yahoo Finance.

# There are many different ways to organise this class. I've opted to hardcode all of the parameters in the class for clarity of the explanation. 
# Notably I've fixed the value of δ=10^−4 and vt=10^−3. They represent the system noise and measurement noise variance in the Kalman Filter model. 
# This could also be implemented as a keyword argument in the __init__ constructor of the class. Such an approach would allow straightforward parameter optimisation.

# The first task is to set the time and invested members to be equal to None, as they will be updated as market data is accepted and trade signals generated. 
# latest_prices is a two-array of the current prices of TLT and IEI, used for convenience through the class.

# The next set of parameters all relate to the Kalman Filter and are explained in depth in the previous two articles here (https://www.quantstart.com/articles/State-Space-Models-and-the-Kalman-Filter) 
# and here (https://www.quantstart.com/articles/Dynamic-Hedge-Ratio-Between-ETF-Pairs-Using-the-Kalman-Filter).

# The final set of parameters include days, used to track how many days have passed as well as qty and cur_hedge_qty, used to track the absolute quantities of ETFs to purchase for 
# both the long and short side. I have set this to be 2,000 units on an account equity of 100,000 USD.

class KalmanPairsTradingStrategy(AbstractStrategy):
    """
    Requires:
    tickers - The list of ticker symbols
    events_queue - A handle to the system events queue
    short_window - Lookback period for short moving average
    long_window - Lookback period for long moving average
    """
    def __init__(self, tickers, events_queue):
        self.tickers = tickers
        self.events_queue = events_queue
        self.time = None
        self.latest_prices = np.array([-1.0, -1.0])
        self.invested = None

        self.delta = 1e-4
        self.wt = self.delta / (1 - self.delta) * np.eye(2)
        self.vt = 1e-3
        self.theta = np.zeros(2)
        self.P = np.zeros((2, 2))
        self.R = None

        self.days = 0
        self.qty = 2000
        self.cur_hedge_qty = self.qty
        
    # The next method _set_correct_time_and_price is a "helper" method utilised to ensure that the Kalman Filter has all of the correct pricing information available at the right point. 
    # This is necessary because in an event-driven backtest system such as QSTrader market information arrives sequentially.

    # We might be in a situation on day K where we've received a price for IEI, but not TFT. Hence we must wait until both TFT and IEI market events have arrived from the backtest loop, 
    # through the events queue. In live trading this is not an issue since they will arrive almost instantaneously compared to the trading period of a few days. However, 
    # in an event-driven backtest we must wait for both prices to arrive before calculating the new Kalman filter update.

    # The code essentially checks if the subsequent event is for the current day. If it is, then the correct price is added to the latest_price list of TLT and IEI. 
    # If it is a new day then the latest prices are reset and the correct prices are once again added.

    # This type of "housekeeping" method will likely be absorbed into the QSTrader codebase in the future, reducing the necessity to write "boilerplate" code, 
    # but for now it must form part of the strategy itself.
    def _set_correct_time_and_price(self, event):
        """
        Sets the correct price and event time for prices
        that arrive out of order in the events queue.
        """
        # Set the first instance of time
        if self.time is None:
            self.time = event.time
            
        # Set the correct latest prices depending upon
        # order of arrival of market bar event
        price = event.adj_close_price/PriceParser.PRICE_MULTIPLIER
        if event.time == self.time:
            if event.ticker == self.tickers[0]:
                self.latest_prices[0] = price
            else:
                self.latest_prices[1] = price
        else:
            self.time = event.time
            self.days += 1
            self.latest_prices = np.array([-1.0, -1.0])
            if event.ticker == self.tickers[0]:
                self.latest_prices[0] = price
            else:
                self.latest_prices[1] = price  
                
    # The core of the strategy is carried out in the calculate_signals method. Firstly we set the correct times and prices (as described above). 
    # Then we check that we have both prices for TLT and IEI, at which point we can consider new trading signals. 
    # y is set equal to the latest price for IEI, while FF is the observation matrix containing the latest price for TLT, as well as a unity placeholder to represent the intercept 
    # in the linear regression. The Kalman Filter is subsequently updated with these latest prices. Finally we calculate the forecast error etet and the standard deviation of the predictions, 
    # Qt−−√Qt. Let's run through this code step-by-step, as it looks a little complicated.

    # The first task is to form the scalar value y and the observation matrix F, containing the prices of IEI and and TLT respectively. 
    # We calculate the variance-covariance matrix R or set it to the zero-matrix if it has not yet been initialised. Subsequently we calculate the new prediction of the observation 
    # yhat as well as the forecast error et.

    # We then calculate the variance of the observation predictions Qt as well as the standard deviation sqrt_Qt. We use the update rules derived here to obtain the posterior 
    # distribution of the states theta, which contains the hedge ratio/slope between the two prices:
                
    def calculate_signals(self, event):
        """
        Calculate the Kalman Filter strategy.
        """
        if event.type == EventType.BAR:
            self._set_correct_time_and_price(event)
            
        # Only trade if we have both observations
        if all(self.latest_prices > -1.0):
            # Create the observation matrix of the latest prices
            # of TLT and the intercept value (1.0) as well as the
            # scalar value of the latest price from IEI
            F = np.asarray([self.latest_prices[0], 1.0]).reshape((1, 2))
            y = self.latest_prices[1]
            
            # The prior value of the states \theta_t is
            # distributed as a multivariate Gaussian with
            # mean a_t and variance-covariance R_t
            if self.R is not None:
                self.R = self.C + self.wt
            else:
                self.R = np.zeros((2, 2))

            # Calculate the Kalman Filter update
            # ----------------------------------
            # Calculate prediction of new observation
            # as well as forecast error of that prediction
            yhat = F.dot(self.theta)
            et = y - yhat

            # Q_t is the variance of the prediction of
            # observations and hence \sqrt{Q_t} is the
            # standard deviation of the predictions
            Qt = F.dot(self.R).dot(F.T) + self.vt
            sqrt_Qt = np.sqrt(Qt)

            # The posterior value of the states \theta_t is
            # distributed as a multivariate Gaussian with mean
            # m_t and variance-covariance C_t
            At = self.R.dot(F.T) / Qt
            self.theta = self.theta + At.flatten() * et
            self.C = self.R - At * F.dot(self.R)
            
            # Finally we generate the trading signals based on the values of etet and Qt−−√Qt. To do this we need to check what the "invested" status is - either "long", "short" or "None". 
            # Notice how we need to adjust the cur_hedge_qty current hedge quantity when we go long or short as the slope θt0 is constantly adjusting in time:
            
            # Only trade if days is greater than a "burn in" period
            if self.days > 1:
                # If we're not in the market...
                if self.invested is None:
                    if et < -sqrt_Qt:
                        # Long Entry
                        print("LONG: %s" % event.time)
                        self.cur_hedge_qty = int(floor(self.qty*self.theta[0]))
                        self.events_queue.put(SignalEvent(self.tickers[1], "BOT", self.qty))
                        self.events_queue.put(SignalEvent(self.tickers[0], "SLD", self.cur_hedge_qty))
                        self.invested = "long"
                    elif et > sqrt_Qt:
                        # Short Entry
                        print("SHORT: %s" % event.time)
                        self.cur_hedge_qty = int(floor(self.qty*self.theta[0]))
                        self.events_queue.put(SignalEvent(self.tickers[1], "SLD", self.qty))
                        self.events_queue.put(SignalEvent(self.tickers[0], "BOT", self.cur_hedge_qty))
                        self.invested = "short"
                # If we are in the market...
                if self.invested is not None:
                    if self.invested == "long" and et > -sqrt_Qt:
                        print("CLOSING LONG: %s" % event.time)
                        self.events_queue.put(SignalEvent(self.tickers[1], "SLD", self.qty))
                        self.events_queue.put(SignalEvent(self.tickers[0], "BOT", self.cur_hedge_qty))
                        self.invested = None
                    elif self.invested == "short" and et < sqrt_Qt:
                        print("CLOSING SHORT: %s" % event.time)
                        self.events_queue.put(SignalEvent(self.tickers[1], "BOT", self.qty))
                        self.events_queue.put(SignalEvent(self.tickers[0], "SLD", self.cur_hedge_qty))
                        self.invested = None
                        
# This is all of the code necessary for the Strategy object. We also need to create a backtest file to encapsulate all of our trading logic and class choices. 
# The particular version is very similar to those used in the examples directory and replaces the equity of 500,000 USD with 100,000 USD.
            
# It also changes the FixedPositionSizer to the NaivePositionSizer. The latter is used to "naively" accept the suggestions of absolute quantities of ETF units 
# to trade as determined in the KalmanPairsTradingStrategy class. In a production environment it would be necessary to adjust this depending upon the risk management 
# goals of the portfolio.