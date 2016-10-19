'''
DYNAMIC HEDGE RATIO BETWEEN ETF PAIRS USING THE KALMAN FILTER

URL: https://www.quantstart.com/articles/Dynamic-Hedge-Ratio-Between-ETF-Pairs-Using-the-Kalman-Filter

References

[1] Chan, E.P. (2013). Algorithmic Trading: Winning Strategies and Their Rationale.
[2] O'Mahony, A. (2014). Online Linear Regression using a Kalman Filter
[3] Kinlay, J. (2015). Statistical Arbitrage Using the Kalman Filter
[4] Cowpertwait, P.S.P. and Metcalfe, A.V. (2009). Introductory Time Series with R.
[5] Pole, A., West, M., and Harrison, J. (1994). Applied Bayesian Forecasting.

'''

# TLT and ETF
# ===========

# We are going to consider two fixed income ETFs, namely the iShares 20+ Year Treasury Bond ETF (TLT) and the iShares 3-7 Year Treasury Bond ETF (IEI). 
# Both of these ETFs track the performance of varying duration US Treasury bonds and as such are both exposed to similar market factors. We will analyse their regression behaviour 
# over the last five years or so.

# Scatterplot of ETF Prices
# =========================

# We are now going to use a variety of Python libraries, including numpy, matplotlib, pandas and pykalman to to analyse the behaviour of a dynamic linear regression 
# between these two securities. As with all Python programs the first task is to import the necessary libraries:

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
from pykalman import KalmanFilter                # Note: You will likely need to run pip install pykalman to install the PyKalman library.

# The next step is write the function draw_date_coloured_scatterplot to produce a scatterplot of the asset adjusted closing prices 
# (such a scatterplot is inspired by that produced by Aidan O'Mahony). The scatterplot will be coloured using a matplotlib colour map, 
# specifically "Yellow To Red", where yellow represents price pairs closer to 2010, while red represents price pairs closer to 2016:

def draw_date_coloured_scatterplot(etfs, prices):
    """
    Create a scatterplot of the two ETF prices, which is
    coloured by the date of the price to indicate the 
    changing relationship between the sets of prices    
    """
    # Create a yellow-to-red colourmap where yellow indicates
    # early dates and red indicates later dates
    plen = len(prices)
    colour_map = plt.cm.get_cmap('YlOrRd')
    colours = np.linspace(0.1, 1, plen)
    
    # Create the scatterplot object
    scatterplot = plt.scatter(
        prices[etfs[0]], prices[etfs[1]], s=30, c=colours, cmap=colour_map, edgecolor='k', alpha=0.8)
    
    # Add a colour bar for the date colouring and set the 
    # corresponding axis tick labels to equal string-formatted dates
    colourbar = plt.colorbar(scatterplot)
    colourbar.ax.set_yticklabels(
        [str(p.date()) for p in prices[::plen//9].index]    
    )
    plt.xlabel(prices.columns[0])
    plt.ylabel(prices.columns[1])
    plt.show()
    
# Time-Varying Slope and Intercept
# ================================
    
# The next step is to actually use pykalman to dynamically adjust the intercept and slope between TFT and IEI. This function is more complex and requires some explanation.
# Firstly we define a variable called delta, which is used to control the transition covariance for the system noise. In my original article on the Kalman Filter this was denoted by WtWt. 
# We simply multiply such a value by the two-dimensional identity matrix. The next step is to create the observation matrix. As we previously described this matrix is a row vector 
# consisting of the prices of TFT and a sequence of unity values. To construct this we utilise the numpy vstack method to vertically stack these two price series into a single column vector, 
# which we then transpose. 
    
# At this point we use the KalmanFilter class from pykalman to create the Kalman Filter instance. We supply it with the dimensionality of the observations (unity in this case), 
# the dimensionality of the states (two in this case as we are looking at the intercept and slope in the linear regression).

# We also need to supply the mean and covariance of the initial state. In this instance we set the initial state mean to be zero for both intercept and slope, 
# while we take the two-dimensional identity matrix for the initial state covariance. The transition matrices are also given by the two-dimensional identity matrix.

# The last terms to specify are the observation matrices as above in obs_mat, with its covariance equal to unity. Finally the transition covariance matrix (controlled by delta) 
# is given by trans_cov, described above.

# Now that we have the kf Kalman Filter instance we can use it to filter based on the adjusted prices from IEI. This provides us with the state means of the intercept and slope, 
# which is what we're after. In addition we also receive the covariances of the states.

# This is all wrapped up in the calc_slope_intercept_kalman function:

def calc_slope_intercept_using_kalman_filter(etfs, prices):
    """
    Utilise the Kalman Filter from the pyKalman package
    to calculate the slope and intercept of the regressed
    ETF prices.
    """
    delta = 1e-5
    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_mat = np.vstack(
        [prices[etfs[0]], np.ones(prices[etfs[0]].shape)]
    ).T[:, np.newaxis]
    
    kf = KalmanFilter(
        n_dim_obs=1, 
        n_dim_state=2,
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=1.0,
        transition_covariance=trans_cov
    )
    
    state_means, state_covs = kf.filter(prices[etfs[1]].values)
    return state_means, state_covs
    
# Finally we plot these values as returned from the previous function. To achieve this we simply create a pandas DataFrame of the slopes and intercepts at time values tt, 
# using the index from the prices DataFrame, and plot each column as a subplot:
def draw_slope_intercept_changes(prices, state_means):
    """
    Plot the slope and intercept changes from the Kalman Filter calculated values.
    """
    pd.DataFrame(
        dict(
            slope=state_means[:, 0], 
            intercept=state_means[:, 1]
        ), index=prices.index
    ).plot(subplots=True)
    plt.show()
    
    
if __name__ == "__main__":
    # Choose the ETF symbols to work with along with 
    # start and end dates for the price histories
    etfs = ['TLT', 'IEI']
    start_date = "2010-8-01"
    end_date = "2016-08-01"    
    
    # Obtain the adjusted closin prices from Yahoo finance
    prices = pdr.DataReader(
        etfs, 'yahoo', start_date, end_date
    )['Adj Close']

    draw_date_coloured_scatterplot(etfs, prices)
    state_means, state_covs = calc_slope_intercept_using_kalman_filter(etfs, prices)
    draw_slope_intercept_changes(prices, state_means)    