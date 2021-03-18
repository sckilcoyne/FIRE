# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 21:35:32 2021

@author: Scott
"""

# %% Set up
import warnings

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from ForcePickle import pickle_protocol
import pandas as pd

# %% Parameters
goodFitThresh = 0.95  # How good should a distribution be to used in sims
simYears = 100  # Number of years to generate random data for
N = 1000  # Number of randomized timeseries per distribution

# %% Import Data
# sp500 = pd.read_csv('Data/SlickCharts SP500 History.csv',
#                     header=0, names=['Year', 'SP500 Returns'])
# # print(sp500.head(5))
# # print(sp500.tail(5))

# cpi = pd.read_csv('Data/Yearly CPI Change.csv')
# # print(cpi.tail(5))
# # print(cpi.head(5))

# returns = pd.merge(sp500, cpi, on='Year')
# returns['Net Returns'] = returns['SP500 Returns'] - \
#     returns['Annual CPI Change']
# print(returns)

shiller = pd.read_excel('Data/Shiller Yearly Market Data.xlsx',
                        sheet_name='Data', header=7, index_col=0)

marketReturns = pd.DataFrame()
marketReturns['Net Returns'] = shiller['Unnamed: 16']
marketReturns.dropna(inplace=True)
marketReturns['Net Returns 100'] = marketReturns['Net Returns'].multiply(100)

# %% Functions


def yearly2cumalitive(yearlyCol):
    """
    Convert yearly performance to cumalative timeseries performance.

    yearlyCol: Timeseries list of fractional percentages
    """
    yearlyCol = yearlyCol.to_numpy()

    cumalitiveCol = np.empty([0, 1])
    for idx, x in enumerate(yearlyCol):
        if idx == 0:
            cumalitiveCol = np.append(cumalitiveCol, yearlyCol[0])
        else:
            cumalitiveResult = cumalitiveCol[idx-1] + \
                yearlyCol[idx] * cumalitiveCol[idx-1]
            cumalitiveCol = np.append(cumalitiveCol, cumalitiveResult)

    return cumalitiveCol


def cumalitive2yearly(cumalitiveCol):
    """
    Convert cumalative timeseries performance to yearly performance.
    """
    cumalitiveCol = cumalitiveCol.to_numpy()

    yearlyCol = np.empty([0, 1])
    for idx, x in enumerate(cumalitiveCol):
        if idx == 0:
            yearlyCol = np.append(yearlyCol, cumalitiveCol[0])
        else:
            yearlyResult = cumalitiveCol[idx] / cumalitiveCol[idx-1] - 1
            yearlyCol = np.append(yearlyCol, yearlyResult)

    return yearlyCol


def set_shared_xlabel(a, xlabel, labelpad=0.01):
    """
    Set a x label shared by multiple axes.

    https://stackoverflow.com/questions/6963035/pyplot-axes-labels-for-subplots

    Parameters
    ----------
    a: list of axes
    xlabel: string
    labelpad: float
        Sets the padding between ticklabels and axis label
    """
    f = a[0].get_figure()
    f.canvas.draw()  # sets f.canvas.renderer needed below

    # get the center position for all plots
    right = a[0].get_position().x1
    left = a[-1].get_position().x0

    # get the coordinates of the left side of the tick labels
    y0 = 1
    for at in a:
        at.set_xlabel('')  # prevent multiple labels
        bboxes, _ = at.xaxis.get_ticklabel_extents(f.canvas.renderer)
        bboxes = bboxes.transformed(f.transFigure.inverted())
        yt = bboxes.y0
        if yt < y0:
            y0 = yt
    tick_label_up = y0

    # set position of label
    a[-1].set_xlabel(xlabel)
    a[-1].xaxis.set_label_coords((right + left)/2,
                                 tick_label_up - labelpad,
                                 transform=f.transFigure)


# %% Distribution Determinination
# https://medium.com/@amirarsalan.rajabi/distribution-fitting-with-python-scipy-bb70a42c0aed
# https://nedyoxall.github.io/fitting_all_of_scipys_distributions.html
list_of_dists = ['alpha', 'anglit', 'arcsine', 'argus', 'beta', 'betaprime',
                 'bradford', 'burr', 'burr12', 'cauchy', 'chi', 'chi2',
                 'cosine', 'crystalball', 'dgamma', 'dweibull', 'erlang',
                 'expon', 'exponnorm', 'exponweib', 'exponpow', 'f',
                 'fatiguelife', 'fisk', 'foldcauchy', 'foldnorm',
                 'genlogistic',  'genpareto', 'gennorm', 'genexpon',
                 'genextreme', 'gausshyper', 'gamma', 'gengamma',
                 'genhalflogistic', 'geninvgauss', 'gilbrat', 'gompertz',
                 'gumbel_r', 'gumbel_l', 'halfcauchy', 'halflogistic',
                 'halfnorm', 'halfgennorm', 'hypsecant', 'invgamma',
                 'invgauss', 'invweibull', 'johnsonsb', 'johnsonsu', 'kappa4',
                 'kappa3', 'ksone',  # 'kstwo',
                 'kstwobign', 'laplace',
                 'laplace_asymmetric', 'levy', 'levy_l',  # 'levy_stable',
                 'logistic', 'loggamma', 'loglaplace', 'lognorm', 'lomax',
                 'maxwell', 'mielke', 'moyal', 'nakagami', 'ncx2', 'ncf',
                 'nct', 'norm', 'norminvgauss', 'pareto', 'pearson3',
                 'powerlaw', 'powerlognorm', 'powernorm', 'rdist',
                 'reciprocal', 'rayleigh', 'rice', 'recipinvgauss',
                 'semicircular', 'skewnorm', 't', 'triang', 'truncexpon',
                 'truncnorm', 'tukeylambda', 'uniform', 'vonmises',
                 'vonmises_line', 'wald', 'weibull_min', 'weibull_max',
                 'wrapcauchy']

# results = []
resultsDf = pd.DataFrame(columns=['Distribution', 'Statistic', 'p-value'])
with warnings.catch_warnings():
    warnings.filterwarnings(
        'ignore', category=RuntimeWarning)
    for i in list_of_dists:
        print(i)
        dist = getattr(stats, i)
        param = dist.fit(marketReturns['Net Returns'])
        a = stats.kstest(marketReturns['Net Returns'], i, args=param)
        newRow = pd.Series([i, a[0], a[1]], index=resultsDf.columns)
        resultsDf = resultsDf.append(newRow, ignore_index=True)

resultsDf.sort_values('p-value', ascending=False,
                      inplace=True, na_position='last')


goodFits = resultsDf[resultsDf['p-value'] > goodFitThresh]
print(goodFits)


# %% Create Random Data with Good Distribution(s)

# Generate [N] time series of length [simYears] of randomly generated yearly
# market performance for each distribution determined to be a good fit to the
# historical data
simulatedReturns = pd.DataFrame()
for distName in goodFits['Distribution']:
    # print(distName)
    distFunc = getattr(stats, distName)
    distParams = distFunc.fit(marketReturns['Net Returns'])

    for i in range(N):
        iterName = distName + str(i)
        simulatedReturns[iterName] = [1] + \
            distFunc.rvs(*distParams[:-2],
                         loc=distParams[-2], scale=distParams[-1],
                         size=simYears).tolist()

# Convert yearly returns to net performance for each time series
simulatedPerformance = simulatedReturns.apply(yearly2cumalitive, axis=0)

# Convert from fractional to percentage
simulatedReturns = simulatedReturns.multiply(100)
simulatedPerformance = simulatedPerformance.multiply(100)

# Find each whole percentile of performance along the time series
simulatedPerformanceStats = pd.DataFrame()
for i in np.linspace(0.01, 0.99, 99):
    percentile = str(int(i * 100)) + ' Percentile'
    simulatedPerformanceStats[percentile] = simulatedPerformance.quantile(
        i, axis=1)

simPercentileYearly = simulatedPerformanceStats.divide(100)
simPercentileYearly = simPercentileYearly.apply(
    cumalitive2yearly, axis=0)
simPercentileYearly = simPercentileYearly.multiply(100)


# %% Analysis Plot
fig, axs = plt.subplots(figsize=(15, 10))
fig.suptitle('Market Returns \n Historical and Simulated')

# -----Historical Returns-----
# Returns Distribution Histogram
ax1 = plt.subplot(2, 2, (1, 2))
ax1.hist(marketReturns['Net Returns 100'], density=True,
         bins=20, label='Net Returns', alpha=0.5)

# Median Return
medianReturn = marketReturns['Net Returns 100'].median()
ax1.axvline(medianReturn, label='Meadin Return',
            linestyle='--', linewidth=2, color='k')
ax1.annotate(
    str(round(medianReturn, 1)) + '%',  # annotation text
    (medianReturn, 0),  # this is the point to label
    textcoords="offset points",  # how to position the text
    xytext=(3, 2),  # distance from text to points (x,y)
    color='k', size='large', weight='demi',
    ha='left')  # horizontal alignment can be left, right or center

# Best Continuous distributions
for distName in goodFits['Distribution']:
    # print(distName)
    distFunc = getattr(stats, distName)
    distParams = distFunc.fit(marketReturns['Net Returns'])
    startDistLine = distFunc.ppf(0.01, *distParams[:-2],
                                 loc=distParams[-2] * 100,
                                 scale=distParams[-1] * 100)
    endDistLine = distFunc.ppf(0.99, *distParams[:-2],
                               loc=distParams[-2] * 100,
                               scale=distParams[-1] * 100)
    x = np.linspace(startDistLine, endDistLine, 100)

    rv = distFunc(*distParams[:-2],
                  loc=distParams[-2] * 100,
                  scale=distParams[-1] * 100)

    distLabel = distName + ' pdf: ' + \
        str(round(resultsDf['p-value'].iloc[0], 4))
    ax1.plot(x, rv.pdf(x), '-', lw=2, label=distLabel)

# Return distribution plot formatting
ax1.legend()
ax1.xaxis.set_major_formatter('{x:1.0f}%')
yearFirst = marketReturns.head(1).index.item()
yearLast = marketReturns.tail(1).index.item()
ax1.set_xlabel('Historical Market Returns (' +
               str(int(yearFirst)) + '-' + str(int(yearLast)) + ')')

# -----Compounding Growth Over Time-----
yearSwitchOver = 30
plottedPercentile = [5, 20, 50, 80, 95]

# Match naming scheme
plottedPercentile = [
    str(int(per)) + ' Percentile' for per in plottedPercentile]

# Only plot desired percentiles
plotSimmedStats = simulatedPerformanceStats[plottedPercentile]

# Prep heatmap
data = simulatedPerformance.stack().droplevel(level=1)

# Linear plot limits
xLinMin = 0
xLinMax = yearSwitchOver
yLinMin = 0
yLinMax = simulatedPerformanceStats['50 Percentile'][yearSwitchOver] * 1.5

# Set up desity grid - linear
xbinsLin = xLinMax - xLinMin + 1
ybinsLin = int((yLinMax - yLinMin) / 10)
xGridLin = np.linspace(xLinMin, xLinMax, xbinsLin)
yGridLin = np.linspace(yLinMin, yLinMax, ybinsLin)
xLin, yLin = np.meshgrid(xGridLin, yGridLin)

# Get density map - linear
y = data.iloc[(data.index <= xLinMax) &
              (data.values <= yLinMax)]
x = y.index.values
k = stats.gaussian_kde(np.vstack([x, y]))
zLin = k(np.vstack([xLin.flatten(), yLin.flatten()]))

# Log Plot limits
xLogMin = yearSwitchOver
xLogMax = simYears
yLogMin = 50
yLogMax = simulatedPerformanceStats['95 Percentile'].max() * 1.2

# # Set up desity grid - log
# xbinsLog = xLogMax - xLogMin + 1
# # ybinsLog = int((yLogMax - yLogMin) / 100)
# ybinsLog = 500
# xGridLog = np.linspace(xLogMin, xLogMax, xbinsLog)
# # yGridLog = np.linspace(yLogMin, yLogMax, ybinsLog)
# yGridLog = np.geomspace(yLogMin, yLogMax, ybinsLog)
# xLog, yLog = np.meshgrid(xGridLog, yGridLog)

# # Get density map - log
# y = data.iloc[(data.index >= xLogMin) &
#               (data.values >= yLogMin) &
#               (data.values <= yLogMax)]
# x = y.index.values
# k = stats.gaussian_kde(np.vstack([x, y]))
# zLog = k(np.vstack([xLog.flatten(), yLog.flatten()]))


# Plot simulated data on two subplots for short term and long term display
ax2 = plt.subplot(2, 2, 3)  # Linear axes for early growth
ax3 = plt.subplot(2, 2, 4)  # Log axis for long-term growth

# Plot density map
ax2.pcolormesh(xLin, yLin, np.power(zLin.reshape(xLin.shape), 0.5),
               shading='gouraud', cmap='Greys')
# ax3.pcolormesh(xLog, yLog, np.power(zLog.reshape(xLog.shape), 0.3),
#                shading='gouraud', cmap='Greys')

for ax in [ax2, ax3]:
    # simulatedPerformance.plot(
    #     legend=False, color='gainsboro', alpha=0.01, ax=ax)
    plotSimmedStats.plot(ax=ax)

# Format Linear Plot
ax2.set_ylim([yLinMin, yLinMax])
ax2.set_xlim([xLinMin, xLinMax])
ax2.set_ylabel('Initial Value')
ax2.yaxis.set_major_formatter('{x:1.0f}%')
# ax2.get_legend().remove()  # Legend only on one plot

# Format Log Plot
ax3.set_yscale('log')
ax3.set_ylim([yLogMin, yLogMax])
ax3.set_xlim([xLogMin, xLogMax])
ax3.yaxis.set_major_formatter('{x:1.0f}%')
ax3.yaxis.tick_right()
ax3.set_xlabel('placeholder for tight layout')
ax3.get_legend().remove()  # Legend only on one plot


# Overall figure formatting
fig.tight_layout()
set_shared_xlabel([ax2, ax3], 'Years of Compounding Growth')


# %% Save Analysis for App Usage
fig.savefig('Outputs/Market Returns.png')

with pickle_protocol(2):

    marketReturns.to_hdf('Outputs/marketReturns.h5',
                         key='marketReturns', mode='w')
    resultsDf.to_hdf('Outputs/results.h5',
                     key='results', mode='w')
    simulatedPerformance.to_hdf('Outputs/simulatedPerformance.h5',
                                key='simulatedPerformance', mode='w')
    simulatedPerformanceStats.to_hdf('Outputs/simulatedPerformanceStats.h5',
                                     key='simulatedPerformanceStats', mode='w')
    simPercentileYearly.to_hdf('Outputs/simPercentileYearly.h5',
                               key='simPercentileYearly', mode='w')
    goodFits.to_hdf('Outputs/goodFits.h5',
                    key='goodFits', mode='w')


# %% Testing
