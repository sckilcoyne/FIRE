# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 21:35:32 2021

@author: Scott
"""

# %% Set up
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pylab
from scipy import stats
import numpy as np

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

returns = pd.DataFrame()
returns['Net Returns'] = shiller['Unnamed: 16']
returns.dropna(inplace=True)
returns['Net Returns 100'] = returns['Net Returns'].multiply(100)

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
        param = dist.fit(returns['Net Returns'])
        a = stats.kstest(returns['Net Returns'], i, args=param)
        # results.append((i, a[0], a[1]))
        newRow = pd.Series([i, a[0], a[1]], index=resultsDf.columns)
        resultsDf = resultsDf.append(newRow, ignore_index=True)


# results.sort(key=lambda x: float(x[2]), reverse=True)
# resultsTop = results[:5]
# for j in resultsTop:
#     print("{}: statistic={}, pvalue={}".format(j[0], j[1], j[2]))

resultsDf.sort_values('p-value', ascending=False,
                      inplace=True, na_position='last')
print(resultsDf.head(5))


# %% Create Random Data with Best Distribution
# bestDistName = results[0][0]
bestDistName = resultsDf['Distribution'].iloc[0]
bestDist = getattr(stats, bestDistName)
bestDistParams = bestDist.fit(returns['Net Returns'])

randomData = pd.DataFrame()
simYears = 100
N = 1000

for i in range(N):
    randomData[i] = [1] + bestDist.rvs(
        bestDistParams[0], bestDistParams[1], loc=bestDistParams[2],
        scale=bestDistParams[3], size=simYears).tolist()


def yearly2cumalitive(yearlyCol):
    # print(type(yearlyCol))
    yearlyCol = yearlyCol.to_numpy()
    # print(type(yearlyCol))
    # print(yearlyCol)

    cumalitiveCol = np.empty([0, 1])
    # print(cumalitiveCol)
    for idx, x in enumerate(yearlyCol):
        # print(idx, yearlyCol[idx])
        if idx == 0:
            cumalitiveCol = np.append(cumalitiveCol, yearlyCol[0])
            # print(cumalitiveCol)
            # print(cumalitiveCol[0])
        else:
            cumalitiveResult = cumalitiveCol[idx-1] + \
                yearlyCol[idx] * cumalitiveCol[idx-1]
            # print(cumalitiveResult)
            cumalitiveCol = np.append(cumalitiveCol, cumalitiveResult)

    return cumalitiveCol


randomGrowth = randomData.apply(yearly2cumalitive, axis=0)

# Convert from fractional to percentage
randomData = randomData.multiply(100)
randomGrowth = randomGrowth.multiply(100)

rowMedian = randomGrowth.median(axis=1)
rowPercentileHigh = randomGrowth.quantile(0.95, axis=1)
rowPercentileLow = randomGrowth.quantile(0.05, axis=1)

randomGrowthStats = pd.DataFrame()
randomGrowthStats['Median'] = rowMedian
randomGrowthStats['Percentile 95th'] = rowPercentileHigh
randomGrowthStats['Percentile 5th'] = rowPercentileLow

# %% Analysis Plot


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
        # just to make sure we don't and up with multiple labels
        at.set_xlabel('')
        bboxes, _ = at.xaxis.get_ticklabel_extents(f.canvas.renderer)
        # bboxes = bboxes.inverse_transformed(f.transFigure)
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


fig, axs = plt.subplots(figsize=(15, 8))
fig.suptitle('Market Returns \n Historical and Simulated')

# Returns Distribution
ax1 = plt.subplot(2, 2, (1, 2))
ax1.hist(returns['Net Returns 100'], density=True,
         bins=20, label='Net Returns', alpha=0.5)

medianReturn = returns['Net Returns 100'].median()
ax1.axvline(medianReturn, label='Meadin Return',
            linestyle='--', linewidth=2, color='r')
ax1.annotate(
    str(round(medianReturn, 1)) + '%',  # annotation text
    (medianReturn, 0),  # this is the point to label
    textcoords="offset points",  # how to position the text
    xytext=(3, 2),  # distance from text to points (x,y)
    color='r', size='large', weight='demi',
    ha='left')  # horizontal alignment can be left, right or center

x = np.linspace(bestDist.ppf(0.01, bestDistParams[0], bestDistParams[1],
                             loc=bestDistParams[2] * 100,
                             scale=bestDistParams[3] * 100),
                bestDist.ppf(0.99, bestDistParams[0], bestDistParams[1],
                             loc=bestDistParams[2] * 100,
                             scale=bestDistParams[3] * 100),
                100)

rv = bestDist(bestDistParams[0], bestDistParams[1],
              loc=bestDistParams[2] * 100, scale=bestDistParams[3] * 100)
distLabel = bestDistName + ' pdf: ' + \
    str(round(resultsDf['p-value'].iloc[0], 4))
ax1.plot(x, rv.pdf(x), 'k-', lw=2, label=distLabel)
ax1.legend()
ax1.xaxis.set_major_formatter('{x:1.0f}%')
yearFirst = returns.head(1).index.item()
yearLast = returns.tail(1).index.item()
ax1.set_xlabel('Historical Market Returns (' +
               str(int(yearFirst)) + '-' + str(int(yearLast)) + ')')

# Compounding Growth Over Time
yearSwitchOver = 30

ax2 = plt.subplot(2, 2, 3)  # Linear axes for early growth
randomGrowth.plot(legend=False, color='grey', alpha=0.01, ax=ax2)
randomGrowthStats.plot(ax=ax2, legend=False)
ax2.set_ylim([.1, randomGrowthStats['Median'][yearSwitchOver] * 1.5])
ax2.set_xlim([0, yearSwitchOver])
ax2.set_ylabel('Initial Value')
ax2.yaxis.set_major_formatter('{x:1.0f}%')

ax3 = plt.subplot(2, 2, 4)  # Log axis for long-term growth
randomGrowth.plot(legend=False, color='grey', alpha=0.01, ax=ax3)
randomGrowthStats.plot(ax=ax3)
ax3.set_yscale('log')
ax3.set_ylim([50, randomGrowthStats['Percentile 95th'].max() * 1.2])
ax3.set_xlim([yearSwitchOver, randomGrowthStats.tail(1).index.item()])
ax3.yaxis.set_major_formatter('{x:1.0f}%')
ax3.yaxis.tick_right()
ax3.set_xlabel('placeholder for tight layout')

fig.tight_layout()
set_shared_xlabel([ax2, ax3], 'Years of Compounding Growth')


# %% Save Analysis for App Usage
fig.savefig('Outputs/Market Returns.png')


storeResults = pd.HDFStore('Outputs/storeResults.h5')
storeRandomGrowth = pd.HDFStore('Outputs/storeRandomGrowth.h5')
storeRandomGrowthStats = pd.HDFStore('Outputs/storeRandomGrowthStats.h5')

storeResults['resultsDf'] = resultsDf
storeRandomGrowth['randomGrowth'] = randomGrowth
storeRandomGrowthStats['randomGrowthStats'] = randomGrowthStats

# store['df'] = df  # save it
# store['df']  # load it
