# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 17:32:25 2021

@author: Scott
"""

# %% Set up
# import git
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
# import numpy as np
import math
import pandas as pd
import requests
import os

# Extras for requirements
import tables
# import pickle5 as pickle

# Set styles and themes
# matplotlib.rcParams.update(matplotlib.rcParamsDefault)
githubContent = 'https://raw.githubusercontent.com/sckilcoyne/FIRE/'
githubBranch = 'main'
styleFile = 'fig_style'
styleFile = githubContent + githubBranch + '/' + styleFile + '.mplstyle'
plt.style.use(styleFile)

# %% App Header
st.set_page_config(page_title='FIRE Calculator',
                   page_icon=':fire:',
                   initial_sidebar_state='expanded',
                   layout='wide')

st.title('FIRE Calculator')
'''
[@sckilcoyne](https://github.com/sckilcoyne/FIRE)
'''

# %%Import Data

# Github data sources
githubRepo = 'https://github.com/sckilcoyne/FIRE/'
githubBranch = 'main' + '/'
githubFolder = 'Outputs/'
githubURL = githubRepo + 'blob/' + githubBranch + githubFolder


@st.cache
def import_from_github(githubURL):

    # Save loaded files in temp directory
    # https://discuss.streamlit.io/t/file-permisson-error-on-streamlit-sharing/8291/5
    temp = '/tmp/'
    os.makedirs(temp, exist_ok=True)  # Make temp directory if needed

    raw = '?raw=true'
    distFitsFile = githubURL + 'results.h5' + raw
    simPerformFile = githubURL + 'simulatedPerformance.h5' + raw
    simPercentileFile = githubURL + 'simulatedPerformanceStats.h5' + raw
    simPercentileYearlyFile = githubURL + 'simPercentileYearly.h5' + raw
    marketReturnsFile = githubURL + 'marketReturns.h5' + raw

    print(distFitsFile + '\n' + simPerformFile +
          '\n' + simPercentileFile + '\n' + marketReturnsFile)

    # Import data
    r = requests.get(distFitsFile, allow_redirects=True)
    tempFile = os.path.join(temp, 'resultsDf_github.h5')
    print(tempFile)
    open(tempFile, 'wb').write(r.content)
    distFits = pd.read_hdf(tempFile, 'results')

    r = requests.get(simPerformFile, allow_redirects=True)
    tempFile = os.path.join(temp, 'simulatedPerformance_github.h5')
    open(tempFile, 'wb').write(r.content)
    simPerform = pd.read_hdf(tempFile, 'simulatedPerformance')

    r = requests.get(simPercentileFile, allow_redirects=True)
    tempFile = os.path.join(temp, 'simulatedPerformanceStats_github.h5')
    open(tempFile, 'wb').write(r.content)
    simPercentile = pd.read_hdf(tempFile, 'simulatedPerformanceStats')

    r = requests.get(simPercentileYearlyFile, allow_redirects=True)
    tempFile = os.path.join(temp, 'simPercentileYearly_github.h5')
    open(tempFile, 'wb').write(r.content)
    simPercentileYearly = pd.read_hdf(tempFile, 'simPercentileYearly')

    r = requests.get(marketReturnsFile, allow_redirects=True)
    tempFile = os.path.join(temp, 'marketReturns_github.h5')
    open(tempFile, 'wb').write(r.content)
    marketReturns = pd.read_hdf(tempFile, 'marketReturns')

    return distFits, simPerform, simPercentile, simPercentileYearly, marketReturns


distFits, simPerform, simPercentile, simPercentileYearly, marketReturns = import_from_github(
    githubURL)

# %% Functions


def growth(currentSavings, yearsAway, RoR, yearlySavings):
    """
    Calculate Future Value of investments with regular deposits.

    Parameters
    ----------
    currentSavings : TYPE
        DESCRIPTION.
    yearsAway : TYPE
        DESCRIPTION.
    RoR : TYPE
        DESCRIPTION.
    yearlySavings : TYPE
        DESCRIPTION.

    Returns
    -------
    FV : TYPE
        DESCRIPTION.
    totalGrowth : TYPE
        DESCRIPTION.

    """
    if type(RoR) == float:
        principalGrowth = [currentSavings * (1 + RoR) ** x for x in yearsAway]

        savingsGrowth = [yearlySavings *
                         ((1 + RoR) ** x - 1) / RoR for x in yearsAway]

        FV = [sum(i) for i in zip(principalGrowth, savingsGrowth)]

        totalGrowth = [FV[i] - (i+1)*yearlySavings -
                       currentSavings for i, x in enumerate(FV)]

    elif type(RoR) == pd.core.series.Series:
        # RoR[0] = 0
        principalGrowth = []
        savingsGrowth = []
        for x in yearsAway:
            if x == 0:
                principalGrowth.append(currentSavings)
                savingsGrowth.append(yearlySavings)
            else:
                principalGrowth.append(principalGrowth[x-1] * (1 + RoR[x]))
                savingsGrowth.append(
                    yearlySavings + savingsGrowth[x-1] * (1 + RoR[x]))

        FV = [sum(i) for i in zip(principalGrowth, savingsGrowth)]

        totalGrowth = [FV[i] - (i+1)*yearlySavings -
                       currentSavings for i, x in enumerate(FV)]

    else:
        errorMsg = 'Unknown Rate of return'
        print(errorMsg)
        st.write(errorMsg)

    return FV, totalGrowth


def milestone_calc(RoR, currentSavings, yearlySavings, milestoneGoal):
    if type(RoR) is float:
        years2goal = math.log((milestoneGoal + yearlySavings / RoR) /
                              (currentSavings + yearlySavings / RoR)) / \
            math.log(1 + RoR)
    else:
        print('Wrong milestone function: ' + str(type(RoR)))
        years2goal = 0

    return years2goal


def milestone_interp(yearlyValues, yearList, milestoneGoal):
    # Linear interpolation of milestone achievement
    if type(yearlyValues) is list:
        # Find values greater than goal
        valueAbove = list(
            filter(lambda k: k > milestoneGoal, yearlyValues))
        try:
            valueAbove = valueAbove[0]
            idxAbove = yearlyValues.index(valueAbove)

            valueBelow = yearlyValues[idxAbove - 1]
            yearBelow = yearList[idxAbove - 1]

            linInterp = (milestoneGoal - valueBelow) / \
                (valueAbove - valueBelow)

            years2goal = yearBelow + linInterp

        except IndexError:  # If don't reach goal, set to max year
            years2goal = yearList[-1]
            print('Does not reach goal!')

    else:
        print('Wrong milestone function: ' + str(type(yearlyValues)))
        years2goal = 0

    return years2goal


# %% Sidebar Inputs
age = st.sidebar.number_input(
    'Current Age', min_value=18, max_value=70, value=30, step=1)

currentSavings = st.sidebar.number_input(
    'Current Invested Savings ($)', value=200000, step=1000)

currentSalary = st.sidebar.number_input(
    'Salary ($)', min_value=0, value=90000, step=1000)

savingsRate = st.sidebar.number_input(
    'Savings Rate (%)', min_value=5., max_value=80., value=30.,
    step=0.5, format='%.1f') / 100

SWR = st.sidebar.number_input(
    'Withdrawal Rate (%)', min_value=1., max_value=10., value=3.5,
    step=0.25, format='%.2f') / 100

# RoR = st.sidebar.number_input(
#     'Average Rate of Return (%)', min_value=1., max_value=15., value=6.,
#     step=0.25, format='%.1f') / 100
# RoR = marketReturns['Net Returns'].median()

returnRange = st.sidebar.slider(
    'Return Bounds (Percentile)',
    min_value=1, max_value=99,
    value=[25, 75], step=1)

ageMax = st.sidebar.slider(
    'Age of Survivability',
    min_value=age + 5,
    max_value=age + 100,
    value=100)

# %% Calculations

# Get rate of returns; median and upper and lower bounds (from user)
medianReturn = simPercentileYearly['50 Percentile'].divide(100)
upperReturn = simPercentileYearly[str(
    int(returnRange[1])) + ' Percentile'].divide(100)
lowerReturn = simPercentileYearly[str(
    int(returnRange[0])) + ' Percentile'].divide(100)

# List of ages
ages = list(range(age, 81))
yearsAway = [x - age for x in ages]

# Basic FIRE Parameters
yearlySavings = currentSalary * savingsRate
netIncome = currentSalary - yearlySavings
retirementGoal = netIncome / SWR

# Accumalation Years Growth
print('Running lower growth')
minFV, minTotalGrowth = growth(
    currentSavings, yearsAway, lowerReturn, yearlySavings)
print('Running upper growth')
maxFV, maxTotalGrowth = growth(
    currentSavings, yearsAway, upperReturn, yearlySavings)
print('Running median growth')
FV, totalGrowth = growth(
    currentSavings, yearsAway, medianReturn, yearlySavings)

achieveRE = []
achieveRE.append(milestone_interp(
    FV, yearsAway, retirementGoal) + age)
achieveRE.append(milestone_interp(
    minFV, yearsAway, retirementGoal) + age)
achieveRE.append(milestone_interp(
    maxFV, yearsAway, retirementGoal) + age)
print('achieveRE: ' + str(achieveRE))

# Retirement Years
retireAge = math.ceil(achieveRE[0])
yearsRetire = list(range(ageMax - retireAge + 1))
retirementValue, _ = growth(
    retirementGoal, yearsRetire, medianReturn, -netIncome)
retirementValueMax, _ = growth(
    retirementGoal, yearsRetire, upperReturn, -netIncome)
retirementValueMin, _ = growth(
    retirementGoal, yearsRetire, lowerReturn, -netIncome)

# %% Plot Investment Growth and FIRE Milestones
# fig, ax = plt.subplots(3, sharex=True)
fig = plt.figure(figsize=(10, 10))

ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
ax3 = fig.add_subplot(3, 1, 3)

# Savings over time
ax1.plot(ages, FV)
ax1.fill_between(ages, minFV, maxFV, alpha=0.2)
ax1.axhline(retirementGoal, label='Retirement Goal',
            linestyle='--', linewidth=1, color='g')
ax1.annotate(
    'Retirement Goal: $' + str(round(retirementGoal / 1e6, 2)) + ' mil',
    (age, retirementGoal),  # this is the point to label
    textcoords="offset points",  # how to position the text
    xytext=(1, 2),  # distance from text to points (x,y)
    ha='left')  # horizontal alignment can be left, right or center
ax1.annotate(
    str(round(achieveRE[0], 1)),  # this is the text
    (achieveRE[0], retirementGoal),  # this is the point to label
    textcoords="offset points",  # how to position the text
    xytext=(0, 10),  # distance from text to points (x,y)
    ha='center')  # horizontal alignment can be left, right or center
ax1.annotate(
    str(round(achieveRE[1], 1)),  # this is the text
    (achieveRE[1], retirementGoal),  # this is the point to label
    textcoords="offset points",  # how to position the text
    xytext=(+10, -10),  # distance from text to points (x,y)
    ha='center')  # horizontal alignment can be left, right or center
ax1.annotate(
    str(round(achieveRE[2], 1)),  # this is the text
    (achieveRE[2], retirementGoal),  # this is the point to label
    textcoords="offset points",  # how to position the text
    xytext=(-10, -10),  # distance from text to points (x,y)
    ha='center')  # horizontal alignment can be left, right or center

ax1.set_xlim([age, achieveRE[1] + 5])
ax1.set_ylim([0, retirementGoal * 2])
ax1.yaxis.set_major_formatter('${x:,.0f}')
ax1.set_title('Retirement Savings')

# Investment Gains
ax2.plot(ages, totalGrowth)
ax2.fill_between(ages, minTotalGrowth, maxTotalGrowth, alpha=0.2)
ax2.axhline(currentSalary, label='Salary',
            linestyle='--', linewidth=1, color='darkorange')
ax2.axhline(netIncome, label='Net Income',
            linestyle='--', linewidth=1, color='r')
ax2.axhline(yearlySavings, label='Yearly Savings',
            linestyle='--', linewidth=1, color='g')

ax2.set_xlabel('Age')
ax2.set_ylim([0, currentSalary * 3])
ax2.yaxis.set_major_formatter('${x:,.0f}')
ax2.legend()
ax2.set_title('Invesment Growth')

# Plot Retirement Strategies
# yearMax = 100 - retireAge
yearMax = ageMax - retireAge

ax3.plot(yearsRetire, retirementValue)
ax3.fill_between(yearsRetire,
                 retirementValueMin,
                 retirementValueMax,
                 alpha=0.2)
ax3.axhline(retirementGoal,
            linestyle='--', linewidth=1, color='g')
ax3.annotate(
    'Retirement Start: $' + str(round(retirementGoal / 1e6, 2)) + ' mil',
    (yearMax, retirementGoal),  # this is the point to label
    textcoords="offset points",  # how to position the text
    xytext=(-2, -10),  # distance from text to points (x,y)
    ha='right')  # horizontal alignment can be left, right or center

ax3.set_xlabel('Retirement Year\nAge')
ax3.set_ylim([0, retirementGoal * 2])
ax3.set_xlim([0, yearMax])
ax3.yaxis.set_major_formatter('${x:,.0f}')
ax3.xaxis.set_major_formatter(
    lambda x, pos: str(int(x)) + '\n' + str(int(x + retireAge)))
ax3.set_title('Savings through Retirement')


fig.tight_layout()
st.pyplot(fig)

# %% Description
with st.expander('Description of Method'):

    '''
    This model assumes:
    * 100% stock portfolio matching the market\'s returns
    * Market returns are after inflation; everything is presented in
    today\'s dollars
    * Your salary, savings rate and expenses stay static
    '''

    st.latex('Yearly Savings = Salary * Savings Rate')
    st.latex('Net Income = Salary - Yearly Savings')
    st.latex(r'Retirement Goal = \frac{Net Income}{SWR}')

    '''
    Market return data from [Shiller](http://www.econ.yale.edu/~shiller/data.htm).
    Yearly market returns were fit to (nearly) all continuous distributions in
    SciPy and found all that returned >0.95 p-value. These distributions are
    shown in the first plot below. Then random values for each distribution
    (with p-value >0.95) were created for 100 years 1000 times. The lower plots
    show the percentiles of these randomized sets of values. \n
    The median of these simulated market returns is used as the projected
    return for the savings calculations. Error bars are set using the
    _Return Bounds_ sliders in the options.
    '''

    githubRepo = 'https://raw.githubusercontent.com/sckilcoyne/FIRE/'
    githubURL = githubRepo + githubBranch + githubFolder
    st.markdown('![Market Returns](' + githubURL + 'Market%20Returns.png)')
