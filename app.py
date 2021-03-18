# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 17:32:25 2021

@author: Scott
"""

# %% Set up
# import git
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import requests

# Extras for requirements
import tables
# import pickle5 as pickle


# %% App Header
st.set_page_config(page_title='FIRE Milestone Calculator', page_icon=None)
st.title('FIRE Milestone Calculator')

# %%Import Data

# Github data sources
githubRepo = 'https://github.com/sckilcoyne/FIRE/'
# githubBranch = git.Repo(
#     search_parent_directories=True).active_branch.name + '/'
# githubBranch = 'main' + '/'
githubBranch = 'sim-error-bars' + '/'
githubFolder = 'Outputs/'
githubURL = githubRepo + 'blob/' + githubBranch + githubFolder


@st.cache
def import_from_github(githubURL):

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
    open('resultsDf_github.h5', 'wb').write(r.content)
    distFits = pd.read_hdf('resultsDf_github.h5', 'results')

    r = requests.get(simPerformFile, allow_redirects=True)
    open('simulatedPerformance_github.h5', 'wb').write(r.content)
    simPerform = pd.read_hdf('simulatedPerformance_github.h5',
                             'simulatedPerformance')

    r = requests.get(simPercentileFile, allow_redirects=True)
    open('simulatedPerformanceStats_github.h5', 'wb').write(r.content)
    simPercentile = pd.read_hdf(
        'simulatedPerformanceStats_github.h5', 'simulatedPerformanceStats')

    r = requests.get(simPercentileYearlyFile, allow_redirects=True)
    open('simPercentileYearly_github.h5', 'wb').write(r.content)
    simPercentileYearly = pd.read_hdf(
        'simPercentileYearly_github.h5', 'simPercentileYearly')

    r = requests.get(marketReturnsFile, allow_redirects=True)
    open('marketReturns_github.h5', 'wb').write(r.content)
    marketReturns = pd.read_hdf(
        'marketReturns_github.h5', 'marketReturns')

    return distFits, simPerform, simPercentile, simPercentileYearly, marketReturns


distFits, simPerform, simPercentile, simPercentileYearly, marketReturns = import_from_github(
    githubURL)

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
    step=0.25, format='%.1f') / 100

# RoR = st.sidebar.number_input(
#     'Average Rate of Return (%)', min_value=1., max_value=15., value=6.,
#     step=0.25, format='%.1f') / 100
RoR = marketReturns['Net Returns'].median()

returnRange = st.sidebar.slider(
    'Return Bounds (Percentile)',
    min_value=1, max_value=99,
    value=[25, 75], step=1)

# simPercentileYearly[0:1] = 0
medianReturn = simPercentileYearly['50 Percentile'].divide(100)
upperReturn = simPercentileYearly[str(
    int(returnRange[1])) + ' Percentile'].divide(100)
lowerReturn = simPercentileYearly[str(
    int(returnRange[0])) + ' Percentile'].divide(100)

# %% Calculations

ages = list(range(age, 81))
yearsAway = [x - age for x in ages]

yearlySavings = currentSalary * savingsRate
netIncome = currentSalary - yearlySavings
retirementGoal = netIncome / SWR


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
    if type(yearlyValues) is list:
        # print(yearlyValues)
        overshootValue = list(
            filter(lambda k: k > milestoneGoal, yearlyValues))
        # print('past goal: ' + str(overshootValue))
        try:
            overshootValue = overshootValue[0]
        except IndexError:
            overshootValue = yearlyValues[-1]
            print('Does not reach goal!')

        # print('past goal: ' + str(overshootValue))
        overshotYear = yearlyValues.index(overshootValue)
        # print('goal index: ' + str(overshotYear))
        years2goal = yearList[overshotYear]
    else:
        print('Wrong milestone function: ' + str(type(yearlyValues)))
        years2goal = 0

    return years2goal

# # Assumed RoR
# minFV, minTotalGrowth = growth(
#     currentSavings, yearsAway, RoR - 0.02, yearlySavings)
# maxFV, maxTotalGrowth = growth(
#     currentSavings, yearsAway, RoR + 0.02, yearlySavings)
# FV, totalGrowth = growth(currentSavings, yearsAway, RoR, yearlySavings)

# achieveRE = []
# achieveRE.append(milestone_achievement(
#     RoR, currentSavings, yearlySavings, retirementGoal) + age)
# achieveRE.append(milestone_achievement(
#     RoR - 0.02, currentSavings, yearlySavings, retirementGoal) + age)
# achieveRE.append(milestone_achievement(
#     RoR + 0.02, currentSavings, yearlySavings, retirementGoal) + age)


# Sim RoR
print('Running lower growth')
minFV, minTotalGrowth = growth(
    currentSavings, yearsAway, lowerReturn, yearlySavings)
# print(minFV)
print('Running upper growth')
maxFV, maxTotalGrowth = growth(
    currentSavings, yearsAway, upperReturn, yearlySavings)
# print(maxFV)
print('Running median growth')
FV, totalGrowth = growth(
    currentSavings, yearsAway, medianReturn, yearlySavings)
# print(FV)

achieveRE = []
# achieveRE.append(milestone_calc(
#     medianReturn, currentSavings, yearlySavings, retirementGoal) + age)
# achieveRE.append(milestone_calc(
#     lowerReturn, currentSavings, yearlySavings, retirementGoal) + age)
# achieveRE.append(milestone_calc(
#     upperReturn, currentSavings, yearlySavings, retirementGoal) + age)

achieveRE.append(milestone_interp(
    FV, yearsAway, retirementGoal) + age)
achieveRE.append(milestone_interp(
    minFV, yearsAway, retirementGoal) + age)
achieveRE.append(milestone_interp(
    maxFV, yearsAway, retirementGoal) + age)

print('achieveRE: ' + str(achieveRE))


# %% Plot Investment Growth and FIRE Milestones
fig, ax = plt.subplots(2, sharex=True)

# Savings over time
ax[0].plot(ages, FV)
ax[0].fill_between(ages, minFV, maxFV, alpha=0.2)
ax[0].axhline(retirementGoal, label='Retirement Goal',
              linestyle='--', linewidth=1, color='g')
ax[0].annotate(
    'Retirement Goal: $' + str(round(retirementGoal / 1e6, 2)) + ' mil',
    (age, retirementGoal),  # this is the point to label
    textcoords="offset points",  # how to position the text
    xytext=(1, 2),  # distance from text to points (x,y)
    ha='left')  # horizontal alignment can be left, right or center
ax[0].annotate(
    str(round(achieveRE[0], 1)),  # this is the text
    (achieveRE[0], retirementGoal),  # this is the point to label
    textcoords="offset points",  # how to position the text
    xytext=(0, 10),  # distance from text to points (x,y)
    ha='center')  # horizontal alignment can be left, right or center
ax[0].annotate(
    str(round(achieveRE[1], 1)),  # this is the text
    (achieveRE[1], retirementGoal),  # this is the point to label
    textcoords="offset points",  # how to position the text
    xytext=(+10, -10),  # distance from text to points (x,y)
    ha='center')  # horizontal alignment can be left, right or center
ax[0].annotate(
    str(round(achieveRE[2], 1)),  # this is the text
    (achieveRE[2], retirementGoal),  # this is the point to label
    textcoords="offset points",  # how to position the text
    xytext=(-10, -10),  # distance from text to points (x,y)
    ha='center')  # horizontal alignment can be left, right or center

ax[0].set_xlim([age, achieveRE[1] + 5])
# ax[0].set_xlim([age, 60])
ax[0].set_ylim([0, retirementGoal * 2])
ax[0].yaxis.set_major_formatter('${x:1.0f}')
# ax[0].legend()
ax[0].set_title('Retirement Savings')

# Investment Gains
ax[1].plot(ages, totalGrowth)
ax[1].fill_between(ages, minTotalGrowth, maxTotalGrowth, alpha=0.2)
ax[1].axhline(currentSalary, label='Salary',
              linestyle='--', linewidth=1, color='k')
ax[1].axhline(netIncome, label='Net Income',
              linestyle='--', linewidth=1, color='r')
ax[1].axhline(yearlySavings, label='Yearly Savings',
              linestyle='--', linewidth=1, color='g')

ax[1].set_xlabel('Age')
ax[1].set_ylim([0, currentSalary * 3])
ax[1].yaxis.set_major_formatter('${x:1.0f}')
ax[1].legend()
ax[1].set_title('Invesment Growth')

fig.tight_layout()
st.pyplot(fig)

# %% Description
st.subheader('Description')
st.text('Yearly Savings = Salary * Savings Rate' + '\n' +
        'Net Income = Salary - Yearly Savings' + '\n' +
        'Retirement Goal = Net Income / SWR' + '\n')

githubRepo = 'https://raw.githubusercontent.com/sckilcoyne/FIRE/'
githubURL = githubRepo + githubBranch + githubFolder
st.markdown('![Market Returns](' + githubURL + 'Market%20Returns.png)')
