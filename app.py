# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 17:32:25 2021

@author: Scott
"""

# %% Set up
import streamlit as st
import matplotlib.pyplot as plt
# import numpy as np
import math
import pandas as pd
import requests

# Extras for requirements
import tables
import pickle5


# %% App Header
st.set_page_config(page_title='FIRE Milestone Calculator', page_icon=None)
st.title('FIRE Milestone Calculator')

# %%Import Data


@st.cache
def import_from_github():
    # Github data sources
    githubRepo = 'https://github.com/sckilcoyne/FIRE/'
    githubBranch = 'main/'
    githubFolder = 'Outputs/'
    githubURL = githubRepo + 'blob/' + githubBranch + githubFolder
    raw = '?raw=true'

    resultsDfFile = githubURL + 'results.h5' + raw
    randomGrowthFile = githubURL + 'randomGrowth.h5' + raw
    randomGrowthStatsFile = githubURL + 'randomGrowthStats.h5' + raw
    returnsFile = githubURL + 'returns.h5' + raw

    print(resultsDfFile + '\n' + randomGrowthFile +
          '\n' + randomGrowthStatsFile + '\n' + returnsFile)

    # Import data
    r = requests.get(resultsDfFile, allow_redirects=True)
    open('resultsDf_github.h5', 'wb').write(r.content)
    resultsDf = pd.read_hdf('resultsDf_github.h5', 'results')

    r = requests.get(randomGrowthFile, allow_redirects=True)
    open('randomGrowth_github.h5', 'wb').write(r.content)
    randomGrowth = pd.read_hdf('randomGrowth_github.h5', 'randomGrowth')

    r = requests.get(randomGrowthStatsFile, allow_redirects=True)
    open('randomGrowthStats_github.h5', 'wb').write(r.content)
    randomGrowthStats = pd.read_hdf(
        'randomGrowthStats_github.h5', 'randomGrowthStats')

    r = requests.get(returnsFile, allow_redirects=True)
    open('returns_github.h5', 'wb').write(r.content)
    returns = pd.read_hdf(
        'returns_github.h5', 'returns')

    return resultsDf, randomGrowth, randomGrowthStats, returns


resultsDf, randomGrowth, randomGrowthStats, returns = import_from_github()
# print(resultsDf.head(5))
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
RoR = returns['Net Returns'].median()


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
    principalGrowth = [currentSavings * (1 + RoR) ** x for x in yearsAway]

    savingsGrowth = [yearlySavings *
                     ((1 + RoR) ** x - 1) / RoR for x in yearsAway]

    FV = [sum(i) for i in zip(principalGrowth, savingsGrowth)]

    totalGrowth = [FV[i] - (i+1)*yearlySavings -
                   currentSavings for i, x in enumerate(FV)]

    return FV, totalGrowth


def milestone_achievement(RoR, currentSavings, yearlySavings, milestoneGoal):

    years2goal = math.log((milestoneGoal + yearlySavings / RoR) /
                          (currentSavings + yearlySavings / RoR)) / \
        math.log(1 + RoR)

    return years2goal


minFV, minTotalGrowth = growth(
    currentSavings, yearsAway, RoR - 0.02, yearlySavings)
maxFV, maxTotalGrowth = growth(
    currentSavings, yearsAway, RoR + 0.02, yearlySavings)
FV, totalGrowth = growth(currentSavings, yearsAway, RoR, yearlySavings)

achieveRE = []
achieveRE.append(milestone_achievement(
    RoR, currentSavings, yearlySavings, retirementGoal) + age)
achieveRE.append(milestone_achievement(
    RoR - 0.02, currentSavings, yearlySavings, retirementGoal) + age)
achieveRE.append(milestone_achievement(
    RoR + 0.02, currentSavings, yearlySavings, retirementGoal) + age)

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
        'Retirement Goal = Net Income / SWR' + '\n' +
        '\n' +
        'Error range on plots is +/-2% of median RoR')
# 'Error range on plots is 5th-95th Percentile of Simulated Returns')
st.markdown("![Market Returns](https://raw.githubusercontent.com/sckilcoyne/FIRE/f8969ea9bc17dc007accf02aba3134d756c91db4/Outputs/Market%20Returns.png)")
