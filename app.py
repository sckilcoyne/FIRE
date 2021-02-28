# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 17:32:25 2021

@author: Scott
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.title('FIRE')


# Sidebar Inputs
age = st.sidebar.number_input(
    'Age', min_value=18, max_value=70, value=30, step=1)

currentSavings = st.sidebar.number_input(
    'Current Savings ($)', value=200000, step=1000)

currentSalary = st.sidebar.number_input(
    'Salary ($)', min_value=0, value=90000, step=1000)

savingsRate = st.sidebar.number_input(
    'Savings Rate (%)', min_value=5., max_value=80., value=30.,
    step=1., format='%.1f') / 100

SWR = st.sidebar.number_input(
    'Withdrawal Rate (%)', min_value=1., max_value=10., value=3.5,
    step=0.5, format='%.1f') / 100

RoR = st.sidebar.number_input(
    'Average Rate of Return (%)', min_value=1., max_value=15., value=6.,
    step=0.5, format='%.1f') / 100


# Calculations

ages = list(range(age, 81))
yearsAway = [x - age for x in ages]

yearlySavings = currentSalary * savingsRate
netIncome = currentSalary - yearlySavings
retirementGoal = netIncome / SWR


def growth(currentSavings, yearsAway, RoR, yearlySavings):
    principalGrowth = [currentSavings * (1 + RoR) ** x for x in yearsAway]

    savingsGrowth = [yearlySavings *
                     ((1 + RoR) ** x - 1) / RoR for x in yearsAway]

    FV = [sum(i) for i in zip(principalGrowth, savingsGrowth)]

    totalGrowth = [FV[i] - (i+1)*yearlySavings -
                   currentSavings for i, x in enumerate(FV)]

    return FV, totalGrowth


minFV, minTotalGrowth = growth(
    currentSavings, yearsAway, RoR - 0.02, yearlySavings)
maxFV, maxTotalGrowth = growth(
    currentSavings, yearsAway, RoR + 0.02, yearlySavings)
FV, totalGrowth = growth(currentSavings, yearsAway, RoR, yearlySavings)

# Plots
fig, ax = plt.subplots(2, sharex=True)

ax[0].plot(ages, FV)
ax[0].fill_between(ages, minFV, maxFV, alpha=0.2)
ax[0].axhline(retirementGoal, label='Retirement Goal',
              linestyle='--', linewidth=1, color='g')

ax[0].set_ylim([0, retirementGoal * 2])
ax[0].yaxis.set_major_formatter('${x:1.0f}')
ax[0].legend()
ax[0].set_title('Retirement Savings')

ax[1].plot(ages, totalGrowth)
ax[1].fill_between(ages, minTotalGrowth, maxTotalGrowth, alpha=0.2)
ax[1].axhline(netIncome, label='Net Income',
              linestyle='--', linewidth=1, color='r')
ax[1].axhline(yearlySavings, label='Yearly Savings',
              linestyle='--', linewidth=1, color='g')
ax[1].axhline(currentSalary, label='Salary',
              linestyle='--', linewidth=1, color='k')
ax[1].set_xlabel('Age')
ax[1].set_ylim([0, currentSalary * 3])
ax[1].yaxis.set_major_formatter('${x:1.0f}')
ax[1].legend()
ax[1].set_title('Invesment Growth')

fig.tight_layout()
st.pyplot(fig)
