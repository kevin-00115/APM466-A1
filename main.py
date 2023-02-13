import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import datetime
import numpy_financial as npf
import numpy as np
from scipy import optimize
from scipy.interpolate import interp1d

class bond:
    def __init__(self, ISIN, maturity_date, coupon, issue_date, price:list):
        self.ISIN = ISIN
        self.maturity_date = maturity_date
        self.issue_date = issue_date
        self.coupon_rate = coupon
        self.price = price

    def ytm(self):
        ytm_list = []
        face_value = 100
        coupon_amount = self.coupon_rate * face_value
        time = int((self.maturity_date - self.issue_date).days/365)
        for price in self.price:
            ytm = ( coupon_amount + (face_value - price)/time)/(face_value + price)/2
            ytm_list.append(ytm)
        return ytm_list

    def present_value(self, day):
        dirty_price = bond.coupon_rate * (183 - (31 - day) - 29) / 183
        # with dirty price and close price we can calculate the present
        # value for a certain bond based on this day's observation
        present_value = round(dirty_price + bond.price[day], 4)
        return present_value

    def get_spot_rate(self):
        terms = np.arange(1,5.1,0.1)
        yield_interpolated = np.interp(terms, np.arange(len(self.price)), self.ytm())
        return yield_interpolated

bond_list = []
ytm_list = []
price_E679 = [94.84,94.94,95.27,95.34,95.02,94.95,94.96,95.1,95.09,94.8]
maturity1 = datetime.date(2026,6,1)
issue1 = datetime.date(2015,7,21)
bond_E679 = bond("CA135087E679", maturity1, 0.015, issue1, price_E679)
bond_list.append(bond_E679)
#print(bond_E679.get_spot_rate())


price_F825 = [92.2,92.27,92.38,92.87,92.85,92.35,92.36,92.40,92.54,92.5]
maturity2 = datetime.date(2027,6,1)
issue2 = datetime.date(2016,8,3)
bond_F825 = bond("CA135087F825",maturity2, 0.001, issue2, price_F825)
bond_list.append(bond_F825)
print(bond_F825.get_spot_rate())

price_H235 = [95.91,96.02,96.61,96.59,96.04,95.94,95.95,96.12,96.04,95.62]
maturity3 = datetime.date(2028,6,1)
issue3 = datetime.date(2017,8,1)
bond_H235 = bond("CA135087H235", maturity3, 0.002, issue3, price_H235)
bond_list.append(bond_H235)

price_J397 = [96.81, 96.9, 97.59, 97.57, 96.87, 96.8, 96.82, 97.04, 96.92, 96.39]
maturity4 = datetime.date(2029,6,1)
issue4 = datetime.date(2018,7,27)
bond_J397 = bond("CA135087J397", maturity4, 0.0225, issue4, price_J397)
bond_list.append(bond_J397)

price_L443 = [83.54, 83.68, 84.42,84.44,83.74,83.65,83.79,84,83.91,83.43]
maturity5 = datetime.date(2030,12,1)
issue5 = datetime.date(2020,10,5)
bond_L443 = bond("CA135087L443", maturity5, 0.0005, issue5, price_L443)
bond_list.append(bond_L443)

price_L518 = [91.08, 91.19,91.43, 91.51, 91.32, 91.23, 91.20, 91.41, 91.45, 91.17]
maturity6 = datetime.date(2026,3,1)
issue6 = datetime.date(2020, 10,9)
bond_L518 = bond("CA135087L518", maturity6, 0.0025, issue6, price_L518)
bond_list.append(bond_L518)

price_L930 = [92.72, 92.84,93.34,93.29,92.95,92.90,92.91,93.05,93.04,92.77]
maturity7 = datetime.date(2026,9,1)
issue7 = datetime.date(2021,4,16)
bond_L930 = bond("CA135087L930", maturity7, 0.001, issue7, price_L930)
bond_list.append(bond_L930)

price_M276 = [89.89, 90.06, 90.85, 90.86, 90.08, 90, 90.14, 90.35, 90.23, 89.72]
maturity8 = datetime.date(2031,6,1)
issue8 = datetime.date(2021,4,26)
bond_M276 = bond("CA135087M276", maturity8, 0.015, issue8, price_M276)
bond_list.append(bond_M276)

price_M847 = [93.14, 93.25, 93.5, 93.78, 93.39, 93.33, 93.35, 93.5, 93.46, 93.14]
maturity9 = datetime.date(2027, 3,1)
issue9 = datetime.date(2021, 10, 15)
bond_M847 = bond("CA135087M847", maturity9, 0.0125, issue9, price_M847)
bond_list.append(bond_M847)

price_N266 = [89.32, 89.37, 89.52, 90.32, 90.32, 89.51, 89.42, 89.55, 89.78, 89.55]
maturity10 = datetime.date(2031, 12, 1)
issue10 = datetime.date(2021, 10, 25)
bond_N266 = bond("CA135087N266", maturity10, 0.015, issue10, price_N266)
bond_list.append(bond_N266)

for bond in bond_list:
    ytm_list.append(bond.ytm())

spot_list = []
for bond in bond_list:
    spot_list.append(bond.get_spot_rate())

YTM = ['YTM1', 'YTM2', 'YTM3', 'YTM4', 'YTM5', 'YTM6', 'YTM7', 'YTM8', 'YTM9', 'YTM10']
plt.xlabel('time to maturity')
plt.ylabel('yield to maturity')
plt.title('five year yield curve')
plt.xticks(ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],labels = ['23/3','23/9','24/3','24/9', '25/3', '25/9', '26/3', '26/9', '27/3', '27/9','2028/3'])
for i in range(10):
    plt.plot(ytm_list[i], label = YTM[i])
plt.legend(loc=1, prop={'size': 6})

plt.show()

YTM = ['YTM1', 'YTM2', 'YTM3', 'YTM4', 'YTM5', 'YTM6', 'YTM7', 'YTM8', 'YTM9', 'YTM10']
plt.xlabel('time to maturity')
plt.ylabel('yield to maturity')
plt.title('five year spot rate curve')
plt.xticks(ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],labels = ['23/3','23/9','24/3','24/9', '25/3', '25/9', '26/3', '26/9', '27/3', '27/9','2028/3'])
for i in range(10):
    plt.plot(spot_list[i], label = YTM[i])
plt.legend(loc=1, prop={'size': 6})

plt.show()

terms = [1, 2, 3, 4,5]

# Define an empty list to store the forward rates
forward_rates = []

# Loop over the days of data
for day in range(len(spot_list)):

    # Define a list to store the daily forward rates
    daily_forward_rates = []

    # Loop over the terms
    for i, term in enumerate(terms):

        # If the term is 2 years, the forward rate is equal to the average of the 1-year and 2-year spot rates
        if term == 2:
            forward_rate = (spot_list[day][0] + spot_list[day][1]) / 2

        # If the term is greater than 2 years, calculate the forward rate by discounting the future spot rate with the previous forward rate
        else:
            forward_rate = (spot_list[day][i] * term - spot_list[day][i - 1] * (term - 1)) / term

        # Add the calculated forward rate to the daily forward rates list
        daily_forward_rates.append(forward_rate)

    # Add the daily forward rates to the overall forward rates list
    forward_rates.append(daily_forward_rates)

# Plot the forward rates, with one curve corresponding to each day of data superimposed on top of each other
for i, day in enumerate(forward_rates):
    plt.plot(terms, day, label=f"Day {i}")

plt.xlabel("Term (years)")
plt.ylabel("Forward rate")
plt.title('1 year forward rate')
plt.legend()
plt.show()


#Assume that yield and forward rates are given as two 2-dimensional arrays, where each row represents a day of data and each column represents a term
# Convert each element in the list into a numpy array
ytm_arrays = [np.array(ytm) for ytm in ytm_list]
# Concatenate the arrays along a new dimension
yields = np.stack(ytm_arrays, axis=0)
forward_rates = np.array(forward_rates)


# Calculate the log-returns of yield and forward rates
yield_log_returns = np.diff(np.log(yields), axis=1)
forward_rate_log_returns = np.diff(np.log(forward_rates), axis=1)

# Calculate the covariance matrix of the log-returns of yield
yield_covariance_matrix = np.cov(yield_log_returns.T)

# Calculate the covariance matrix of the log-returns of forward rates
forward_rate_covariance_matrix = np.cov(forward_rate_log_returns.T)

# The covariance matrices are now stored in the yield_covariance_matrix and forward_rate_covariance_matrix variables, respectively
# print(yield_covariance_matrix)
# print(forward_rate_covariance_matrix)

forward_rate_eigenvalues, forward_rate_eigenvectors = np.linalg.eig(forward_rate_covariance_matrix)
print(forward_rate_eigenvalues)
print(forward_rate_eigenvectors)

yield_log_return_eigenvalues, yield_log_returns_eigenvectors = np.linalg.eig(yield_covariance_matrix)
print(yield_log_return_eigenvalues)
print(yield_log_returns_eigenvectors)