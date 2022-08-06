#%% import data
%matplotlib inline

import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels as sm
import matplotlib
import matplotlib.pyplot as plt
import pathlib
import json


matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')

datafolder = pathlib.Path.cwd().parent / 'Data'
outputfolder = pathlib.Path.cwd() / 'Forecasting' / 'Output'

sep_2018 = datafolder / 'feature_gen_dists_sep_2018.csv'
oct_2018 = datafolder / 'feature_gen_dists_oct_2018.csv'
nov_2018 = datafolder / 'feature_gen_dists_nov_2018.csv'
dec_2018 = datafolder / 'feature_gen_dists_dec_2018.csv'
jan_2019 = datafolder / 'feature_gen_dists_jan_2019.csv'
feb_2019 = datafolder / 'feature_gen_dists_feb_2019.csv'
mar_2019 = datafolder / 'feature_gen_dists_mar_2019.csv'
apr_2019 = datafolder / 'feature_gen_dists_apr_2019.csv'
may_2019 = datafolder / 'feature_gen_dists_may_2019.csv'
jun_2019 = datafolder / 'feature_gen_dists_jun_2019.csv'
jul_2019 = datafolder / 'feature_gen_dists_jul_2019.csv'
aug_2019 = datafolder / 'feature_gen_dists_aug_2019.csv'


data = pd.read_csv(sep_2018, sep=',', low_memory=False).append(pd.read_csv(oct_2018, sep=',', low_memory=False)).append(pd.read_csv(nov_2018, sep=',', low_memory=False)).append(pd.read_csv(dec_2018, sep=',', low_memory=False)).append(pd.read_csv(jan_2019, sep=',', low_memory=False)).append(pd.read_csv(feb_2019, sep=',', low_memory=False)).append(pd.read_csv(mar_2019, sep=',', low_memory=False)).append(pd.read_csv(apr_2019, sep=',', low_memory=False)).append(pd.read_csv(may_2019, sep=',', low_memory=False)).append(pd.read_csv(jun_2019, sep=',', low_memory=False)).append(pd.read_csv(jul_2019, sep=',', low_memory=False)).append(pd.read_csv(aug_2019, sep=',', low_memory=False))

data.columns

data.loc[data['StRank'] == 100, 'State'] = 'MS'
data.loc[data['StRank'] == 76, 'State'] = 'LA'
data.loc[data['StRank'] == 56, 'State'] = 'SC'
data.loc[data['StRank'] == 36, 'State'] = 'TN'
data.loc[data['StRank'] == 50, 'State'] = 'AL'
data.loc[data['StRank'] == 24, 'State'] = 'GA'
data.loc[data['StRank'] == 8, 'State'] = 'TX'
data.loc[data['StRank'] == 18, 'State'] = 'KY'


# %% the for loop of distribution making

count=1
dist_dic = {}

for product_type in data['ProductType'].unique():
    dist_dic[product_type] = {}
    for state in data.loc[data['ProductType'] == product_type]['State'].unique():
        y, x = np.histogram(data.loc[(data['ProductType'] == product_type) & (data['State'] == state)]['Tier_MultipleModels'], bins=5, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0
        print(x)
        print(y)
        ax = data.loc[(data['ProductType'] == product_type) & (data['State'] == state)]['Tier_MultipleModels'].plot(kind='hist', bins=5, normed=True)
        dataYLim = ax.get_ylim()
        # Distributions to check

        DISTRIBUTIONS = [
            st.alpha,st.beta,st.chi,st.chi2,st.dgamma,st.dweibull,st.erlang,st.expon,st.gamma,st.cauchy,
            st.halflogistic,st.halfnorm,st.invweibull,st.levy,st.logistic,st.loggamma,st.lognorm,st.maxwell,
            st.norm,st.pareto,st.pearson3,st.powerlognorm,st.powernorm,st.t,st.triang,st.uniform,st.weibull_min,st.weibull_max
        ]
        # Best holders
        best_distribution = st.norm
        best_params = (0.0, 1.0)
        best_sse = np.inf
        print(best_sse)
        # Estimate distribution parameters from data
        for distribution in DISTRIBUTIONS:

            # Try to fit the distribution
            try:
                # Ignore warnings from data that can't be fit
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')

                    # fit dist to data
                    params = distribution.fit(data.loc[(data['ProductType'] == product_type) & (data['State'] == state)]['Tier_MultipleModels'])
                    print(params)
                    # Separate parts of parameters
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]
                    print(arg)
                    print(loc)
                    print(scale)
                    # Calculate fitted PDF and error with fit in distribution
                    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                    sse = np.sum(np.power(y - pdf, 2.0))
                    print(pdf)
                    print(sse)
                    # if axis pass in add to plot
                    try:
                        if ax:
                            pd.Series(pdf, x).plot(ax=ax)
                        end
                    except Exception:
                        pass

                    # identify if this distribution is better
                    if best_sse > sse > 0:
                        best_distribution = distribution
                        best_params = params
                        best_sse = sse

            except Exception:
                pass
        print(best_distribution.name,'  Params = ', best_params)
        best_fit_name = best_distribution.name
        params = best_params

        size = 5
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        # Get sane start and end points of distribution
        dist = getattr(st, best_fit_name)
        start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
        end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

        # Build PDF and turn into pandas Series
        x = np.linspace(start, end, size)
        y = dist.pdf(x, loc=loc, scale=scale, *arg)
        pdf = pd.Series(y, x)
        print(pdf)

        dist_dic[product_type][state] = {'Best_Dist':best_fit_name, 'Best_Params':params, 'pdf':pdf}


outfile = outputfolder / 'tier_distributions.json'
with open(outfile, 'w') as output:
    json.dump(dist_dic, output)


dist_dic
