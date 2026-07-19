import numpy as np
import statsmodels.api as sm
import yfinance as yf
import matplotlib.pyplot as plt
from collections import defaultdict


START = '1990-01-01'
END   = '2019-12-31'

#### Market data api ####
def get_data(mkt):
    res = yf.download(mkt, start=START, end=END, auto_adjust=True)
    res.columns = res.columns.droplevel(1) # flatten multiindex
    return res

#### Log return construction ####
def make_returns(data):
    rets = 10000 * (np.log(data['Close'] / data['Close'].shift(1))) # convert to bp for readability
    return rets.dropna().to_frame(name='logreturns')

#### Calendar dummies ####
def add_dummies(df):
    dates = df.index
    df['year']  = dates.year
    df['month'] = dates.month
    df['dow']   = dates.dayofweek

    # jan dummy
    df['D_jan'] = [1 if m == 1 else 0 for m in df['month']]
    # halloween dummy
    df['D_hal'] = [1 if m in [11, 12, 1, 2, 3, 4] else 0 for m in df['month']]
    # days of the week dummy (Monday omitted)
    df['D_tue'] = [1 if d == 1 else 0 for d in df['dow']]
    df['D_wed'] = [1 if d == 2 else 0 for d in df['dow']]
    df['D_thu'] = [1 if d == 3 else 0 for d in df['dow']]
    df['D_fri'] = [1 if d == 4 else 0 for d in df['dow']]
    # monthly dummies (January omitted)
    for m in range(2, 13):
        df[f'M_{m}'] = [1 if month == m else 0 for month in df['month']]
    # turn of the month dummy: last trading day + first three trading days
    totm_dates = []
    i = 0
    while i < len(dates):
        month_idx = dates[i].month
        year_idx = dates[i].year
        row = []
        while i < len(dates) and dates[i].month == month_idx and dates[i].year == year_idx:
            row.append(dates[i])
            i += 1
        totm_dates += row[:3]
        totm_dates += row[-1:]
    df['D_totm'] = [1 if d in totm_dates else 0 for d in df.index]

    return df

#### OLS with HAC ####
def ols_hac(df, use_month_dummies=False, lags=5):
    df = df.copy()
    
    # choose regressors being regressed
    x_cols = ['D_tue', 'D_wed', 'D_thu', 'D_fri', 'D_totm']
    if use_month_dummies:
        x_cols += ['M_2', 'M_3', 'M_4', 'M_5', 'M_6', 'M_7', 'M_8', 'M_9', 'M_10', 'M_11', 'M_12']
    else:
        x_cols += ['D_jan', 'D_hal'] # minimise overlapping dummies

    # intercept y and x
    y = df['logreturns']
    df['B'] = 1
    x = df[['B'] + x_cols]

    # do the regression
    model = sm.OLS(y, x).fit( cov_type='HAC', cov_kwds={'maxlags': lags})

    return model

#### returns by time plot ####
def cluster_png(returns, title):
    res = returns.logreturns
    plt.figure()
    plt.plot(range(len(res)), res, linewidth=0.1)   

    plt.xlabel(r'$\Delta t$')
    plt.ylabel('log returns')
    plt.title(f'Visualisation Returns Clustering : 1990-2019 : {title}')
    plt.show()
    plt.close()

#### plot returns by month ####
def bymonth_png(df, title):
    data_dict = defaultdict(list)
    
    for row in df.itertuples():
        data_dict[row.month].append(row.logreturns)
    for key in data_dict.keys():
        data_dict[key] = np.mean(data_dict[key])
    items = sorted(data_dict.items())
    _, y = zip(*items)

    plt.figure()
    plt.bar(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], y)
    plt.xticks(rotation=90)
    plt.xlabel('Month')
    plt.ylabel('Average Returns (bp)')
    plt.title(f'Average Returns by Month : 1990-2019 : {title}')
    plt.show()
    plt.close()

#### plot returns by weekday ####
def bydow_png(df, title):
    data_dict = defaultdict(list)
    
    for row in df.itertuples():
        data_dict[row.dow].append(row.logreturns)
    for key in data_dict.keys():
        data_dict[key] = np.mean(data_dict[key])
    items = sorted(data_dict.items())
    _, y = zip(*items)

    plt.figure()
    plt.bar(['Mon', 'Tue', 'Wed', 'Thu', 'Fri'], y)
    plt.xlabel('Day of the Week')
    plt.ylabel('Average Returns (bp)')
    plt.title(f'Average Returns by Weekday : 1990-2019 : {title}')
    plt.show()
    plt.close()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

markets = [{'mkt': '^FTSE', 'name': 'FTSE 100'},
           {'mkt': '^GSPC', 'name': 'S&P 500'}]

# run
for market in markets:
    res = get_data(market['mkt'])
    data_rets = make_returns(res)
    cluster_png(data_rets, title=market['name'])
    data_dum = add_dummies(data_rets)
    bydow_png(data_dum, title=market['name'])

    models = [{'monthly': False, 'title': f"\n{market['name']} : Baseline model"},
              {'monthly': True,  'title': f"\n{market['name']} : Month-of-year model"}]

    for model in models:
        if model['monthly']:
            bymonth_png(data_dum, title=model['title'])
    
        df_pre2008 = data_dum[data_dum.index < '2008-01-01'] # split subsamples
        ols = ols_hac(df_pre2008, use_month_dummies=model['monthly'])
        print(model['title'] + ' : pre 2008:\n', ols.summary())
        if 2008 in data_dum.index.year:
            df_post2008 = data_dum[data_dum.index >= '2008-01-01']
            ols = ols_hac(df_post2008, use_month_dummies=model['monthly'])
            print(model['title'] + ' : post 2008:\n', ols.summary())

