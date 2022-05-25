import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from binance.client import Client
client = Client()




# Get Data
# coins = ('BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT', 'XRPUSDT', 'LTCUSDT', 'BCHUSDT', 'LINKUSDT', 'MANAUSDT', 'DOGEUSDT', 'VETUSDT')

# function to get data from binance api and convert it to readable data frame
def get30minutedata(symbol, lookback):
    # grab data in data frame
    frame = pd.DataFrame(client.get_historical_klines(symbol, '30m', lookback + ' days ago UTC'))
    # get first 5 columns as those are the only ones we need
    frame = frame.iloc[:, :5]
    # rename columns
    frame.columns = ['Date', 'Open', 'High', 'Low', 'Close']
    # turn values to floats
    frame[['Open', 'High', 'Low', 'Close']] = frame[['Open', 'High', 'Low', 'Close']].astype(float)
    # change from unix time to standard time system
    frame.Date = pd.to_datetime(frame.Date, unit='ms')
    # make index the first column which is Dates
    frame.set_index('Date', inplace=True, drop=True)
    return frame

# call function to get BTC data in 30 minute intervals for the previous 20 days
bin_data = get30minutedata('BTCUSDT', '20')





# Create Strategy Parameters as a fucntion
# short_term_sma : takes an int for the length of smaller SMA
# long_term_sma : takes an int for the length of larger SMA
# data : data to create indicators and back test on
def SMABacktest(data, short_term_sma, long_term_sma, fees, shorts=False):

    # create both SMAs (Simple moving Average)
    data['SMA1'] = data['Close'].rolling(short_term_sma).mean()
    data['SMA2'] = data['Close'].rolling(long_term_sma).mean()

    # get previous value of SMA's
    data['SMA1[1]'] = data['SMA1'].shift(1)
    data['SMA2[1]'] = data['SMA2'].shift(1)


    # make column equal to 1 when a long trade is open, -1 for a short trade, and 0 for no trade
    if shorts:
        data['position'] = np.where(
            data['SMA1'] > data['SMA2'], 1, -1)

    else:
        data['position'] = np.where(
            data['SMA1'] > data['SMA2'], 1, 0)

        # get number of trades opened and closed
        data['trade_num_open'] = np.where(
            np.logical_and(data['SMA1'] > data['SMA2'], data['SMA1[1]'] < data['SMA2[1]']), 1, 0)
        data['trade_num_closed'] = np.where(
            np.logical_and(data['SMA1'] < data['SMA2'], data['SMA1[1]'] > data['SMA2[1]']), 1, 0)


    data['opened_trades'] = data['trade_num_open'].sum()
    data['closed_trades'] = data['trade_num_closed'].sum()
    data['fees_paid'] = (data['closed_trades'] * fees) + (data['opened_trades'] * fees)


    # Calculate returns by creating new columns in data frame
    data['returns'] = data['Close'] / data['Close'].shift(1)
    data['log_returns'] = np.log(data['returns'])
    data['strat_returns'] = data['position'].shift(1) * \
                            data['returns']
    data['strat_log_returns'] = data['position'].shift(1) * \
                                data['log_returns']
    data['cum_returns'] = np.exp(data['log_returns'].cumsum())

    # get cumulative return and sutract fees from it
    data['strat_cum_returns'] = (np.exp(data['strat_log_returns'].cumsum())) - (data['fees_paid']/100) #- (data['trade_num_open'] * fees) - (data['trade_num_closed'] * fees)

    data['peak'] = data['cum_returns'].cummax()
    data['strat_peak'] = data['strat_cum_returns'].cummax()

    return data

# Call Function
short_term_sma = 50
long_term_sma = 200
data = SMABacktest(bin_data, short_term_sma, long_term_sma, fees=0.1)

print(data.to_string())


# Returns more stats on strategy
# risk_free_rate : used to calc sharpe ratio
def getStratStats(data, risk_free_rate=0.02):
    sma_strat, buy_hold_strat = {}, {}

    # Total Returns
    sma_strat['tot_returns'] = np.exp(data['strat_log_returns'].sum()) - 1
    buy_hold_strat['tot_returns'] = np.exp(data['log_returns'].sum()) - 1

    # Mean Annual Returns
    sma_strat['annual_returns'] = np.exp(data['strat_log_returns'].mean() * 252) - 1
    buy_hold_strat['annual_returns'] = np.exp(data['log_returns'].mean() * 252) - 1

    # Annual Volatility
    sma_strat['annual_volatility'] = data['strat_log_returns'].std() * np.sqrt(252)
    buy_hold_strat['annual_volatility'] = data['log_returns'].std() * np.sqrt(252)

    # Max Drawdown
    _strat_dd = data['strat_peak'] - data['strat_cum_returns']
    _buy_hold_dd = data['peak'] - data['cum_returns']
    sma_strat['max_drawdown'] = _strat_dd.max()
    buy_hold_strat['max_drawdown'] = _buy_hold_dd.max()


    # Max Drawdown Duration
    strat_dd = _strat_dd[_strat_dd == 0]
    strat_dd_diff = strat_dd.index[1:] - strat_dd.index[:-1]
    strat_dd_days = strat_dd_diff.map(lambda x: x.days).values
    strat_dd_days = np.hstack([strat_dd_days,
                               (_strat_dd.index[-1] - strat_dd.index[-1]).days])

    buy_hold_dd = _buy_hold_dd[_buy_hold_dd == 0]
    buy_hold_diff = buy_hold_dd.index[1:] - buy_hold_dd.index[:-1]
    buy_hold_days = buy_hold_diff.map(lambda x: x.days).values
    buy_hold_days = np.hstack([buy_hold_days,
                               (_buy_hold_dd.index[-1] - buy_hold_dd.index[-1]).days])
    sma_strat['max_drawdown_duration'] = strat_dd_days.max()
    buy_hold_strat['max_drawdown_duration(intervals)'] = buy_hold_days.max()

    # Calmar Ratio
    sma_strat["calamar_ratio"] = sma_strat['annual_returns'] / sma_strat['max_drawdown']
    buy_hold_strat["calamar_ratio"] = buy_hold_strat['annual_returns'] / buy_hold_strat['max_drawdown']

    # Sharpe Ratio
    sma_strat['sharpe_ratio'] = (sma_strat['annual_returns'] - risk_free_rate) \
                                / sma_strat['annual_volatility']
    buy_hold_strat['sharpe_ratio'] = (
                                             buy_hold_strat['annual_returns'] - risk_free_rate) \
                                     / buy_hold_strat['annual_volatility']


    # Fees paid


    # Trades Closed

    stats_dict = {'strat_stats': sma_strat,
                  'buy&hold_stats': buy_hold_strat}

    return stats_dict


# call stratstats function and create dataframe to display
stats_dict = getStratStats(data, risk_free_rate=0.02)
stats_dict = pd.DataFrame(stats_dict).round(3)

print(stats_dict)






# Plot Data
fig, ax = plt.subplots(2, figsize=(10, 5), sharex=True)

# define width of candlestick elements
width = .01
width2 = .0018

# define up and down prices
up = bin_data[bin_data.Close>=bin_data.Open]
down = bin_data[bin_data.Close<bin_data.Open]

# define colors to use
col1 = 'green'
col2 = 'red'

# plot up prices
ax[0].bar(up.index,up.Close-up.Open,width,bottom=up.Open,color=col1)
ax[0].bar(up.index,up.High-up.Close,width2,bottom=up.Close,color=col1)
ax[0].bar(up.index,up.Low-up.Open,width2,bottom=up.Open,color=col1)

# plot down prices
ax[0].bar(down.index,down.Close-down.Open,width,bottom=down.Open,color=col2)
ax[0].bar(down.index,down.High-down.Open,width2,bottom=down.Open,color=col2)
ax[0].bar(down.index,down.Low-down.Close,width2,bottom=down.Close,color=col2)

ax[0].plot(data['SMA1'], label=f"{short_term_sma}-Day SMA")
ax[0].plot(data['SMA2'], label=f"{long_term_sma}-Day SMA")
ax[0].set_ylabel('Price ($)')
ax[0].set_title(f'{bin_data} Price with {short_term_sma}-Day SMA and {long_term_sma}-Day SMA')
ax[0].legend(bbox_to_anchor=[1, 0.75])
ax[0].grid()


ax[1].plot((data['strat_cum_returns'] - 1) * 100, label='SMA Strategy')
ax[1].plot((data['cum_returns'] - 1) * 100, label='Buy and Hold Strategy')
ax[1].set_ylabel('Returns (%)')
ax[1].set_xlabel('Date')
ax[1].set_title(f'Cumulative Returns for SMA and Buy and Hold Strategy')
ax[1].legend(bbox_to_anchor=[1.25, 0.75])
ax[1].grid()
plt.show()