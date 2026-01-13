# basic pyhton libraries
import streamlit as st
import pandas as pd
import numpy as np
from datetime import  date
from datetime import datetime, timedelta
import time

# libraries to retrieve and download data
import requests
from io import BytesIO
#import yfinance  as yf
import pandas as pd

import pandas_datareader.data as web
import base64

# ploting libraries
import plotly.express as px
import plotly.graph_objects as go

# for combinations
import itertools

def candlestick_chart(dfs, selected_var):
    suffixes = ['Close_', 'Open_', 'Low_', 'High_']
    candles = []

    # Criar todos os sufixos
    for var in selected_var:
        for suffix in suffixes:
            column_name = f"{suffix}{var}"
            candles.append(column_name)

    # Filtrar o DataFrame com base nos column_names criados
    dfs = dfs[candles]
    traces = []

    for var in selected_var:
        trace = go.Candlestick(
            x=dfs.index,
            open=dfs[f"Open_{var}"],
            high=dfs[f"High_{var}"],
            low=dfs[f"Low_{var}"],
            close=dfs[f"Close_{var}"],
            name=var
        )

        traces.append(trace)

    layout = go.Layout(
        title="Gráfico de Candlestick",
        xaxis=dict(title="Data"),
        yaxis=dict(title="Preço"),
    )

    fig = go.Figure(data=traces, layout=layout)
    return fig


def download_data(data, period="1y", max_retries=3, sleep=1.0, auto_adjust=False):
    dfs = []

    if isinstance(data, dict):
        items = list(data.items())  # (name, ticker)
    elif isinstance(data, list):
        items = [(t, t) for t in data]  # (prefix, ticker)
    else:
        raise TypeError("data must be a dict {name: ticker} or a list of tickers")

    # Convert yfinance-like period strings to a start date for Stooq
    # Stooq via pandas-datareader works best with explicit date ranges.
    period_map_days = {
        "1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180,
        "1y": 365, "2y": 730, "5y": 1825, "10y": 3650, "ytd": None, "max": None
    }

    today = datetime.utcnow().date()

    if period == "ytd":
        start = datetime(today.year, 1, 1).date()
        end = today
    elif period == "max":
        start = None
        end = None
    else:
        days = period_map_days.get(period, 365)  # default to 1y if unknown
        start = today - timedelta(days=days) if days is not None else None
        end = today

    for prefix, ticker in items:
        hist = pd.DataFrame()
        symbol = f"{ticker}.US"  # Stooq uses .US suffix for US equities/ETFs

        for attempt in range(1, max_retries + 1):
            try:
                hist = web.DataReader(symbol, "stooq", start=start, end=end)

                # Stooq returns descending by date; sort ascending
                if hist is not None and not hist.empty:
                    hist = hist.sort_index()
                    break
            except Exception:
                pass

            time.sleep(sleep)

        if hist is None or hist.empty:
            print(f"WARNING: No data returned for {ticker} (prefix={prefix}) for period={period}. Skipping.")
            continue

        # Match yfinance-style columns as closely as possible
        # Stooq typically provides: Open, High, Low, Close, Volume
        if "Close" in hist.columns and "Adj Close" not in hist.columns:
            hist["Adj Close"] = hist["Close"]

        # Ensure DatetimeIndex and normalize to date (no timezone/time component)
        hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()

        # Optional: auto_adjust behavior approximation (Stooq doesn't provide splits/dividends adjustments)
        # If auto_adjust=True, we can at least drop "Adj Close" consistency by setting Close=Adj Close
        if auto_adjust and "Adj Close" in hist.columns and "Close" in hist.columns:
            hist["Close"] = hist["Adj Close"]

        # Keep a consistent column set (if present)
        cols_order = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        hist = hist[[c for c in cols_order if c in hist.columns]]

        # Prefix columns
        hist.columns = [f"{prefix}_{col}" for col in hist.columns]
        dfs.append(hist)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, axis=1, join="outer").sort_index()

def upload_file(file):
    df = pd.read_excel(file) if file.name.endswith('xlsx') else pd.read_csv(file)
    df.set_index('Date', inplace = True, drop = True)
    df.index = pd.to_datetime(df.index)
    return df

def download_dfs(session_state, download_option, mapping):
    attribute_name = mapping.get(download_option)
    if attribute_name is not None:
        data_attribute = getattr(session_state, attribute_name)
        if data_attribute is not None:
            df_to_download = pd.DataFrame(data_attribute)
            csv = df_to_download.to_csv(index=True)  # incluir o índice no arquivo CSV
            b64 = base64.b64encode(csv.encode()).decode()  # codificação B64 para o link de download
            href = f'<a href="data:file/csv;base64,{b64}" download="{download_option}.csv">Download {download_option} Data</a>'
            return href
        else:
            st.warning(f"No data available for {download_option}.")
            return None
    else:
        st.error(f"Invalid download option: {download_option}")
        return None
    
def load_data_from_github(url):
    response = requests.get(url)
    content = BytesIO(response.content)
    data = pd.read_pickle(content)
    return data

def date_resample(df, period='M', aggregation='sum'):
    result_series = df.resample(period).agg(aggregation)
    return result_series

def fill_moving_avg(df, window_size, method='gap'):
    if method == 'gap':
        date_index = df.index
        df.reset_index(drop=True, inplace=True)
        for col in df.select_dtypes(include=[np.number]).columns:
            nan_indices = df[df[col].isna()].index
            for index in nan_indices:
                start = max(0, index - window_size)
                end = index + 1
                window_data = df[col].iloc[start:end]
                mean_value = round(window_data.mean(), 4)
                df.at[index, col] = mean_value
        df.index = date_index
    else:
        numeric_cols = df.select_dtypes(include='number').columns
        for col in numeric_cols:
            df[col] = df[col].rolling(window=window_size, min_periods=1).mean()
            df[col] = df[col].fillna(method='bfill')
    return df
	
def get_latest_values(df, tickers):
    latest_values = {}
    for ticker in tickers:
        latest_index = df[f"{ticker}_Close"].idxmax()
        latest_value = df.at[latest_index, f"{ticker}_Close"]
        latest_values[ticker] = latest_value
    
    return latest_values

def compute_investments(df, weights_df, available_cash):
    latest_values = get_latest_values(df, weights_df['Tickers'])
    investments = pd.DataFrame(columns=['Ticker', 'Price', 'Papers', 'Share %', 'Invested', 'Cash'])

    cash_to_invest = available_cash  # Initialize cash_to_invest

    for ticker, weight in zip(weights_df['Tickers'], weights_df['Weights']):
        price = latest_values[ticker]
        shares = (weight * available_cash) / price
        invested = shares * price
        cash_to_invest -= invested  # Subtract the invested amount for each iteration

        row_data = {'Ticker': ticker, 'Price': price, 'Invested': invested, 
                    'Share %': weight * 100, 'Papers': shares, 'Cash': round(cash_to_invest, 2)}
        investments = pd.concat([investments, pd.DataFrame([row_data])], ignore_index=True)

    return investments[['Ticker', 'Price', 'Papers', 'Share %', 'Invested', 'Cash']]



def logreturns(df):
    df.columns = df.columns.str.split('_').str[0]
    log_returns  = np.log(df)
    log_returns = df.iloc[:, 0:].pct_change()
    fig = px.line(log_returns, x=log_returns.index, y=log_returns.columns[0:],
                  labels={'value': 'log'},
                  title='log returns')
    fig.update_layout(legend_title_text='Assets')
    st.plotly_chart(fig)

    return log_returns

def return_over_time(df):
    return_df = pd.DataFrame()
    df.columns = df.columns.str.split('_').str[0]
    for col in df.columns:
        return_df[col] = df[col] / df[col].iloc[0] -1
        
    fig = px.line(return_df, x=return_df.index, y=return_df.columns[0:],
                  labels={'value': 'Returns to Date'},
                  title='returns')
    fig.update_layout(legend_title_text='Assets')
    st.plotly_chart(fig)
    
    
def efficient_frontier(df, trading_days,weights_df, risk_free_rate, simulations= 1000, resampler='A'):
    return_over_time(df)
    logreturns(df)

    Weights = weights_df['Weights'].values
    cov_matrix = df.pct_change().apply(lambda x: np.log(1+x)).cov()
    portfolio_var = np.dot(Weights.T, np.dot(cov_matrix, Weights))
    
    standard_deviation = np.sqrt(portfolio_var) 
    annualized_risk  = standard_deviation *np.sqrt(trading_days)
    simple_annualized_risk = np.exp(annualized_risk) - 1
    simple_risk = np.exp(annualized_risk) - 1
    
    st.markdown(f'Annualized portfolio risk:  **{simple_risk:.4f}**')
    annualized_returns = df.resample(resampler).last()    
    annualized_returns = df.pct_change().apply(lambda x: np.log(1 + x)).mean() * trading_days
    annualized_returns = annualized_returns.rename('Log Returns')
    simple_returns = np.exp(annualized_returns) - 1
    simple_returns = simple_returns.rename('Simple Returns')
    weights_series = pd.Series(Weights, index=df.columns, name='Weights')
  
    
    returns_df = pd.concat([annualized_returns, simple_returns, weights_series], axis=1)
    returns_df['Weighted Returns'] = returns_df['Simple Returns'] * returns_df['Weights']
    returns_df['Cumulative Portfolio Returns'] = returns_df['Weighted Returns'].cumsum()

    st.markdown(f'**Annualized portfolio return:**')
    st.dataframe(returns_df)
    assets_return = simple_returns * Weights
    portfolio_return = assets_return.sum()
    st.markdown(f'Annualized portfolio return: **{portfolio_return:.4f}**')
    sharpe_ratio = (portfolio_return - risk_free_rate)/ simple_annualized_risk
    st.markdown(f'Initial allocation Sharpe Ratio  **{sharpe_ratio:.4f}**')

    portfolio_returns = [] 
    portfolio_variance = [] 
    portfolio_weights = [] 
    
    num_assets = len(df.columns)    

    for _ in range(simulations):
        weights = np.random.random(num_assets)
        weights = weights/np.sum(weights)
        portfolio_weights.append(weights)
        returns = np.dot(weights, simple_returns) # Returns are the product of individual expected returns of asset and its 
        portfolio_returns.append(returns)
        var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()# Portfolio Variance
        sd = np.sqrt(var) # Daily standard deviation
        ann_sd = sd*np.sqrt(trading_days) # Annual standard deviation = volatility
        portfolio_variance.append(ann_sd)

        data = {'Returns':portfolio_returns, 'Volatility':portfolio_variance}

    for counter, symbol in enumerate(df.columns.tolist()):
        data[symbol+' weight'] = [w[counter] for w in portfolio_weights]
    
    simulated_portfolios  = pd.DataFrame(data)
    simulated_portfolios = simulated_portfolios.round(6)

    return simulated_portfolios

def plot_efficient_frontier(simulated_portfolios, risk_free_rate, expected_sharpe, expected_return, risk_taken):
    simulated_portfolios = simulated_portfolios.sort_values(by='Volatility')
    simulated_portfolios['Sharpe_ratio'] = (simulated_portfolios['Returns'] - risk_free_rate) / simulated_portfolios['Volatility']
    
    simulated_portfolios['Weights'] = simulated_portfolios.iloc[:, 2:-1].apply(
        lambda row: ', '.join([f"{asset}: {weight:.4f}" for asset, weight in zip(simulated_portfolios.columns[2:-1], row)]),
        axis=1
    )

    frontier = px.scatter(simulated_portfolios, x='Volatility', y='Returns', width=800, height=600, 
                          title="Markowitz's Efficient Frontier", labels={'Volatility': 'Volatility', 'Returns': 'Return'},
                          hover_name='Weights')
    
    max_sharpe_ratio_portfolio = simulated_portfolios.loc[simulated_portfolios['Sharpe_ratio'].idxmax()]
    frontier.add_trace(go.Scatter(x=[max_sharpe_ratio_portfolio['Volatility']], 
                                  y=[max_sharpe_ratio_portfolio['Returns']],
                                  mode='markers',
                                  marker=dict(color='green', size=10),
                                  name='Max Sharpe Ratio',
                                  text=max_sharpe_ratio_portfolio['Weights']))

    low_risk_portfolios = simulated_portfolios[
        (simulated_portfolios['Returns'] >= expected_return) & 
        (simulated_portfolios['Volatility'] <= risk_taken)
    ]
    
    frontier.add_trace(go.Scatter(x=low_risk_portfolios['Volatility'], 
                                  y=low_risk_portfolios['Returns'],
                                  mode='markers',
                                  marker=dict(color='purple', size=5),
                                  name='Expected Return & Risk Taken',
                                  text=low_risk_portfolios['Weights']))
    
    expected_portfolio = simulated_portfolios[
        (simulated_portfolios['Sharpe_ratio'] >= expected_sharpe - 0.001) & 
        (simulated_portfolios['Sharpe_ratio'] <= expected_sharpe + 0.001)
    ]
    
    if not expected_portfolio.empty:
        frontier.add_trace(go.Scatter(x=[expected_portfolio['Volatility'].values[0]], 
                                      y=[expected_portfolio['Returns'].values[0]],
                                      mode='markers',
                                      marker=dict(color='orange', size=10),
                                      name='Expected Sharpe Ratio',
                                      text=expected_portfolio['Weights']))
    
    frontier.add_trace(go.Scatter(x=[simulated_portfolios.iloc[0]['Volatility']], 
                                  y=[simulated_portfolios.iloc[0]['Returns']],
                                  mode='markers',
                                  marker=dict(color='red', size=10),
                                  name='Min Volatility', 
                                  text=simulated_portfolios.iloc[0]['Weights']))
    
    frontier.update_layout(legend=dict(
                           orientation='h',
                           yanchor='top',
                           y=1.1,
                           xanchor='center',
                           x=0.5))
    
    max_sharpe_ratio_value = simulated_portfolios['Sharpe_ratio'].max()
    st.markdown(f'Max Sharpe Ratio: **{max_sharpe_ratio_value:.2f}**')
    st.write(f'Max_sharpe:  {max_sharpe_ratio_value}')
    max_sharpe_weights = max_sharpe_ratio_portfolio["Weights"]
    st.write("### Max Sharpe Ratio Weights")
    for item in max_sharpe_weights.split(","):
        st.write(item.strip())
    st.plotly_chart(frontier)
    
    
def backtest_frontier(df_list, risk_free_rate, trading_days, simulations=1000):
    result_dfs = []
    for df in df_list:
        cov_matrix = df.drop(columns=['ID', 'date']).pct_change().apply(lambda x: np.log(1 + x)).cov() * trading_days
        portfolio_returns = [] 
        portfolio_variance = [] 
        portfolio_weights = [] 
        portfolio_sharpe_ratio =[]
        num_assets = len(df.columns)-2  # Subtracting 'ID' and 'date' columns

        annualized_returns = df.drop(columns=['ID', 'date']).pct_change().apply(lambda x: np.log(1 + x)).mean() * trading_days
        for _ in range(simulations):
            weights = np.random.random(num_assets)
            weights = weights/np.sum(weights)
            returns = np.dot(weights, annualized_returns)
            portfolio_weights.append(weights)
            portfolio_returns.append(returns)
            var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()  
            sd = np.sqrt(var) 
            ann_sd = sd * np.sqrt(trading_days)  
            portfolio_variance.append(ann_sd)
            if returns > 0:
                sharpe_ratio = (returns - np.log(1 + risk_free_rate)) / ann_sd
            else:
                sharpe_ratio = (returns + np.log(1 + risk_free_rate)) / ann_sd

            portfolio_sharpe_ratio.append(sharpe_ratio)

        data = {
            'Log Returns': portfolio_returns,
            'Volatility': portfolio_variance,
            'Sharpe_ratio': portfolio_sharpe_ratio,
            'ID': df['ID'].iloc[0],
            'Date': df['date'].iloc[0]  # Insert starting day for each ID
        }

        for counter, symbol in enumerate(df.columns):
                if symbol == 'ID' or symbol == 'date':
                    continue
                ticker_name = symbol.split('_')[0]  # Remove underscores from ticker name
                data[ticker_name + '_Weight'] = [w[counter] for w in portfolio_weights]
            
        simulated_portfolios = pd.DataFrame(data)
        simulated_portfolios = simulated_portfolios.round(4)
        result_dfs.append(simulated_portfolios)
        
    final_df = pd.concat(result_dfs, ignore_index=True)

    return final_df

def get_max_sharpe_per_id(final_df, max_values, min_values):
    weight_columns = [column for column in final_df.columns if column.endswith('_Weight')]
    num_columns = len(weight_columns)

    filtered_indices = set(range(len(final_df)))

    for i in range(num_columns):
        weight_column = weight_columns[i]
        min_value = min_values[i]
        max_value = max_values[i]

        filtered_indices_temp = set(index for index, row in final_df.iterrows() if min_value <= row[weight_column] <= max_value)
        filtered_indices.intersection_update(filtered_indices_temp)

    filtered_df = final_df.iloc[list(filtered_indices)]
    max_sharpe_rows = filtered_df.loc[filtered_df.groupby('ID')['Sharpe_ratio'].idxmax()]
    
    return max_sharpe_rows

def optimizeBySharpe(df):
    pd.options.display.float_format = '{:.3f}'.format
    pricesT2 = df['pricesT2'].values
    qtT1 = df['qtT1'].values
    weightsT2 = df['weightsT2'].values
    pricesT1 = df['pricesT1'].values
    investedT1 = pricesT1 * qtT1
    totalInvestedT1 = np.sum(investedT1)
    fund = qtT1 * pricesT2
    funds = np.sum(fund)
    maxQtT2 = funds * weightsT2 / pricesT2
    maxInvestedT2 = maxQtT2 * pricesT2
    optQtT2 = np.sum(maxInvestedT2) * weightsT2 /  pricesT2
    optWeightsT2 = optQtT2 * pricesT2 / np.sum(maxInvestedT2)
    otimizationROI = (funds /totalInvestedT1)
    portfolioROI = funds / invested_cash
    df['optWeightsT2'] = optWeightsT2
    df['optQtT2'] = optQtT2
    df['qtBuyOrSell'] = abs(optQtT2 - qtT1)
    df['investedT1'] = investedT1
    df['investedT2'] = fund
    
    df = df[['pricesT1', 'pricesT2', 'investedT1', 'investedT2',
             'qtT1', 'optQtT2', 'qtBuyOrSell', 'weightsT1', 
             'optWeightsT2', 'weightsT2']]
    
    st.markdown(f'**Total Invested T1:** {totalInvestedT1:.3f}')
    st.markdown(f'**Available Cash T2:** {funds:.3f}')
    st.markdown(f'**Otimization ROI:** {otimizationROI:.3f}')
    st.markdown(f'**Portfolio ROI:** {portfolioROI:.3f}')
    st.dataframe(df)
    
    return df

def returns_till_date(df, invested_cash, price_columns, weight_columns):
    first_row = df.iloc[0]
    initial_quantities_per_asset = []
    for price_col, weight_col in zip(price_columns, weight_columns):
        initial_quantity = invested_cash * first_row[weight_col] / first_row[price_col]
        initial_quantities_per_asset.append(initial_quantity)
        
    initial_quantities = np.array([initial_quantities_per_asset])
    initial_values = initial_quantities * df[price_columns].values
    invested_till_date = np.sum(initial_values, axis=1)
    return_till_date = invested_till_date / invested_cash
    ROItilldate = pd.DataFrame({'invested_till_date': invested_till_date, 'return_till_date': return_till_date}, index=df.index)
    st.dataframe(ROItilldate)
    return ROItilldate 

st.set_page_config(page_title='Portfolio Balancer', layout = 'wide', initial_sidebar_state = 'auto')
st.title("Portfolio Balancer")

# Importar SessionState
class SessionState:
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

@st.cache(allow_output_mutation=True)
def get_session():
    return SessionState(df=pd.DataFrame(), data=pd.DataFrame(), portfolio=pd.DataFrame(), backtest =pd.DataFrame(), optimized_data =pd.DataFrame())
session_state = get_session()

st.subheader('Pick your assets',divider='rainbow')

with st.expander("See explanation"):
    st.markdown("""
        1. You can combine assets from currencies, crypto, commodities, Nasdaq, S&P500, B3, and Indexes.
        
        2. Once you have selected one or more dictionaries, you will be able to choose tickers to download.
        
        3. You must press the download button to retrieve the data.
        
        4. It downloads data from Yahoo Finance, yet there may be tickers which return None or fragmented data.
        
        5. Use the resample and rolling average functions to prepare your data for optimization.
        
        6. If you prefer, download data with the first and last index for all tickers. However, this option may reduce the length of your resulting dataframe. It is optional to remove NaNs without any further data filling techniques.
        
        7. Additionally, you have the ability to:
            - Download your data as a CSV or Excel file for further analysis.
            - Upload your own portfolio data for educational purposes, allowing you to experiment with different optimization strategies.
    """)

file  = st.file_uploader("Upload Excel/CSV file", type=["xlsx", "csv"])
if file:
    session_state.data = upload_file(file)

B3_weights = {
    "RRRP3.SA": 0.00388, "ALOS3.SA": 0.00552, "ALPA4.SA": 0.00078, "ABEV3.SA": 0.02473, "ARZZ3.SA": 0.00154,
    "ASAI3.SA": 0.00877, "AZUL4.SA": 0.00154, "B3SA3.SA": 0.02948, "BBSE3.SA": 0.01036, "BBDC3.SA": 0.00859,
    "BBDC4.SA": 0.03338, "BRAP4.SA": 0.0025, "BBAS3.SA": 0.03721, "BRKM5.SA": 0.00285, "BRFS3.SA": 0.01382,
    "BPAC11.SA": 0.01788, "CRFB3.SA": 0.00288, "BHIA3.SA": 0.00027, "CCRO3.SA": 0.0059, "CMIG4.SA": 0.00878,
    "CIEL3.SA": 0.00291, "COGN3.SA": 0.00173, "CPLE6.SA": 0.00734, "CSAN3.SA": 0.00799, "CPFE3.SA": 0.00311,
    "CMIN3.SA": 0.00266, "CVCB3.SA": 0.00052, "CYRE3.SA": 0.00269, "DXCO3.SA": 0.00104, "ELET3.SA": 0.03495,
    "ELET6.SA": 0.00538, "EMBR3.SA": 0.01083, "ENGI11.SA": 0.0072, "ENEV3.SA": 0.00933, "EGIE3.SA": 0.00485,
    "EQTL3.SA": 0.01632, "EZTC3.SA": 0.00059, "FLRY3.SA": 0.00168, "GGBR4.SA": 0.01199, "GOAU4.SA": 0.00338,
    "NTCO3.SA": 0.0067, "SOMA3.SA": 0.00145, "HAPV3.SA": 0.00753, "HYPE3.SA": 0.00564, "IGTI11.SA": 0.00217,
    "IRBR3.SA": 0.00157, "ITSA4.SA": 0.02445, "ITUB4.SA": 0.07289, "JBSS3.SA": 0.01186, "KLBN11.SA": 0.008,
    "RENT3.SA": 0.02034, "LREN3.SA": 0.00702, "LWSA3.SA": 0.00091, "MGLU3.SA": 0.00215, "MRFG3.SA": 0.00151,
    "BEEF3.SA": 0.00075, "MRVE3.SA": 0.00115, "MULT3.SA": 0.00305, "PCAR3.SA": 0.0005, "PETR3.SA": 0.04907,
    "PETR4.SA": 0.08974, "RECV3.SA": 0.00275, "PRIO3.SA": 0.01867, "PETZ3.SA": 0.00076, "RADL3.SA": 0.01519,
    "RAIZ4.SA": 0.00174, "RDOR3.SA": 0.01363, "RAIL3.SA": 0.01219, "SBSP3.SA": 0.01341, "SANB11.SA": 0.00442,
    "SMTO3.SA": 0.00191, "CSNA3.SA": 0.00411, "SLCE3.SA": 0.00176, "SUZB3.SA": 0.01949, "TAEE11.SA": 0.00368,
    "VIVT3.SA": 0.00968, "TIMS3.SA": 0.00658, "TOTS3.SA": 0.00693, "TRPL4.SA": 0.00465, "UGPA3.SA": 0.01342,
    "USIM5.SA": 0.00222, "VALE3.SA": 0.12501, "VAMO3.SA": 0.00146, "VBBR3.SA": 0.01209, "WEGE3.SA": 0.0267,
    "YDUQ3.SA": 0.00195
}

currencies_dict  =  {'USD/JPY': 'USDJPY=X', 'USD/BRL': 'BRL=X', 'USD/ARS': 'ARS=X', 'USD/PYG': 'PYG=X', 'USD/UYU': 'UYU=X',
                     'USD/CNY': 'CNY=X', 'USD/KRW': 'KRW=X', 'USD/MXN': 'MXN=X', 'USD/IDR': 'IDR=X', 'USD/EUR': 'EUR=X',
                     'USD/CAD': 'CAD=X', 'USD/GBP': 'GBP=X', 'USD/CHF': 'CHF=X', 'USD/AUD': 'AUD=X', 'USD/NZD': 'NZD=X',
                     'USD/HKD': 'HKD=X', 'USD/SGD': 'SGD=X', 'USD/INR': 'INR=X', 'USD/RUB': 'RUB=X', 'USD/ZAR': 'ZAR=X',
                     'USD/SEK': 'SEK=X', 'USD/NOK': 'NOK=X', 'USD/TRY': 'TRY=X', 'USD/AED': 'AED=X', 'USD/SAR': 'SAR=X',
                     'USD/THB': 'THB=X', 'USD/DKK': 'DKK=X', 'USD/MYR': 'MYR=X', 'USD/PLN': 'PLN=X', 'USD/EGP': 'EGP=X',
                     'USD/CZK': 'CZK=X', 'USD/ILS': 'ILS=X', 'USD/HUF': 'HUF=X', 'USD/PHP': 'PHP=X', 'USD/CLP': 'CLP=X',
                     'USD/COP': 'COP=X', 'USD/PEN': 'PEN=X', 'USD/KWD': 'KWD=X', 'USD/QAR': 'USD/QAR'
                    }
crypto_dict = {'Bitcoin USD': 'BTC-USD', 'Ethereum USD': 'ETH-USD', 'Tether USDT USD': 'USDT-USD',
               'Bnb USD': 'BNB-USD', 'Solana USD': 'SOL-USD', 'Xrp USD': 'XRP-USD', 'Usd Coin USD': 'USDC-USD',
               'Lido Staked Eth USD': 'STETH-USD', 'Cardano USD': 'ADA-USD', 'Avalanche USD': 'AVAX-USD',
               'Dogecoin USD': 'DOGE-USD', 'Wrapped Tron USD': 'WTRX-USD', 'Tron USD': 'TRX-USD',
               'Polkadot USD': 'DOT-USD', 'Chainlink USD': 'LINK-USD', 'Toncoin USD': 'TON11419-USD',
               'Polygon USD': 'MATIC-USD', 'Wrapped Bitcoin USD': 'WBTC-USD', 'Shiba Inu USD': 'SHIB-USD',
               'Internet Computer USD': 'ICP-USD', 'Dai USD': 'DAI-USD', 'Litecoin USD': 'LTC-USD',
               'Bitcoin Cash USD': 'BCH-USD', 'Uniswap USD': 'UNI7083-USD', 'Cosmos USD': 'ATOM-USD',
               'Unus Sed Leo USD': 'LEO-USD', 'Ethereum Classic USD': 'ETC-USD', 'Stellar USD': 'XLM-USD',
               'Okb USD': 'OKB-USD', 'Near Protocol USD': 'NEAR-USD', 'Optimism USD': 'OP-USD',
               'Injective USD': 'INJ-USD', 'Aptos USD': 'APT21794-USD', 'Monero USD': 'XMR-USD',
               'Filecoin USD': 'FIL-USD', 'Lido Dao USD': 'LDO-USD', 'Celestia USD': 'TIA22861-USD',
               'Hedera USD': 'HBAR-USD', 'Wrapped Hbar USD': 'WHBAR-USD', 'Immutable USD': 'IMX10603-USD',
               'Wrapped Eos USD': 'WEOS-USD', 'Arbitrum USD': 'ARB11841-USD', 'Kaspa USD': 'KAS-USD',
               'Bitcoin Bep2 USD': 'BTCB-USD', 'Stacks USD': 'STX4847-USD', 'Mantle USD': 'MNT27075-USD',
               'First Digital Usd Usd': 'FDUSD-USD', 'Vechain USD': 'VET-USD', 'Cronos USD': 'CRO-USD',
               'Wrapped Beacon Eth USD': 'WBETH-USD', 'Trueusd USD': 'TUSD-USD', 'Sei USD': 'SEI-USD',
               'Maker USD': 'MKR-USD', 'Hex USD': 'HEX-USD', 'Rocket Pool Eth USD': 'RETH-USD',
               'Bitcoin Sv USD': 'BSV-USD', 'Render USD': 'RNDR-USD', 'Bittensor USD': 'TAO22974-USD',
               'The Graph USD': 'GRT6719-USD', 'Algorand USD': 'ALGO-USD', 'Ordi USD': 'ORDI-USD',
               'Aave USD': 'AAVE-USD', 'Thorchain USD': 'RUNE-USD', 'Quant USD': 'QNT-USD',
               'Multiversx USD': 'EGLD-USD', 'Sui USD': 'SUI20947-USD', 'Mina USD': 'MINA-USD',
               'Sats USD': '1000SATS-USD', 'Flow USD': 'FLOW-USD', 'Helium USD': 'HNT-USD',
               'Fantom USD': 'FTM-USD', 'Synthetix USD': 'SNX-USD', 'The Sandbox USD': 'SAND-USD',
               'Theta Network USD': 'THETA-USD', 'Axie Infinity USD': 'AXS-USD', 'Tezos USD': 'XTZ-USD',
               'Beam USD': 'BEAM28298-USD', 'Bittorrent(New) USD': 'BTT-USD', 'Kucoin Token USD': 'KCS-USD',
               'Dydx (Ethdydx) USD': 'ETHDYDX-USD', 'Ftx Token USD': 'FTT-USD', 'Astar USD': 'ASTR-USD',
               'Wemix USD': 'WEMIX-USD', 'Blur USD': 'BLUR-USD', 'Cheelee USD': 'CHEEL-USD',
               'Chiliz USD': 'CHZ-USD', 'Bitget Token USD': 'BGB-USD', 'Decentraland USD': 'MANA-USD',
               'Neo USD': 'NEO-USD', 'Osmosis USD': 'OSMO-USD', 'Eos USD': 'EOS-USD', 'Bonk USD': 'BONK-USD',
               'Kava USD': 'KAVA-USD', 'Woo USD': 'WOO-USD', 'Klaytn USD': 'KLAY-USD', 'Flare USD': 'FLR-USD',
               'Oasis Network USD': 'ROSE-USD', 'Iota USD': 'IOTA-USD', 'Usdd USD': 'USDD-USD',
               'Terra Classic USD': 'LUNC-USD'}

commodities_dict = { "BRENT CRUDE OIL LAST DAY FINANC": "BZ=F", "COCOA": "CC=F", "COFFEE": "KC=F", "COPPER": "HG=F",
                    "CORN FUTURES": "ZC=F", "COTTON": "CT=F", "HEATING OIL": "HO=F", "KC HRW WHEAT FUTURES": "KE=F",
                    "LEAN HOGS FUTURES": "HE=F", "LIVE CATTLE FUTURES": "LE=F", "MONT BELVIEU LDH PROPANE (OPIS)": "B0=F",
                    "NATURAL GAS": "NG=F", "ORANGE JUICE": "OJ=F", "GOLD": "GC=F", "OAT FUTURES": "ZO=F",
                    "PALLADIUM": "PA=F", "CRUDE OIL": "CL=F", "PLATINUM": "PL=F", "RBOB GASOLINE": "RB=F",
                    "RANDOM LENGTH LUMBER FUTURES": "LBS=F", "ROUGH RICE FUTURES": "ZR=F", "SILVER": "SI=F",
                    "SOYBEAN FUTURES": "ZS=F", "SOYBEAN OIL FUTURES": "ZL=F", "S&P COMPOSITE 1500 ESG TILTED I": "ZM=F",
                    "SUGAR": "SB=F", "WISDOMTREE INTERNATIONAL HIGH D": "GF=F"
                }

b3_stocks = {
    "3m": "MMMC34.SA", "Aes brasil": "AESB3.SA", "Af invest": "AFHI11.SA", "Afluente t": "AFLT3.SA", "Agribrasil": "GRAO3.SA",
    "Agogalaxy": "AGXY3.SA", "Alliar": "AALR3.SA", "Alper": "APER3.SA", "Google": "GOGL35.SA", "Alupar investimento": "ALUP4.SA",
    "American express": "AXPB34.SA", "Arcelor": "ARMT34.SA", "Att inc": "ATTB34.SA", "Auren energia": "AURE3.SA", "Banco do brasil": "BBAS3.SA",
    "Banco mercantil de investimentos": "BMIN3.SA", "Banco pan": "BPAN4.SA", "Bank america": "BOAC34.SA", "Banrisul": "BRSR3.SA",
    "Baumer": "BALM3.SA", "Bb seguridade": "BBSE3.SA", "Biomm": "BIOM3.SA", "Bmg": "BMGB4.SA", "Caixa agências": "CXAG11.SA",
    "Camden prop": "C2PT34.SA", "Camil": "CAML3.SA", "Carrefour": "CRFB3.SA", "Cartesia fiici": "CACR11.SA", "Casan": "CASN4.SA",
    "Ceb": "CEBR6.SA", "Ceee-d": "CEED4.SA", "Ceg": "CEGR3.SA", "Celesc": "CLSC4.SA", "Cemig": "CMIG4.SA", "Chevron": "CHVX34.SA",
    "Churchill dw": "C2HD34.SA", "Cisco": "CSCO34.SA", "Citigroup": "CTGP34.SA", "Clearsale": "CLSA3.SA", "Coca-cola": "COCA34.SA",
    "Coelce": "COCE6.SA", "Coinbase glob": "C2OI34.SA", "Colgate": "COLG34.SA", "Comgás": "CGAS3.SA", "Conocophillips": "COPH34.SA",
    "Copel": "CPLE6.SA", "Cpfl energia": "CPFE3.SA", "Csn": "CSNA3.SA", "Dexco": "DXCO3.SA", "Dexxos part": "DEXP3.SA",
    "Dimed": "PNVL3.SA", "Doordash inc": "D2AS34.SA", "Draftkings": "D2KN34.SA", "Ebay": "EBAY34.SA", "Enauta part": "ENAT3.SA",
    "Energisa mt": "ENMT3.SA", "Engie brasil": "EGIE3.SA", "Eqi receci": "EQIR11.SA", "Eucatex": "EUCA4.SA", "Exxon mobil": "EXXO34.SA",
    "Ferbasa": "FESA4.SA", "Fiagro jgp ci": "JGPX11.SA", "Fiagro riza ci": "RZAG11.SA", "Fii brio me ci": "BIME11.SA", "Fii cyrela ci es": "CYCR11.SA",
    "Fii gtis lg": "GTLG11.SA", "Fii husi ci es": "HUSI11.SA", "Fii js a finci": "JSAF11.SA", "Fii more crici er": "MORC11.SA", "Fii rooftopici": "ROOF11.SA",
    "Fleury": "FLRY3.SA", "Freeport": "FCXO34.SA", "Ft cloud cpt": "BKYY39.SA", "Ft dj intern": "BFDN39.SA", "Ft nasd cyber": "BCIR39.SA",
    "Ft nasd100 eq": "BQQW39.SA", "Ft nasd100 tc": "BQTC39.SA", "Ft nat gas": "BFCG39.SA", "Ft nyse biot drn": "BFBI39.SA", "Ft risi divid": "BFDA39.SA",
    "G2d investments": "G2DI33.SA", "Ge": "GEOO34.SA", "General shopping": "GSHP3.SA", "Gerd paranapanema": "GEPA4.SA", "Golias": "GOAU4.SA",
    "Godaddy inc": "G2DD34.SA", "Goldman sachs": "GSGI34.SA", "Grd": "IGBR3.SA", "Halliburton": "HALI34.SA", "Honeywell": "HONB34.SA",
    "Hp company": "HPQB34.SA", "Hypera pharma": "HYPE3.SA", "Ibm": "IBMB34.SA", "Iguatemi s.a.": "IGTI3.SA", "Infracommerce": "IFCM3.SA",
    "Intel": "ITLC34.SA", "Investo alug": "ALUG11.SA", "Investo ustk": "USTK11.SA", "Investo wrld": "WRLD11.SA", "Irb brasil re": "IRBR3.SA",
    "Isa cteep": "TRPL4.SA", "Itaú unibanco": "ITUB4.SA", "Itaúsa": "ITSA4.SA", "Jbs": "JBSS3.SA", "Johnson": "JNJB34.SA",
    "Jpmorgan": "JPMC34.SA", "Kingsoft chl": "K2CG34.SA", "Klabin s/a": "KLBN11.SA", "Livetech": "LVTC3.SA", "Locaweb": "LWSA3.SA",
    "Log": "LOGG3.SA", "Lps brasil": "LPSB3.SA", "Marfrig": "MRFG3.SA", "Mastercard": "MSCD34.SA", "Mdiasbranco": "MDIA3.SA",
    "Melnick": "MELK3.SA", "Meliuz": "CASH3.SA", "Mercado livre": "MELI34.SA", "Microsoft": "MSFT34.SA", "Mrv engenharia": "MRVE3.SA",
    "Natura": "NTCO3.SA", "Netflix": "NFLX34.SA", "Oi": "OIBR3.SA", "Oracle": "ORCL34.SA", "Pão de açúcar": "PCAR3.SA",
    "Petrobras": "PETR4.SA", "Petróleo": "PEAB3.SA", "Pfizer": "PFIZ34.SA", "Plascar": "PLAS3.SA", "Porto seguro": "PSSA3.SA",
    "Positivo": "POSI3.SA", "Procter": "PGCO34.SA", "Qualicorp": "QUAL3.SA", "Randon": "RAPT4.SA", "Raia drogasil": "RADL3.SA",
    "Renner": "LREN3.SA", "Rossi": "RSID3.SA", "Rumo s.a.": "RAIL3.SA", "Santander": "SANB11.SA", "Telefônica": "VIVT3.SA",
    "Tim": "TIMS3.SA", "Totvs": "TOTS3.SA", "Trisul": "TRIS3.SA", "Ultrapar": "UGPA3.SA", "Unipar": "UNIP6.SA", "Usiminas": "USIM5.SA",
    "Vale": "VALE3.SA", "Vivara": "VIVA3.SA", "Vulcabras": "VULC3.SA", "Weg": "WEGE3.SA", "Whirlpool": "WHRL3.SA", "Yduqs": "YDUQ3.SA"
}


idx_dict =     {
                    'S&P GSCI': 'GD=F', 'IBOVESPA': '^BVSP', 'S&P/CLX IPSA': '^IPSA',
                    'MERVAL': '^MERV', 'IPC MEXICO': '^MXX', 'S&P 500': '^GSPC',
                    'Dow Jones Industrial Average': '^DJI', 'NASDAQ Composite': '^IXIC',
                    'NYSE COMPOSITE (DJ)': '^NYA', 'NYSE AMEX COMPOSITE INDEX': '^XAX',
                    'Russell 2000': '^RUT', 'CBOE Volatility Index': '^VIX',
                    'S&P/TSX Composite index': '^GSPTSE', 'FTSE 100': '^FTSE',
                    'DAX PERFORMANCE-INDEX': '^GDAXI', 'CAC 40': '^FCHI',
                    'ESTX 50 PR.EUR': '^STOXX50E', 'Euronext 100 Index': '^N100',
                    'BEL 20': '^BFX', 'MOEX Russia Index': 'IMOEX.ME', 'Nikkei 225': '^N225',
                    'HANG SENG INDEX': '^HSI', 'SSE Composite Index': '000001.SS',
                    'Shenzhen Index': '399001.SZ', 'STI Index': '^STI', 'S&P/ASX 200': '^AXJO',
                    'ALL ORDINARIES': '^AORD', 'S&P BSE SENSEX': '^BSESN', 'IDX COMPOSITE': '^JKSE',
                    'FTSE Bursa Malaysia KLCI': '^KLSE', 'S&P/NZX 50 INDEX GROSS': '^NZ50',
                    'KOSPI Composite Index': '^KS11', 'TSEC weighted index': '^TWII',
                    'TA-125': '^TA125.TA', 'Top 40 USD Net TRI Index': '^JN0U.JO', 'NIFTY 50': '^NSEI'
                    }

sp500_dict = {'SPY': 'SPY',  "iShares MSCI USA Momentum Factor": 'MTUM',  "iShares MSCI USA Quality Factor": 'QUAL',  "iShares MSCI USA Value Factor": 'VLUE',
            "iShares MSCI USA Size Factor": 'SIZE',  "iShares MSCI Min Vol USA": 'USMV', "iShares MSCI World": 'URTH',  "iShares MSCI ACWI": 'ACWI',
             "Consumer Discretionary Select": 'XLY',
 "Consumer Staples Select": 'XLP',
 "Energy Select": 'XLE',
 "Financial Select": 'XLF',
 "Health Care Select": 'XLV',
 "Technology Select": 'XLK',
 "Industrial Select": 'XLI',
 "Materials Select": 'XLB',
 "Utilities Select": 'XLU',
 "Real Estate Select": 'XLRE',
 "Communication Services Select": 'XLC',
 "iShares MSCI China": 'MCHI',
 "iShares China Large-Cap": 'FXI',
 "KraneShares CSI China Internet": 'KWEB',
 "Xtrackers Harvest CSI 300": 'ASHR',
 "Vanguard FTSE EM": 'VWO',
 "iShares MSCI EM": 'EEM',
 "iShares Core MSCI EM": 'IEMG',
 "iShares iBoxx $ Investment Grade Corporate Bond": 'LQD',
 "iShares iBoxx $ High Yield Corporate Bond": 'HYG',
 "iShares J.P. Morgan USD EM Bond": 'EMB',
 "iShares TIPS Bond": 'TIP',
 "SPDR Bloomberg 1-3 Month T-Bill": 'BIL',
 "iShares MSCI United Kingdom": 'EWU',
 "Vanguard FTSE Europe": 'VGK',
 "iShares MSCI Eurozone": 'EZU',
 "iShares MSCI Japan": 'EWJ',
 "WisdomTree Japan Hedged Equity": 'DXJ',
 "iShares Semiconductor": 'SOXX',
 "VanEck Semiconductor": 'SMH',
 "iShares Global Tech": 'IXN',
 "VanEck Video Gaming & eSports": 'ESPO',
 "Vanguard Real Estate": 'VNQ',
 "iShares Global REIT": 'REET',
 "SPDR Gold Shares": 'GLD',
 "Invesco DB Commodity Index Tracking": 'DBC',
 "Invesco Optimum Yield Diversified Commodity Strategy": 'PDBC'}

sp500_dict2 = {'3M': 'MMM', 'A. O. Smith': 'AOS', 'Abbott': 'ABT', 'AbbVie': 'ABBV', 'Accenture': 'ACN', 'Adobe Inc.': 'ADBE',
              'Advanced Micro Devices': 'AMD', 'AES Corporation': 'AES', 'Aflac': 'AFL', 'Agilent Technologies': 'A', 'Air Products and Chemicals': 'APD',
              'Airbnb': 'ABNB', 'Akamai': 'AKAM', 'Albemarle Corporation': 'ALB', 'Alexandria Real Estate Equities': 'ARE', 'Align Technology': 'ALGN',
              'Allegion': 'ALLE', 'Alliant Energy': 'LNT', 'Allstate': 'ALL', 'Google': 'GOOGL', 'Google': 'GOOG',
              'Altria': 'MO', 'Amazon': 'AMZN', 'Amcor': 'AMCR', 'Ameren': 'AEE', 'American Airlines Group': 'AAL', 'American Electric Power': 'AEP',
              'American Express': 'AXP', 'American International Group': 'AIG', 'American Tower': 'AMT', 'American Water Works': 'AWK', 'Ameriprise Financial': 'AMP',
              'AMETEK': 'AME', 'Amgen': 'AMGN', 'Amphenol': 'APH', 'Analog Devices': 'ADI', 'ANSYS': 'ANSS', 'Aon': 'AON',
              'APA Corporation': 'APA', 'Apple Inc.': 'AAPL', 'Applied Materials': 'AMAT', 'Aptiv': 'APTV', 'Arch Capital Group': 'ACGL', 'Archer-Daniels-Midland': 'ADM',
              'Arista Networks': 'ANET', 'Arthur J. Gallagher & Co.': 'AJG', 'Assurant': 'AIZ', 'AT&T': 'T', 'Atmos Energy': 'ATO', 'Autodesk': 'ADSK',
              'Automated Data Processing': 'ADP', 'AutoZone': 'AZO', 'Avalonbay Communities': 'AVB', 'Avery Dennison': 'AVY', 'Axon Enterprise': 'AXON', 'Baker Hughes': 'BKR',
              'Ball Corporation': 'BALL', 'Bank of America': 'BAC', 'Bank of New York Mellon': 'BK', 'Bath & Body Works, Inc.': 'BBWI', 'Baxter International': 'BAX', 'Becton Dickinson': 'BDX',
              'Berkshire Hathaway': 'BRK.B', 'Best Buy': 'BBY', 'Bio-Rad': 'BIO', 'Bio-Techne': 'TECH', 'Biogen': 'BIIB', 'BlackRock': 'BLK', 'Blackstone': 'BX',
              'Boeing': 'BA', 'Booking Holdings': 'BKNG', 'BorgWarner': 'BWA', 'Boston Properties': 'BXP', 'Boston Scientific': 'BSX', 'Bristol Myers Squibb': 'BMY', 'Broadcom Inc.': 'AVGO',
              'Broadridge Financial Solutions': 'BR', 'Brown & Brown': 'BRO', 'Brown–Forman': 'BF.B', 'Builders FirstSource': 'BLDR', 'Bunge Global SA': 'BG', 'Cadence Design Systems': 'CDNS',
              'Caesars Entertainment': 'CZR', 'Camden Property Trust': 'CPT', 'Campbell Soup Company': 'CPB', 'Capital One': 'COF', 'Cardinal Health': 'CAH', 'CarMax': 'KMX',
              'Carnival': 'CCL', 'Carrier Global': 'CARR', 'Catalent': 'CTLT', 'Caterpillar Inc.': 'CAT', 'Cboe Global Markets': 'CBOE', 'CBRE Group': 'CBRE', 'CDW': 'CDW',
              'Celanese': 'CE', 'Cencora': 'COR', 'Centene Corporation': 'CNC', 'CenterPoint Energy': 'CNP', 'Ceridian': 'CDAY', 'CF Industries': 'CF', 'CH Robinson': 'CHRW',
              'Charles River Laboratories': 'CRL', 'Charles Schwab Corporation': 'SCHW', 'Charter Communications': 'CHTR', 'Chevron Corporation': 'CVX', 'Chipotle Mexican Grill': 'CMG',
              'Chubb Limited': 'CB', 'Church & Dwight': 'CHD', 'Cigna': 'CI', 'Cincinnati Financial': 'CINF', 'Cintas': 'CTAS', 'Cisco': 'CSCO', 'Citigroup': 'C',
              'Citizens Financial Group': 'CFG', 'Clorox': 'CLX', 'CME Group': 'CME', 'CMS Energy': 'CMS', 'Coca-Cola Company (The)': 'KO', 'Cognizant': 'CTSH', 'Colgate-Palmolive': 'CL',
              'Comcast': 'CMCSA', 'Comerica': 'CMA', 'Conagra Brands': 'CAG', 'ConocoPhillips': 'COP', 'Consolidated Edison': 'ED', 'Constellation Brands': 'STZ', 'Constellation Energy': 'CEG',
              'CooperCompanies': 'COO', 'Copart': 'CPRT', 'Corning Inc.': 'GLW', 'Corteva': 'CTVA', 'CoStar Group': 'CSGP', 'Costco': 'COST', 'Coterra': 'CTRA', 'Crown Castle': 'CCI',
              'CSX': 'CSX', 'Cummins': 'CMI', 'CVS Health': 'CVS', 'Danaher Corporation': 'DHR', 'Darden Restaurants': 'DRI', 'DaVita Inc.': 'DVA', 'John Deere': 'DE', 'Delta Air Lines': 'DAL',
              'Dentsply Sirona': 'XRAY', 'Devon Energy': 'DVN', 'Dexcom': 'DXCM', 'Diamondback Energy': 'FANG', 'Digital Realty': 'DLR', 'Discover Financial': 'DFS', 'Dollar General': 'DG',
              'Dollar Tree': 'DLTR', 'Dominion Energy': 'D', 'Domino\'s': 'DPZ', 'Dover Corporation': 'DOV', 'Dow Inc.': 'DOW', 'DR Horton': 'DHI', 'DTE Energy': 'DTE', 'Duke Energy': 'DUK',
              'Dupont': 'DD', 'Eastman Chemical Company': 'EMN', 'Eaton Corporation': 'ETN', 'eBay': 'EBAY', 'Ecolab': 'ECL', 'Edison International': 'EIX', 'Edwards Lifesciences': 'EW',
              'Electronic Arts': 'EA', 'Elevance Health': 'ELV', 'Eli Lilly and Company': 'LLY', 'Emerson Electric': 'EMR', 'Enphase': 'ENPH', 'Entergy': 'ETR', 'EOG Resources': 'EOG',
              'EPAM Systems': 'EPAM', 'EQT': 'EQT', 'Equifax': 'EFX', 'Equinix': 'EQIX', 'Equity Residential': 'EQR', 'Essex Property Trust': 'ESS', 'Estée Lauder Companies (The)': 'EL',
              'Etsy': 'ETSY', 'Everest Re': 'EG', 'Evergy': 'EVRG', 'Eversource': 'ES', 'Exelon': 'EXC', 'Expedia Group': 'EXPE', 'Expeditors International': 'EXPD', 'Extra Space Storage': 'EXR',
              'ExxonMobil': 'XOM', 'F5, Inc.': 'FFIV', 'FactSet': 'FDS', 'Fair Isaac': 'FICO', 'Fastenal': 'FAST', 'Federal Realty': 'FRT', 'FedEx': 'FDX', 'Fidelity National Information Services': 'FIS',
              'Fifth Third Bank': 'FITB', 'First Solar': 'FSLR', 'FirstEnergy': 'FE', 'Fiserv': 'FI', 'FleetCor': 'FLT', 'FMC Corporation': 'FMC', 'Ford Motor Company': 'F', 'Fortinet': 'FTNT',
              'Fortive': 'FTV', 'Fox Corporation (Class A)': 'FOXA', 'Fox Corporation (Class B)': 'FOX', 'Franklin Templeton': 'BEN', 'Freeport-McMoRan': 'FCX', 'Garmin': 'GRMN', 'Gartner': 'IT',
              'GE Healthcare': 'GEHC', 'Gen Digital': 'GEN', 'Generac': 'GNRC', 'General Dynamics': 'GD', 'General Electric': 'GE', 'General Mills': 'GIS', 'General Motors': 'GM', 'Genuine Parts Company': 'GPC',
              'Gilead Sciences': 'GILD', 'Global Payments': 'GPN', 'Globe Life': 'GL', 'Goldman Sachs': 'GS', 'Halliburton': 'HAL', 'Hartford (The)': 'HIG', 'Hasbro': 'HAS', 'HCA Healthcare': 'HCA',
              'Healthpeak': 'PEAK', 'Henry Schein': 'HSIC', 'Hershey\'s': 'HSY', 'Hess Corporation': 'HES', 'Hewlett Packard Enterprise': 'HPE', 'Hilton Worldwide': 'HLT', 'Hologic': 'HOLX',
              'Home Depot (The)': 'HD', 'Honeywell': 'HON', 'Hormel Foods': 'HRL', 'Host Hotels & Resorts': 'HST', 'Howmet Aerospace': 'HWM', 'HP Inc.': 'HPQ', 'Hubbell Incorporated': 'HUBB',
              'Humana': 'HUM', 'Huntington Bancshares': 'HBAN', 'Huntington Ingalls Industries': 'HII', 'IBM': 'IBM', 'IDEX Corporation': 'IEX', 'IDEXX Laboratories': 'IDXX',
              'Illinois Tool Works': 'ITW', 'Illumina': 'ILMN', 'Incyte': 'INCY', 'Ingersoll Rand': 'IR', 'Insulet': 'PODD', 'Intel': 'INTC', 'Intercontinental Exchange': 'ICE',
              'International Flavors & Fragrances': 'IFF', 'International Paper': 'IP', 'Interpublic Group of Companies (The)': 'IPG', 'Intuit': 'INTU', 'Intuitive Surgical': 'ISRG',
              'Invesco': 'IVZ', 'Invitation Homes': 'INVH', 'IQVIA': 'IQV', 'Iron Mountain': 'IRM', 'J.B. Hunt': 'JBHT', 'Jabil': 'JBL', 'Jack Henry & Associates': 'JKHY', 'Jacobs Solutions': 'J',
              'Johnson & Johnson': 'JNJ', 'Johnson Controls': 'JCI', 'JPMorgan Chase': 'JPM', 'Juniper Networks': 'JNPR', 'Kellanova': 'K', 'Kenvue': 'KVUE', 'Keurig Dr Pepper': 'KDP',
              'KeyCorp': 'KEY', 'Keysight': 'KEYS', 'Kimberly-Clark': 'KMB', 'Kimco Realty': 'KIM', 'Kinder Morgan': 'KMI', 'KLA Corporation': 'KLAC', 'Kraft Heinz': 'KHC', 'Kroger': 'KR',
              'L3Harris': 'LHX'}

nasdaq_dict = {
    'Adobe Inc.': 'ADBE', 'ADP': 'ADP', 'Airbnb': 'ABNB', 'GOOGLE': 'GOOGL', 'GOOGLE': 'GOOG', 'Amazon': 'AMZN',
    'Advanced Micro Devices Inc.': 'AMD', 'American Electric Power': 'AEP', 'Amgen': 'AMGN', 'Analog Devices': 'ADI', 'Ansys': 'ANSS', 'Apple Inc.': 'AAPL',
    'Applied Materials': 'AMAT', 'ASML Holding': 'ASML', 'AstraZeneca': 'AZN', 'Atlassian': 'TEAM', 'Autodesk': 'ADSK', 'Baker Hughes': 'BKR',
    'Biogen': 'BIIB', 'Booking Holdings': 'BKNG', 'Broadcom Inc.': 'AVGO', 'Cadence Design Systems': 'CDNS', 'CDW Corporation': 'CDW',
    'Charter Communications': 'CHTR', 'Cintas': 'CTAS', 'Cisco': 'CSCO', 'Coca-Cola Europacific Partners': 'CCEP', 'Cognizant': 'CTSH', 'Comcast': 'CMCSA',
    'Constellation Energy': 'CEG', 'Copart': 'CPRT', 'CoStar Group': 'CSGP', 'Costco': 'COST', 'CrowdStrike': 'CRWD', 'CSX Corporation': 'CSX',
    'Datadog': 'DDOG', 'DexCom': 'DXCM', 'Diamondback Energy': 'FANG', 'Dollar Tree': 'DLTR', 'DoorDash': 'DASH', 'Electronic Arts': 'EA',
    'Exelon': 'EXC', 'Fastenal': 'FAST', 'Fortinet': 'FTNT', 'GE HealthCare': 'GEHC', 'Gilead Sciences': 'GILD', 'GlobalFoundries': 'GFS',
    'Honeywell': 'HON', 'Idexx Laboratories': 'IDXX', 'Illumina, Inc.': 'ILMN', 'Intel': 'INTC', 'Intuit': 'INTU', 'Intuitive Surgical': 'ISRG',
    'Keurig Dr Pepper': 'KDP', 'KLA Corporation': 'KLAC', 'Kraft Heinz': 'KHC', 'Lam Research': 'LRCX', 'Lululemon': 'LULU', 'Marriott International': 'MAR',
    'Marvell Technology': 'MRVL', 'MercadoLibre': 'MELI', 'Meta Platforms': 'META', 'Microchip Technology': 'MCHP', 'Micron Technology': 'MU', 'Microsoft': 'MSFT',
    'Moderna': 'MRNA', 'Mondelēz International': 'MDLZ', 'MongoDB Inc.': 'MDB', 'Monster Beverage': 'MNST', 'Netflix': 'NFLX', 'Nvidia': 'NVDA', 'NXP': 'NXPI',
    'O\'Reilly Automotive': 'ORLY', 'Old Dominion Freight Line': 'ODFL', 'Onsemi': 'ON', 'Paccar': 'PCAR', 'Palo Alto Networks': 'PANW', 'Paychex': 'PAYX',
    'PayPal': 'PYPL', 'PDD Holdings': 'PDD', 'PepsiCo': 'PEP', 'Qualcomm': 'QCOM', 'Regeneron': 'REGN', 'Roper Technologies': 'ROP', 'Ross Stores': 'ROST',
    'Sirius XM': 'SIRI', 'Splunk': 'SPLK', 'Starbucks': 'SBUX', 'Synopsys': 'SNPS', 'Take-Two Interactive': 'TTWO', 'T-Mobile US': 'TMUS', 'Tesla, Inc.': 'TSLA',
    'Texas Instruments': 'TXN', 'The Trade Desk': 'TTD', 'Verisk': 'VRSK', 'Vertex Pharmaceuticals': 'VRTX', 'Walgreens Boots Alliance': 'WBA',
    'Warner Bros. Discovery': 'WBD', 'Workday, Inc.': 'WDAY', 'Xcel Energy': 'XEL', 'Zscaler': 'ZS',
}


commodities_dict = {
    "Brent Crude Oil": "BZ=F", "Cocoa": "CC=F", "Coffee": "KC=F", "Copper": "HG=F", 
    "Corn Futures": "ZC=F", "Cotton": "CT=F", "Heating Oil": "HO=F", "KC HRW Wheat Futures": "KE=F", 
    "Lean Hogs Futures": "HE=F", "Live Cattle Futures": "LE=F", "Mont Belvieu LDH Propane (OPIS)": "B0=F", 
    "Natural Gas": "NG=F", "Orange Juice": "OJ=F", "OURO": "GC=F", "Oat Futures": "ZO=F", 
    "Palladium": "PA=F", "PETROLEO CRU": "CL=F", "Platinum": "PL=F", "RBOB Gasoline": "RB=F", 
    "Random Length Lumber Futures": "LBS=F", "Rough Rice Futures": "ZR=F", "Silver": "SI=F", 
    "Soybean Futures": "ZS=F", "Soybean Oil Futures": "ZL=F", "S&P Composite 1500 ESG Tilted I": "ZM=F", 
    "Sugar #11": "SB=F", "WisdomTree International High D": "GF=F"
}

selected_timeframes = st.selectbox('Select The Timeframe:', ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'], index=7)

assets_list = {'commodities':commodities_dict, 
               'b3_stocks': b3_stocks,
               'High Sharpe Ratio ETFs': sp500_dict,
               'SP500': sp500_dict2,
               'NASDAC100':nasdaq_dict,
               'indexes': idx_dict,
               'currencies': currencies_dict, 
               'crypto': crypto_dict}

selected_dict_names = st.multiselect('Select dictionaries to combine', list(assets_list.keys()))
combined_dict = {}
for name in selected_dict_names:
    dictionary = assets_list.get(name)
    if dictionary:
        combined_dict.update(dictionary)

selected_ticker_dict = {}

if selected_dict_names:
    tickers = st.multiselect('Asset Selection', list(combined_dict.keys()))
    if tickers and st.button("Download data"):
        for key in tickers:
            if key in combined_dict:
                selected_ticker_dict[key] = combined_dict[key]
        session_state.data = download_data(selected_ticker_dict, selected_timeframes)

# Handle tickers entered manually
type_tickers = st.text_input('Enter Tickers (comma-separated):')
if type_tickers and st.button("Download data"):
    tickers = [ticker.strip() for ticker in type_tickers.split(',')]
    session_state.data = download_data(tickers, selected_timeframes)
    
frequency = {
        'Daily': 'D',
        'Weekly': 'W',
        'Quaterly': '2W',
        'Monthly': 'M',
        'Bimonthly': '2M',
        'Quarterly': '3M',
        'Four-monthly': '4M',
        'Semiannual': '6M',
        'Annual': 'A'
    }	

# sidebar resampling function
st.sidebar.markdown('**Time Series Resampler**')
freq = st.sidebar.selectbox("Freq to resample:", list(frequency.keys()))
agg = st.sidebar.selectbox("Aggregation:", ['sum', 'mean', 'median', 'valor_exato'])
resample = st.sidebar.button("Resample dataframe")
if resample:
	if session_state.data is not None:
		session_state.data = date_resample(session_state.data,frequency[freq],agg)

# moving average for NaN ocurrencies
st.sidebar.markdown('**Moving Avarage**')
moving_avg_days =  st.sidebar.number_input('Day(s):',1, 100, 3,step=1)                     
method = st.sidebar.selectbox("Method:", ['gap', 'rolling'])

if st.sidebar.button("Apply") and session_state.data is not None:
	session_state.data = fill_moving_avg(session_state.data, moving_avg_days, method)       

st.sidebar.markdown('**Missing Values**')    
remove_nan = st.sidebar.button('Dropna')
if remove_nan:
    session_state.data = session_state.data.dropna(axis = 1)

if session_state.data is not None:
    st.markdown(f'**Total of missing entries:** {session_state.data.isna().sum().sum()}')
    st.dataframe(session_state.data.isna().sum().to_frame().T)
    st.dataframe(session_state.data)
    tickers = [str(col).split("_")[0] for col in session_state.data.columns]
    tickers  = set(tickers)
    tickers_df = pd.DataFrame(tickers, columns=['Tickers'])
    tickers_df['Weights'] = np.nan

st.subheader('Assets allocation', divider='rainbow')
with st.expander("See explanation"):
    st.markdown("""
            <div style="text-align: justify">
            
            1. **Enter Available Cash:** Start by entering the amount of cash you currently have available for investment. This helps us understand how much you have to work with.<br><br>
            
            2. **Enter Invested Cash:** If you already have some cash invested, enter that amount here. This allows us to consider your existing investments prior to optimization.<br><br>
            
            3. **Allocate Your Cash:** Once you have provided your financial details, you can allocate your cash to different assets represented by tickers (like stock symbols). For each asset, you can specify the percentage of your available cash you want to invest. Simply enter the percentage of shares you want to allocate for each asset.<br><br>
            
            4. **Review Your Allocation:** As you allocate your cash, we'll keep track of the total percentage of shares allocated across all assets. If the total reaches 100%, your allocation is complete, and you will be able to run optimization.
          
            </div>
        """, unsafe_allow_html=True)

resampling_options = ['A', 'AS', 'BA', 'BAS', '3M', '4M', '6M', '12M', 
                      'Q', 'BQ', 'QS', 'BQS', 
                      'M', 'BM', 'MS', 'BMS', 
                      'W', 'D'] 

invested_cash = st.number_input("Enter invested cash", min_value=0.0, max_value=1e12, step=1000.0, value=100000.00, format="%.2f")

if session_state.data is not None:    
    weights_df = st.data_editor(tickers_df)
    if not weights_df['Weights'].empty:
        sum_of_weights = weights_df['Weights'].sum()
    else:
        sum_of_weights = 0

    if int(sum_of_weights) == 1:
        session_state.df = compute_investments(session_state.data, weights_df , invested_cash)

    else:
        st.markdown(f'Missing Allocation: {round(1- np.sum(weights_df["Weights"]),3)}')

if session_state.df is not None:
   st.dataframe(session_state.df)
   st.subheader('Optimization', divider='rainbow')
   close_price_data = [col for col  in session_state.data.columns if col.endswith('_Close')]
   session_state.portfolio  = session_state.data[close_price_data]

if session_state.portfolio is not None and not session_state.portfolio.empty:
    trading_days = st.number_input(f'Please Select timeframe for returns', min_value=1, max_value=365, step=1, value = 252)
    resample_list = st.selectbox('Select resampling frequency:', options=resampling_options, index=resampling_options.index('A'))
    risk_free_rate = st.number_input(f'Please Select risk free rate', min_value=0.0, max_value=1.0, step=0.03, value = 0.05)
    risk_taken = st.number_input(f'Please Select anualized risk of investment:', min_value=0.0, max_value=1.0, step=0.02, value = 0.1)
    expected_return = st.number_input(f'Please Select anualized expected returns', min_value=0.0, max_value=1.0, step=0.05, value = 0.15)
    simulations = st.number_input(f'Please Select number of simulations', min_value=100, max_value=100000, step=50, value = 1000)
    expected_sharpe = (expected_return - risk_free_rate)/ risk_taken
    st.markdown(f'Expected Sharpe Ratio: **{expected_sharpe:.2f}**')
    run_simulations = st.button('Run simulations')
    if run_simulations:
        simulated_portfolios = efficient_frontier(session_state.portfolio, trading_days, weights_df, risk_free_rate, simulations, resample_list)
        plot_efficient_frontier(simulated_portfolios, risk_free_rate, expected_sharpe,expected_return, risk_taken)
        
if session_state.portfolio is not None and session_state.portfolio.shape[1] >= 2:
    st.subheader('Backtesting Strategy', divider='rainbow')
    
    first_sharpe = st.number_input('Please select number of days to obtain the first  round of weights:', 
                                   min_value=1,
                                   max_value=10000,
                                   step=15, 
                                   value=252)
    offset = st.number_input('Please select number of days to jump:', min_value=1, 
                             max_value=10000, 
                             step=15, 
                             value=30)
    
    
    dates_range = session_state.portfolio.index.unique()
    backtest_dfs = []
    first_df =  session_state.portfolio.iloc[:first_sharpe]
    first_df['ID'] = 1
    first_df['date'] = session_state.portfolio.index[first_sharpe]  # Defina a data para a última data do intervalo
    df_id = 2
    for i in range(first_sharpe, len(dates_range), offset):
        starting_date = dates_range[i] 
        offset_date = dates_range[min(i + offset - 1, len(dates_range) - 1)]  # Garante que offset_date não ultrapasse o final de dates_range
        split_df = session_state.portfolio.loc[starting_date:offset_date]
        split_df['ID'] = df_id
        split_df['date'] = offset_date
        backtest_dfs.append(split_df)
        df_id += 1
        starting_date = offset_date
    
    
    min_constraints = [0.0] * len(tickers)
    max_constraints = [1.0] * len(tickers)

    with st.expander("***Optional Constraints***"):
        st.markdown("""
            1. Choose min and max constraints for the weights of your assets.
            
            2. The selected constraints will be normalized so that weights sum 1. Therefore, you may see weights different from 
            the constraints you selected, however, they should be proportional. 
        
        """)

        for num in range(len(tickers)):
            min_constraint = st.number_input(f'Select min weight constraint for asset {list(tickers)[num]}', min_value=0.0, max_value=1.0, step=0.05, value = 0.0)
            max_constraint = st.number_input(f'Select max weight constraint for asset {list(tickers)[num]}', min_value=0.0, max_value=1.0, step=0.05, value = 1.0)
            min_constraints[num] = min_constraint
            max_constraints[num] = max_constraint
 
backtesting = st.button('backtest')
if backtesting:
    optimized_dfs = backtest_frontier(backtest_dfs, risk_free_rate, trading_days)
    portfolios_anazlyed = len(optimized_dfs)
    backtested_df = get_max_sharpe_per_id(optimized_dfs, max_constraints, min_constraints)
    backtested_df.set_index('Date', inplace=True, drop=True)
    weight_columns = [col for col in backtested_df.columns if col.endswith('_Weight')]
    aux = session_state.data.copy()
    aux.columns = [col.replace('_Close', '_Price') for col in aux.columns] 
    price_columns = [col for col in aux.columns if col.endswith('_Price')]	
    backtested_df = backtested_df.merge(aux[price_columns], left_index=True, right_index=True, how='left')
    roi_df = returns_till_date(backtested_df, invested_cash, price_columns, weight_columns)
    
    data = {'ID': [] ,'date': [], 'asset': [], 'price': [], 'weight': [], 'quantity': []}
    for date, row in backtested_df.iterrows():
        for price_col, weight_col in zip(price_columns, weight_columns):
            asset = price_col.replace('_Price', '')
            data['ID'].append(row['ID'])
            data['date'].append(date)
            data['asset'].append(asset)
            data['price'].append(row[price_col])
            data['weight'].append(row[weight_col])
            data['quantity'].append(invested_cash*row[weight_col] /row[price_col])
        
    optimize_df = pd.DataFrame(data)

    dfs_by_id = {}
    for unique_id in optimize_df['ID'].unique():
        df_id = optimize_df[optimize_df['ID'] == unique_id][['ID', 'asset', 'price', 'weight']]
        dfs_by_id[unique_id] = df_id

    # Combine DataFrames by ID
    combined_dfs = []
    results = []
    unique_ids = sorted(dfs_by_id.keys())
    for i in range(len(unique_ids) - 1):
        current_id = unique_ids[i]
        next_id = unique_ids[i + 1]
        df_t1 = dfs_by_id[current_id].copy()
        df_t1.columns = ['ID', 'asset', f'pricesT1', f'weightsT1']
        
        df_t2 = dfs_by_id[next_id].copy()
        df_t2.columns = ['ID', 'asset', f'pricesT2', f'weightsT2']
        
        combined_df = pd.merge(df_t1, df_t2, on='asset', suffixes=('_t1', '_t2'))
        combined_df['qtT1'] = invested_cash * combined_df['weightsT1'] / combined_df['pricesT1']
        combined_dfs.append(combined_df)
    
    st.markdown(f'***Number of portfolios analyzed:*** {portfolios_anazlyed}')   
    st.dataframe(backtested_df) 
    st.markdown(f'***Number of optimizations:*** {len(combined_dfs)}')
    st.markdown(f"**Return till date:** {roi_df['return_till_date'][1]:.3f}")
    optimizedDf = optimizeBySharpe(combined_dfs[0])
    st.markdown(f"**Return till date:** {roi_df['return_till_date'][1]:.3f}")
    results.append(optimizedDf)
    for i in range(1, len(combined_dfs)):
        lastOptimized = results[i - 1]
        combined_dfs[i]['qtT1'] = lastOptimized['optQtT2']
        combined_dfs[i]['weightsT1'] = lastOptimized['optWeightsT2']
        st.markdown(f"**Return till date:** {roi_df['return_till_date'][i+1]:.3f}")
        optimizedDf = optimizeBySharpe(combined_dfs[i])
        results.append(optimizedDf)

if session_state.df is not None or session_state.data is not None or session_state.portfolio is not None or session_state.backtest is not None or session_state.optimized_data is not None:
    st.subheader("Downloads:", divider='rainbow')
    mapping = {'assets': 'data', 'allocation': 'df', 'portfolio': 'portfolio', 'backtest': 'backtest', 'optimized_data': 'optimized_data'}
    download_option = st.selectbox("Select for Download:", list(mapping.keys()))
    if download_option in mapping:
        download_button = st.button('Download')
        if download_button:
            download_link = download_dfs(session_state, mapping[download_option], mapping)
            if download_link:
                st.markdown(download_link, unsafe_allow_html=True)
    else:
        st.error("Opção de download inválida.")

def generate_combinations(dictionary, assets_per_portfolio, combinations_limit=10000):
    assets = list(dictionary.values())
    combinations = []
    for combination in itertools.combinations(assets, assets_per_portfolio):
        combinations.append(combination)
        if len(combinations) >= combinations_limit:
            break
    return combinations

def download_portfolios(data, period="1y"):
    dfs = {}
    tickers_with_errors = []

    # Map yfinance-style periods to days (Stooq needs date ranges)
    period_map_days = {
        "1y": 365,
        "2y": 730,
        "5y": 1825,
        "10y": 3650,
        "max": None
    }

    today = datetime.utcnow().date()
    days = period_map_days.get(period, 1825)

    start = today - timedelta(days=days) if days else None
    end = today

    def fetch_stooq(ticker):
        symbol = f"{ticker}.US"  # Stooq US ticker format
        df = web.DataReader(symbol, "stooq", start=start, end=end)
        df = df.sort_index()

        # Match yfinance-like columns
        if "Close" in df.columns and "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"]

        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        return df

    if isinstance(data, dict):
        items = list(data.items())  # (name, ticker)
    elif isinstance(data, list):
        items = [(t, t) for t in data]
    else:
        raise TypeError("data must be a dict {name: ticker} or a list of tickers")

    for name, ticker in items:
        try:
            hist = fetch_stooq(ticker)
            hist.columns = [f"{name}_{col}" for col in hist.columns]
            dfs[name] = hist
        except Exception as e:
            print(f"Error occurred while downloading data for {name}: {e}")
            tickers_with_errors.append(ticker)
            time.sleep(0.5)

    return dfs, tickers_with_errors