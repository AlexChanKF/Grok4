import pandas as pd
import numpy as np

def wilder_ma(series, period):
    ma = series.rolling(window=period, min_periods=period).mean()
    wma = ma.copy()
    for i in range(period + 1, len(series)):
        wma[i] = (wma[i - 1] * (period - 1) + series[i]) / period
    return wma

def calculate_rsi(series, period):
    delta = series.diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    gain.iloc[0] = np.nan
    loss.iloc[0] = np.nan
    avg_gain = wilder_ma(gain, period)
    avg_loss = wilder_ma(loss, period)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def main():
    # Read the input CSV
    df = pd.read_csv('input.csv')
    
    # If 'Adj,Close' exists due to typo, rename to 'Adj Close'
    if 'Adj,Close' in df.columns:
        df.rename(columns={'Adj,Close': 'Adj Close'}, inplace=True)
    
    # Convert Date to datetime and sort ascending
    df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d')
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Calculate Moving Averages
    df['10MA'] = df['Close'].rolling(window=10, min_periods=10).mean()
    df['20MA'] = df['Close'].rolling(window=20, min_periods=20).mean()
    df['50MA'] = df['Close'].rolling(window=50, min_periods=50).mean()
    df['200MA'] = df['Close'].rolling(window=200, min_periods=200).mean()
    
    # Calculate Bollinger Bands (20-period, 2 std dev)
    df['BB_Mid'] = df['20MA']
    std_20 = df['Close'].rolling(window=20, min_periods=20).std()
    df['BB_Up'] = df['BB_Mid'] + 2 * std_20
    df['BB_Down'] = df['BB_Mid'] - 2 * std_20
    
    # Calculate RSI(9) and RSI(14)
    df['RSI(9)'] = calculate_rsi(df['Close'], 9)
    df['RSI (14)'] = calculate_rsi(df['Close'], 14)
    
    # Calculate MACD (12,26,9)
    ema_12 = df['Close'].ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False, min_periods=26).mean()
    df['MACD(12,26,9) (MACD Line)'] = ema_12 - ema_26
    df['MACD(12,26,9) (Signal Line)'] = df['MACD(12,26,9) (MACD Line)'].ewm(span=9, adjust=False, min_periods=9).mean()
    df['MACD(12,26,9) (Histogram)'] = df['MACD(12,26,9) (MACD Line)'] - df['MACD(12,26,9) (Signal Line)']
    
    # Calculate ATR(14)
    df['High_Low'] = df['High'] - df['Low']
    df['High_PrevClose'] = abs(df['High'] - df['Close'].shift(1))
    df['Low_PrevClose'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['High_Low', 'High_PrevClose', 'Low_PrevClose']].max(axis=1)
    df['ATR'] = wilder_ma(df['TR'], 14)
    
    # Drop temporary columns
    df.drop(['High_Low', 'High_PrevClose', 'Low_PrevClose', 'TR'], axis=1, inplace=True)
    
    # Round the technical indicators to 2 decimal places
    technical_cols = [
        '10MA', '20MA', '50MA', '200MA',
        'BB_Up', 'BB_Mid', 'BB_Down',
        'RSI(9)', 'RSI (14)',
        'MACD(12,26,9) (MACD Line)', 'MACD(12,26,9) (Signal Line)', 'MACD(12,26,9) (Histogram)',
        'ATR'
    ]
    df[technical_cols] = df[technical_cols].round(2)
    
    # Select and order the output columns
    output_columns = [
        'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
        '10MA', '20MA', '50MA', '200MA',
        'BB(Up)', 'BB(Mid)', 'BB(Down)',
        'RSI(9)', 'RSI (14)',
        'MACD(12,26,9) (MACD Line)', 'MACD(12,26,9) (Signal Line)', 'MACD(12,26,9) (Histogram)',
        'ATR'
    ]
    
    # Rename BB columns to match output format
    df.rename(columns={
        'BB_Up': 'BB(Up)',
        'BB_Mid': 'BB(Mid)',
        'BB_Down': 'BB(Down)'
    }, inplace=True)
    
    # Output to CSV
    df[output_columns].to_csv('output.csv', index=False)

if __name__ == '__main__':
    main()