import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
import io
import base64

def fetch_yfinance_data(symbol, interval, start_date, end_date):
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date, interval=interval)
    df.reset_index(inplace=True)
    df.rename(columns={'Datetime': 'Date'}, inplace=True)
    return df

def is_gap_filled(gap, current_low, current_high):
    if gap['Type'] == 'Bullish':
        return current_low <= gap['FVG_Start']
    else:  # Bearish
        return current_high >= gap['FVG_Start']

def identify_fair_value_gaps(df, is_backtest=False, min_gap_size=0.0001):
    
    fvg_data = []
    open_gaps = []
    for i in range(2, len(df)):
        prev_high, prev_low = df.iloc[i-2]['High'], df.iloc[i-2]['Low']
        current_high, current_low = df.iloc[i]['High'], df.iloc[i]['Low']
        current_time = df.iloc[i]['Date']
        
        if current_low > prev_high:
            gap_size = current_low - prev_high
            if gap_size > min_gap_size:
                new_gap = {
                    'index': i,
                    'Type': 'Bullish',
                    'FVG_Start': prev_high,
                    'FVG_End': current_low,
                    'Start_Time': current_time,
                    'Used': False  # Flag to indicate if the gap has been used for a trade
                }
                fvg_data.append(new_gap)
                open_gaps.append(new_gap)
        elif prev_low > current_high:
            gap_size = prev_low - current_high
            if gap_size > min_gap_size:
                new_gap = {
                    'index': i,
                    'Type': 'Bearish',
                    'FVG_Start': prev_low,
                    'FVG_End': current_high,
                    'Start_Time': current_time,
                    'Used': False
                }
                fvg_data.append(new_gap)
                open_gaps.append(new_gap)
        
        # Remove gaps that are filled
        open_gaps = [gap for gap in open_gaps if not is_gap_filled(gap, current_low, current_high)]
    
    return pd.DataFrame(open_gaps) if not is_backtest else pd.DataFrame(fvg_data)

def backtest_fvg_strategy(df, fvg_df):
    open_gaps = []
    trades = []
    used_fvgs = set()
    closed_trades = []
    
    for i in range(len(df)):
        current_price = df.iloc[i]['Close']
        current_time = df.iloc[i]['Date']
        current_low, current_high = df.iloc[i]['Low'], df.iloc[i]['High']
        
        # Add new gaps if they are open and not used
        new_gaps = fvg_df[(fvg_df['index'] == i) & (~fvg_df['Used'])]
        for _, new_gap in new_gaps.iterrows():
            if not is_gap_filled(new_gap, current_low, current_high):
                open_gaps.append(new_gap.to_dict())
                fvg_df.at[_, 'Used'] = True  # Mark as used
        
        # Check for filled gaps and potential trades
        for gap in open_gaps[:]:
            mid_point = (gap['FVG_Start'] + gap['FVG_End']) / 2
            
            # Prevent multiple entries on the same gap
            if gap['Used']:
                continue
            
            # Check for entry
            entry_condition = False
            trade_direction = None
            entry_price = None
            
            if gap['Type'] == 'Bullish':
                # Enter when price crosses above mid_point from below
                if (df.iloc[i-1]['Close'] < mid_point <= current_price):
                    entry_condition = True
                    trade_direction = 'Long'
                    entry_price = mid_point
            elif gap['Type'] == 'Bearish':
                # Enter when price crosses below mid_point from above
                if (df.iloc[i-1]['Close'] > mid_point >= current_price):
                    entry_condition = True
                    trade_direction = 'Short'
                    entry_price = mid_point
            
            if entry_condition:
                # Define stop-loss and take-profit based on gap size
                gap_size = abs(gap['FVG_Start'] - gap['FVG_End'])
                stop_loss = entry_price - gap_size if trade_direction == 'Long' else entry_price + gap_size
                take_profit = entry_price + 2 * gap_size if trade_direction == 'Long' else entry_price - 2 * gap_size
                
                trades.append({
                    'Entry_Time': current_time,
                    'Entry_Price': entry_price,
                    'Type': gap['Type'],  # Using 'Type' here
                    'Stop_Loss': stop_loss,
                    'Take_Profit': take_profit,
                    'FVG_Index': gap['index'],
                    'Trade_Direction': trade_direction
                })
                used_fvgs.add(gap['index'])
                open_gaps.remove(gap)  # Remove gap after trade entry
    
    # Process trades
    for trade in trades:
        entry_index = df.index[df['Date'] == trade['Entry_Time']].tolist()
        if not entry_index:
            # Find the closest index
            entry_index = (df['Date'] - trade['Entry_Time']).abs().idxmin()
        else:
            entry_index = entry_index[0]
        
        # Iterate through subsequent rows to find exit condition
        exit_found = False
        for i in range(entry_index + 1, len(df)):
            current_price = df.iloc[i]['Close']
            current_time = df.iloc[i]['Date']
            
            if trade['Trade_Direction'] == 'Long':
                if current_price >= trade['Take_Profit']:
                    result = 'Win'
                elif current_price <= trade['Stop_Loss']:
                    result = 'Loss'
                else:
                    continue
            elif trade['Trade_Direction'] == 'Short':
                if current_price <= trade['Take_Profit']:
                    result = 'Win'
                elif current_price >= trade['Stop_Loss']:
                    result = 'Loss'
                else:
                    continue
            
            # Append closed trade with Trade_Direction
            closed_trades.append({
                'Entry_Time': trade['Entry_Time'],
                'Exit_Time': current_time,
                'Entry_Price': trade['Entry_Price'],
                'Exit_Price': current_price,
                'Type': trade['Type'],  # Changed from 'FVG_Type' to 'Type'
                'Result': result,
                'FVG_Index': trade['FVG_Index'],
                'Trade_Direction': trade['Trade_Direction']  # Included Trade_Direction
            })
            exit_found = True
            break
        
        if not exit_found:
            # Handle trades that did not close within the backtest period
            closed_trades.append({
                'Entry_Time': trade['Entry_Time'],
                'Exit_Time': df.iloc[-1]['Date'],
                'Entry_Price': trade['Entry_Price'],
                'Exit_Price': df.iloc[-1]['Close'],
                'Type': trade['Type'],  # Changed from 'FVG_Type' to 'Type'
                'Result': 'Open',
                'FVG_Index': trade['FVG_Index'],
                'Trade_Direction': trade['Trade_Direction']  # Included Trade_Direction
            })
    
    return pd.DataFrame(closed_trades), used_fvgs

def plot_candlesticks_with_fvg_and_trades(df, open_gaps, title):
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Plot candlesticks
    width = 0.6
    width2 = 0.05
    up = df[df.Close >= df.Open]
    down = df[df.Close < df.Open]
    
    # Plot up candles
    ax.bar(up.index, up.Close - up.Open, width, bottom=up.Open, color='g', edgecolor='black', linewidth=0.5)
    ax.bar(up.index, up.High - up.Close, width2, bottom=up.Close, color='black', edgecolor='black', linewidth=0.5)
    ax.bar(up.index, up.Low - up.Open, width2, bottom=up.Open, color='black', edgecolor='black', linewidth=0.5)
    
    # Plot down candles
    ax.bar(down.index, down.Close - down.Open, width, bottom=down.Open, color='r', edgecolor='black', linewidth=0.5)
    ax.bar(down.index, down.High - down.Open, width2, bottom=down.Open, color='black', edgecolor='black', linewidth=0.5)
    ax.bar(down.index, down.Low - down.Close, width2, bottom=down.Close, color='black', edgecolor='black', linewidth=0.5)
    
    # Plot Open Fair Value Gaps (only open gaps)
    # Plot Open Fair Value Gaps and their mid-points
    for gap in open_gaps:
        gap_index = gap['index']
        mid_point = (gap['FVG_Start'] + gap['FVG_End']) / 2
        print(gap)
        rect = Rectangle((gap_index, gap['FVG_End']), 
                         len(df) - gap_index, 
                         gap['FVG_Start'] - gap['FVG_End'], 
                         facecolor="yellow",
                         alpha=0.2,
                         edgecolor='none')
        ax.add_patch(rect)
        ax.axhline(y=mid_point, xmin=gap_index/len(df), xmax=1, color='red', linestyle='--', linewidth=1, alpha=0.7)
   
    ax.set_xlim(-1, len(df))
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(title)
    
    # Set x-axis ticks to show about 10 dates
    step = max(len(df)//10, 1)
    date_ticks = df['Date'][::step]
    ax.set_xticks(range(0, len(df), step))
    ax.set_xticklabels([dt.strftime('%Y-%m-%d') for dt in date_ticks], rotation=45)
    
    plt.tight_layout()
    
    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    
    # Encode the image in base64
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    return image_base64

def calculate_strategy_metrics(df, trades_df, start_date, end_date):
    # Buy and Hold Return
    buy_and_hold_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100

    # Strategy Return
    strategy_returns = trades_df.apply(
        lambda x: (x['Exit_Price'] - x['Entry_Price']) / x['Entry_Price']
        if x['Type'] == 'Bullish' and x['Trade_Direction'] == 'Long'
        else (x['Entry_Price'] - x['Exit_Price']) / x['Entry_Price']
        if x['Type'] == 'Bearish' and x['Trade_Direction'] == 'Short'
        else 0, axis=1
    )
    strategy_return = strategy_returns.sum() * 100

    # Max Drawdown for Strategy
    cumulative_strategy_returns = (1 + strategy_returns).cumprod()
    peak_strategy = cumulative_strategy_returns.expanding(min_periods=1).max()
    drawdown_strategy = (cumulative_strategy_returns / peak_strategy) - 1
    max_drawdown_strategy = drawdown_strategy.min() * 100

    # Max Drawdown for Buy and Hold
    cumulative_bnh_returns = (1 + df['Close'].pct_change().fillna(0)).cumprod()
    peak_bnh = cumulative_bnh_returns.expanding(min_periods=1).max()
    drawdown_bnh = (cumulative_bnh_returns / peak_bnh) - 1
    max_drawdown_bnh = drawdown_bnh.min() * 100

    # Exposure Time
    total_time = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).total_seconds()
    
    # Create list of all trade intervals
    trade_intervals = trades_df[['Entry_Time', 'Exit_Time']].dropna().sort_values(by='Entry_Time').values.tolist()
    
    # Merge overlapping intervals
    merged_intervals = []
    for interval in trade_intervals:
        if not merged_intervals:
            merged_intervals.append(interval)
        else:
            last = merged_intervals[-1]
            if interval[0] <= last[1]:
                merged_intervals[-1][1] = max(last[1], interval[1])
            else:
                merged_intervals.append(interval)
    
    # Calculate total exposure time without double-counting overlaps
    trade_time = sum((pd.to_datetime(end) - pd.to_datetime(start)).total_seconds() for start, end in merged_intervals)
    exposure_time = (trade_time / total_time) * 100 if total_time > 0 else 0

    # Number of Trades
    num_trades = len(trades_df)

    # Win Rate
    win_rate = len(trades_df[trades_df['Result'] == 'Win']) / num_trades * 100 if num_trades > 0 else 0

    # Average Win and Loss
    avg_win = strategy_returns[trades_df['Result'] == 'Win'].mean() * 100 if len(trades_df[trades_df['Result'] == 'Win']) > 0 else 0
    avg_loss = strategy_returns[trades_df['Result'] == 'Loss'].mean() * 100 if len(trades_df[trades_df['Result'] == 'Loss']) > 0 else 0

    # Profit Factor
    total_profit = strategy_returns[trades_df['Result'] == 'Win'].sum()
    total_loss = abs(strategy_returns[trades_df['Result'] == 'Loss'].sum())
    profit_factor = total_profit / total_loss if total_loss != 0 else float('inf')

    return {
        'Buy and Hold Return (%)': round(buy_and_hold_return, 2),
        'Max Drawdown Buy and Hold (%)': round(max_drawdown_bnh, 2),
        'Strategy Return (%)': round(strategy_return, 2),
        'Max Drawdown Strategy (%)': round(max_drawdown_strategy, 2),
        'Exposure Time (%)': round(exposure_time, 2),
        'Number of Trades': num_trades,
        'Win Rate (%)': round(win_rate, 2),
        'Average Win (%)': round(avg_win, 2),
        'Average Loss (%)': round(avg_loss, 2),
        'Profit Factor': round(profit_factor, 2)
    }