import pandas as pd
import numpy as np

def EMA(series, periods):
    """计算EMA指标"""
    return series.ewm(span=periods, adjust=False).mean()

def CROSS(series1, series2):
    """判断向上金叉"""
    return (series1 > series2) & (series1.shift(1) <= series2.shift(1))

def BARSLAST(condition):
    """计算上一次条件成立到当前的周期数"""
    result = np.zeros(len(condition))
    count = 0
    last_true = False
    
    for i in range(len(condition)):
        if condition.iloc[i]:  # 当前位置条件成立
            count = 0  # 改为0
            last_true = True
        elif last_true:  # 之前有条件成立
            count += 1
        result[i] = count
    
    return pd.Series(result, index=condition.index)

def calculate_macd_indicators(df):
    """计算MACD相关指标"""
    # 基础参数
    SHORT = 12
    LONG = 26
    MID = 9
    
    # 基础MACD计算
    df['DIF'] = (EMA(df['close'], SHORT) - EMA(df['close'], LONG)) * 100
    df['DEA'] = EMA(df['DIF'], MID)
    df['MACD'] = 2 * (df['DIF'] - df['DEA'])
    
    # DIF转折信号
    df['DIF4'] = df['DIF'].shift(1)  # 1周期前的DIF
    df['DIF5'] = df['DIF'].shift(2)  # 2周期前的DIF
    df['DIF顶转折'] = (df['DIF'] > df['DEA']) & (df['DIF4'] > df['DIF']) & (df['DIF5'] < df['DIF4'])
    df['DIF底转折'] = (df['DIF'] < df['DEA']) & (df['DIF4'] < df['DIF']) & (df['DIF5'] > df['DIF4'])
    
    # 金叉和死叉
    df['金叉'] = CROSS(df['DIF'], df['DEA'])
    df['死叉'] = CROSS(df['DEA'], df['DIF'])
    
    # 计算各周期金叉死叉位置
    df['M1'] = BARSLAST(df['金叉'])  # 最近一次金叉的位置
    df['N1'] = BARSLAST(df['死叉'])  # 最近一次死叉的位置
    
    # 计算顶底结构信号
    df['直接TG'] = ((df['DIF'] < df['DIF4']) & df['DIF顶转折'].shift(1) & (df['DIF'] > 0))
    df['隔峰TG'] = ((df['DIF'] < df['DIF4']) & df['DIF顶转折'].shift(1) & (df['DIF'] > 0))
    df['TG'] = df['直接TG'] | df['隔峰TG']
    
    df['直接BG'] = ((df['DIF'] > df['DIF4']) & df['DIF底转折'].shift(1) & (df['DIF'] < 0))
    df['隔峰BG'] = ((df['DIF'] > df['DIF4']) & df['DIF底转折'].shift(1) & (df['DIF'] < 0))
    df['BG'] = df['直接BG'] | df['隔峰BG']
    
    # 结构信号
    df['顶结构'] = df['TG']
    df['底结构'] = df['BG']
    
    return df

def calculate_signal_returns(df):
    """
    计算信号出现后的收益率
    
    Parameters:
    -----------
    df : pd.DataFrame
        包含价格和信号的DataFrame
    
    Returns:
    --------
    dict
        包含信号出现后3日、5日、10日收益率的字典
    """
    returns = {}
    
    # 计算顶结构信号的收益率
    top_signals = df[df['顶结构']].index
    if len(top_signals) >= 1:
        last_signal = top_signals[-1]
        returns['last_top'] = {
            '3d': calculate_return(df, last_signal, 3),
            '5d': calculate_return(df, last_signal, 5),
            '10d': calculate_return(df, last_signal, 10)
        }
        
        if len(top_signals) >= 2:
            second_last_signal = top_signals[-2]
            returns['second_last_top'] = {
                '3d': calculate_return(df, second_last_signal, 3),
                '5d': calculate_return(df, second_last_signal, 5),
                '10d': calculate_return(df, second_last_signal, 10)
            }
    
    # 计算底结构信号的收益率
    bottom_signals = df[df['底结构']].index
    if len(bottom_signals) >= 1:
        last_signal = bottom_signals[-1]
        returns['last_bottom'] = {
            '3d': calculate_return(df, last_signal, 3),
            '5d': calculate_return(df, last_signal, 5),
            '10d': calculate_return(df, last_signal, 10)
        }
        
        if len(bottom_signals) >= 2:
            second_last_signal = bottom_signals[-2]
            returns['second_last_bottom'] = {
                '3d': calculate_return(df, second_last_signal, 3),
                '5d': calculate_return(df, second_last_signal, 5),
                '10d': calculate_return(df, second_last_signal, 10)
            }
    
    return returns

def calculate_return(df, signal_date, days):
    """
    计算从信号日期开始的n日收益率
    """
    try:
        signal_idx = df.index.get_loc(signal_date)
        if signal_idx + days >= len(df):
            return None
        
        start_price = df['close'].iloc[signal_idx]
        end_price = df['close'].iloc[signal_idx + days]
        return ((end_price - start_price) / start_price) * 100
    except:
        return None