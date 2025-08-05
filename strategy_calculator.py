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
    """计算完整的MACD相关指标"""
    # 基础参数
    SHORT = 12
    LONG = 26
    MID = 9
    
    # 基础MACD计算
    df['DIF'] = (EMA(df['close'], SHORT) - EMA(df['close'], LONG)) * 100
    df['DEA'] = EMA(df['DIF'], MID)
    df['MACD'] = 2 * (df['DIF'] - df['DEA'])
    
    # MACD柱状图历史数据
    df['MACD1'] = df['MACD']
    df['MACD2'] = df['MACD1'].shift(1)  # MACDSR: 1周期前的MACD
    df['MACD3'] = df['MACD1'].shift(2)  # MACDSSR: 2周期前的MACD

    # DIF转折信号（新增）
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
    
    # 计算M2和M3（到当前的周期数）
    df['M2'] = 0  # 初始化M2
    df['M3'] = 0  # 初始化M3
    
    for i in range(len(df)):
        # 获取当前位置之前的所有金叉位置
        prev_data = df.iloc[:i+1]
        cross_positions = prev_data[prev_data['金叉']].index
        
        if len(cross_positions) >= 2:  # 至少有两次金叉才能计算M2
            second_last_cross = cross_positions[-2]  # 倒数第二次金叉位置
            df.loc[df.index[i], 'M2'] = len(df.loc[second_last_cross:df.index[i]]) - 1
            
        if len(cross_positions) >= 3:  # 至少有三次金叉才能计算M3
            third_last_cross = cross_positions[-3]  # 倒数第三次金叉位置
            df.loc[df.index[i], 'M3'] = len(df.loc[third_last_cross:df.index[i]]) - 1
    
    # 填充NaN值
    df['M2'] = df['M2'].fillna(0)
    df['M3'] = df['M3'].fillna(0)
    
    # 计算N2和N3（到当前的周期数）
    df['N2'] = 0  # 初始化N2
    df['N3'] = 0  # 初始化N3
    
    for i in range(len(df)):
        # 获取当前位置之前的所有死叉位置
        prev_data = df.iloc[:i+1]
        cross_positions = prev_data[prev_data['死叉']].index
        
        if len(cross_positions) >= 2:  # 至少有两次死叉才能计算N2
            second_last_cross = cross_positions[-2]  # 倒数第二次死叉位置
            df.loc[df.index[i], 'N2'] = len(df.loc[second_last_cross:df.index[i]]) - 1
            
        if len(cross_positions) >= 3:  # 至少有三次死叉才能计算N3
            third_last_cross = cross_positions[-3]  # 倒数第三次死叉位置
            df.loc[df.index[i], 'N3'] = len(df.loc[third_last_cross:df.index[i]]) - 1
    
    # 填充NaN值
    df['N2'] = df['N2'].fillna(0)
    df['N3'] = df['N3'].fillna(0)
    
    # 计算各周期高低点位置
    for i in range(len(df)):
        m1 = int(df['M1'].iloc[i])
        
        # 初始化列（如果不存在）
        for col in ['CH1', 'CH2', 'CH3', 'DIFH1', 'DIFH2', 'DIFH3']:
            if col not in df.columns:
                df[col] = 0
        
        # CH1和DIFH1：M1+1日内的最高值
        if i >= m1:
            start_idx = max(0, i-m1-1)  # M1+1日前的位置
            df.loc[df.index[i], 'CH1'] = df['close'].iloc[start_idx:i+1].max()
            df.loc[df.index[i], 'DIFH1'] = df['DIF'].iloc[start_idx:i+1].max()
        
        # CH2和DIFH2：M1+1日前的CH1和DIFH1
        if i > m1:
            ref_idx = i - (m1 + 1)  # 向前推M1+1日
            if ref_idx >= 0:
                df.loc[df.index[i], 'CH2'] = df['CH1'].iloc[ref_idx]
                df.loc[df.index[i], 'DIFH2'] = df['DIFH1'].iloc[ref_idx]
        
        # CH3和DIFH3：M1+1日前的CH2和DIFH2
        if i > m1:
            ref_idx = i - (m1 + 1)  # 向前推M1+1日
            if ref_idx >= 0:
                df.loc[df.index[i], 'CH3'] = df['CH2'].iloc[ref_idx]
                df.loc[df.index[i], 'DIFH3'] = df['DIFH2'].iloc[ref_idx]
        
        n1 = int(df['N1'].iloc[i])
        
        # 初始化列（如果不存在）
        for col in ['CL1', 'CL2', 'CL3', 'DIFL1', 'DIFL2', 'DIFL3']:
            if col not in df.columns:
                df[col] = 0
        
        # CL1和DIFL1：N1+1日内的最低值
        if i >= n1:
            start_idx = max(0, i-n1-1)  # N1+1日前的位置
            df.loc[df.index[i], 'CL1'] = df['close'].iloc[start_idx:i+1].min()
            df.loc[df.index[i], 'DIFL1'] = df['DIF'].iloc[start_idx:i+1].min()
        
        # CL2和DIFL2：N1+1日前的CL1和DIFL1
        if i > n1:
            ref_idx = i - (n1 + 1)  # 向前推N1+1日
            if ref_idx >= 0:
                df.loc[df.index[i], 'CL2'] = df['CL1'].iloc[ref_idx]
                df.loc[df.index[i], 'DIFL2'] = df['DIFL1'].iloc[ref_idx]
        
        # CL3和DIFL3：N1+1日前的CL2和DIFL2
        if i > n1:
            ref_idx = i - (n1 + 1)  # 向前推N1+1日
            if ref_idx >= 0:
                df.loc[df.index[i], 'CL3'] = df['CL2'].iloc[ref_idx]
                df.loc[df.index[i], 'DIFL3'] = df['DIFL2'].iloc[ref_idx]
    
    # 计算PDIFH1和MDIFH1（新增）
    df['PDIFH1'] = df.apply(lambda x: 
        int(np.log10(abs(x['DIFH1']))) - 1 if x['DIFH1'] > 0 
        else int(np.log10(abs(x['DIFH1']))) - 1 if x['DIFH1'] < 0 
        else 0, axis=1)
    df['MDIFH1'] = df.apply(lambda x: 
        int(x['DIFH1'] / (10 ** x['PDIFH1'])) if x['PDIFH1'] != 0 
        else int(x['DIFH1']), axis=1)
    
    # 计算PDIFH2和MDIFH2
    df['PDIFH2'] = df.apply(lambda x: 
        int(np.log10(abs(x['DIFH2']))) - 1 if x['DIFH2'] > 0 
        else int(np.log10(abs(x['DIFH2']))) - 1 if x['DIFH2'] < 0 
        else 0, axis=1)
    df['MDIFH2'] = df.apply(lambda x: 
        int(x['DIFH2'] / (10 ** x['PDIFH2'])) if x['PDIFH2'] != 0 
        else int(x['DIFH2']), axis=1)

    # 计算PDIFH3和MDIFH3
    df['PDIFH3'] = df.apply(lambda x: 
        int(np.log10(abs(x['DIFH3']))) - 1 if x['DIFH3'] > 0 
        else int(np.log10(abs(x['DIFH3']))) - 1 if x['DIFH3'] < 0 
        else 0, axis=1)
    df['MDIFH3'] = df.apply(lambda x: 
        int(x['DIFH3'] / (10 ** x['PDIFH3'])) if x['PDIFH3'] != 0 
        else int(x['DIFH3']), axis=1)

    # 计算MDIFT2和MDIFT3
    df['MDIFT2'] = df.apply(lambda x: 
        int(x['DIF'] / (10 ** x['PDIFH2'])) if x['PDIFH2'] != 0 
        else int(x['DIF']), axis=1)
    df['MDIFT3'] = df.apply(lambda x: 
        int(x['DIF'] / (10 ** x['PDIFH3'])) if x['PDIFH3'] != 0 
        else int(x['DIF']), axis=1)

    # 计算PDIFL1和MDIFL1（新增）
    df['PDIFL1'] = df.apply(lambda x: 
        int(np.log10(abs(x['DIFL1']))) - 1 if x['DIFL1'] > 0 
        else int(np.log10(abs(x['DIFL1']))) - 1 if x['DIFL1'] < 0 
        else 0, axis=1)
    df['MDIFL1'] = df.apply(lambda x: 
        int(x['DIFL1'] / (10 ** x['PDIFL1'])) if x['PDIFL1'] != 0 
        else int(x['DIFL1']), axis=1)

    # 计算PDIFL2和MDIFL2（底背离）
    df['PDIFL2'] = df.apply(lambda x: 
        int(np.log10(abs(x['DIFL2']))) - 1 if x['DIFL2'] > 0 
        else int(np.log10(abs(x['DIFL2']))) - 1 if x['DIFL2'] < 0 
        else 0, axis=1)
    df['MDIFL2'] = df.apply(lambda x: 
        int(x['DIFL2'] / (10 ** x['PDIFL2'])) if x['PDIFL2'] != 0 
        else int(x['DIFL2']), axis=1)

    # 计算PDIFL3和MDIFL3
    df['PDIFL3'] = df.apply(lambda x: 
        int(np.log10(abs(x['DIFL3']))) - 1 if x['DIFL3'] > 0 
        else int(np.log10(abs(x['DIFL3']))) - 1 if x['DIFL3'] < 0 
        else 0, axis=1)
    df['MDIFL3'] = df.apply(lambda x: 
        int(x['DIFL3'] / (10 ** x['PDIFL3'])) if x['PDIFL3'] != 0 
        else int(x['DIFL3']), axis=1)

    # 计算MDIFB2和MDIFB3
    df['MDIFB2'] = df.apply(lambda x: 
        int(x['DIF'] / (10 ** x['PDIFL2'])) if x['PDIFL2'] != 0 
        else int(x['DIF']), axis=1)
    df['MDIFB3'] = df.apply(lambda x: 
        int(x['DIF'] / (10 ** x['PDIFL3'])) if x['PDIFL3'] != 0 
        else int(x['DIF']), axis=1)
    
    # 修改后的顶背离判断（新增DEA>0条件）
    df['直接顶背离'] = ((df['CH1'] > df['CH2']) & 
                    (df['MDIFT2'] < df['MDIFH2']) & 
                    ((df['MACD'] > 0) & (df['MACD'].shift(1) > 0)) & 
                    (df['MDIFT2'] >= df['MDIFT2'].shift(1)) &
                    (df['DEA'] > 0))
    
    df['隔峰顶背离'] = ((df['CH1'] > df['CH3']) & (df['MDIFH3'] >= df['MDIFH2']) &
                    (df['MDIFT3'] < df['MDIFH3']) & 
                    ((df['MACD'] > 0) & (df['MACD'].shift(1) > 0)) & 
                    (df['MDIFT3'] >= df['MDIFT3'].shift(1)) &
                    (df['DEA'] > 0))
    
    # 修改后的底背离判断（新增DEA<0条件）
    df['直接底背离'] = ((df['CL1'] < df['CL2']) & 
                    (df['MDIFB2'] > df['MDIFL2']) & 
                    ((df['MACD'] < 0) & (df['MACD'].shift(1) < 0)) & 
                    (df['MDIFB2'] <= df['MDIFB2'].shift(1)) &
                    (df['DEA'] < 0))
    
    df['隔峰底背离'] = ((df['CL1'] < df['CL3']) & 
                    (df['MDIFB3'] > df['MDIFL3']) & 
                    ((df['MACD'] < 0) & (df['MACD'].shift(1) < 0)) & 
                    (df['MDIFB3'] <= df['MDIFB3'].shift(1)) &
                    (df['DEA'] < 0))
    
    # 顶底背离信号合并
    df['T'] = df['直接顶背离'] | df['隔峰顶背离']
    df['B'] = df['直接底背离'] | df['隔峰底背离']
    
    # 修改后的顶底背离确认信号(TG和BG) - 基于DIF转折
    df['直接TG'] = ((df['DIF'] < df['DIF4']) & df['直接顶背离'].shift(1) & (df['DIF'] > 0))
    df['隔峰TG'] = ((df['DIF'] < df['DIF4']) & df['隔峰顶背离'].shift(1) & (df['DIF'] > 0))
    df['TG'] = df['直接TG'] | df['隔峰TG']
    
    df['直接BG'] = ((df['DIF'] > df['DIF4']) & df['直接底背离'].shift(1) & (df['DIF'] < 0))
    df['隔峰BG'] = ((df['DIF'] > df['DIF4']) & df['隔峰底背离'].shift(1) & (df['DIF'] < 0))
    df['BG'] = df['直接BG'] | df['隔峰BG']
    
    # 结构信号
    df['顶结构'] = df['TG']
    df['底结构'] = df['BG']
    
    # 填充所有可能的NaN值
    df = df.fillna(0)
    
    return df

def calculate_signal_returns(df):
    """
    计算信号出现后的收益率，只返回最后一个交易日出现信号的股票
    
    Parameters:
    -----------
    df : pd.DataFrame
        包含价格和信号的DataFrame
    
    Returns:
    --------
    dict
        包含当日信号和历史信号收益率的字典
    """
    returns = {}
    last_date = df.index[-1]  # 最后一个交易日
    
    # 打印调试信息
    print(f"最后交易日: {last_date}")
    print(f"数据长度: {len(df)}")
    print(f"数据日期范围: {df.index[0]} 到 {df.index[-1]}")
    
    # 检查最后一个交易日是否有信号
    has_top_signal = df.loc[last_date, '顶结构']
    has_bottom_signal = df.loc[last_date, '底结构']
    
    print(f"最后一天顶结构信号: {has_top_signal}")
    print(f"最后一天底结构信号: {has_bottom_signal}")
    
    if not (has_top_signal or has_bottom_signal):
        return None  # 如果最后一个交易日没有信号，返回None
    
    returns['last_date'] = last_date
    
    # 如果有顶结构信号
    if has_top_signal:
        returns['signal_type'] = '顶结构'
        # 获取历史顶结构信号日期（不包括最后一个交易日）
        historical_signals = df[df['顶结构']].index[:-1]
        
        if len(historical_signals) >= 1:
            last_signal = historical_signals[-1]
            returns['last_signal'] = {
                'date': last_signal,
                '3d': calculate_return(df, last_signal, 3),
                '5d': calculate_return(df, last_signal, 5),
                '10d': calculate_return(df, last_signal, 10)
            }
            
            if len(historical_signals) >= 2:
                second_last_signal = historical_signals[-2]
                returns['second_last_signal'] = {
                    'date': second_last_signal,
                    '3d': calculate_return(df, second_last_signal, 3),
                    '5d': calculate_return(df, second_last_signal, 5),
                    '10d': calculate_return(df, second_last_signal, 10)
                }
    
    # 如果有底结构信号
    if has_bottom_signal:
        returns['signal_type'] = '底结构'
        # 获取历史底结构信号日期（不包括最后一个交易日）
        historical_signals = df[df['底结构']].index[:-1]
        
        if len(historical_signals) >= 1:
            last_signal = historical_signals[-1]
            returns['last_signal'] = {
                'date': last_signal,
                '3d': calculate_return(df, last_signal, 3),
                '5d': calculate_return(df, last_signal, 5),
                '10d': calculate_return(df, last_signal, 10)
            }
            
            if len(historical_signals) >= 2:
                second_last_signal = historical_signals[-2]
                returns['second_last_signal'] = {
                    'date': second_last_signal,
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