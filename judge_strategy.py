import pandas as pd
import numpy as np
import akshare as ak
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import pytz  # 需安装：pip install pytz

def get_stock_data(stock_code):
    """获取股票数据"""
    try:
        # 设置开始日期为2020年1月1日
        beijing_tz = pytz.timezone('Asia/Shanghai')
        now = datetime.now(beijing_tz)
        start_date = '2020-01-01'
        end_date = now.strftime('%Y-%m-%d')
        
        # 使用akshare获取指数数据
        df = ak.stock_zh_index_daily(symbol=stock_code)
        
        if df is None or df.empty:
            print(f"无法获取股票 {stock_code} 的数据")
            return None
        
        # 确保date列存在并转换为日期格式
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            # 如果没有date列，使用索引作为日期
            df['date'] = pd.to_datetime(df.index)
        
        # 过滤日期范围
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        if df.empty:
            print(f"过滤后数据为空，请检查日期范围")
            return None
        
        # 设置date为索引
        df.set_index('date', inplace=True)
        
        # 确保所有必需的列都存在
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                print(f"缺少必需的列: {col}")
                return None
        
        print(f"成功获取 {stock_code} 数据，共 {len(df)} 条记录")
        return df
        
    except Exception as e:
        print(f"获取股票数据时出错: {e}")
        print("请确保输入的股票代码格式正确")
        return None

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

def calculate_macd_indicators_new(df):
    """计算修改后的MACD相关指标"""
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
    df['直接TG'] = (df['DIF']<df['DIF4'] & df['直接顶背离'].shift(1) & (df['DIF'] > 0))
    df['隔峰TG'] = (df['DIF']<df['DIF4'] & df['隔峰顶背离'].shift(1) & (df['DIF'] > 0))
    df['TG'] = df['直接TG'] | df['隔峰TG']
    
    df['直接BG'] = (df['DIF']>df['DIF4'] & df['直接底背离'].shift(1) & (df['DIF'] < 0))
    df['隔峰BG'] = (df['DIF']>df['DIF4'] & df['隔峰底背离'].shift(1) & (df['DIF'] < 0))
    df['BG'] = df['直接BG'] | df['隔峰BG']
    
    # 将TG和BG转换为数值：TG=True时为1，BG=True时为-1，其他为0
    df['TG_数值'] = df['TG'].astype(int)
    df['BG_数值'] = -df['BG'].astype(int)
    
    # 修改后的背离消失条件
    df['直接顶背离消失'] = (df['直接顶背离'].shift(1) & (df['MDIFH1'] > df['MDIFH2']))
    df['隔峰顶背离消失'] = (df['隔峰顶背离'].shift(1) & (df['MDIFH1'] > df['MDIFH3']))
    
    df['直接底背离消失'] = (df['直接底背离'].shift(1) & (df['MDIFL1'] <= df['MDIFL2']))
    df['隔峰底背离消失'] = (df['隔峰底背离'].shift(1) & (df['MDIFL1'] <= df['MDIFL3']))
    
    # 钝化信号
    df['底钝化'] = df['B']  
    df['顶钝化'] = df['T']
    
    # 结构信号
    df['顶结构'] = df['TG']
    df['底结构'] = df['BG']
    
    # 最终背离信号
    df['顶背离'] = df['T'] | df['顶结构']
    df['底背离'] = df['B'] | df['底结构']
    
    # 买卖信号
    df['GOLDEN_CROSS'] = CROSS(df['DIF'], df['DEA'])
    df['DEATH_CROSS'] = CROSS(df['DEA'], df['DIF'])
    
    df['低位金叉'] = df['GOLDEN_CROSS'] & (df['DIF'] < -0.1)
    df['二次金叉'] = (df['GOLDEN_CROSS'] & 
                   (df['DEA'] < 0) & 
                   (df['金叉'].rolling(21).sum() == 2))
    
    # 趋势判断
    # 计算120和250日内MACD最大值
    df['MACD120_MAX'] = df['MACD'].rolling(120).max()
    df['MACD250_MAX'] = df['MACD'].rolling(250).max()
    
    # 计算MACD120和MACD250
    for i in range(len(df)):
        # 对于MACD120
        if i >= 120:
            window = df['MACD'].iloc[i-120:i+1]
            max_pos = window[window == window.max()].index[-1]  # 找到最近的最大值位置
            df.loc[df.index[i], 'MACD120'] = df.loc[max_pos, 'MACD'] / 2
        else:
            df.loc[df.index[i], 'MACD120'] = df.loc[df.index[i], 'MACD'] / 2
            
        # 对于MACD250
        if i >= 250:
            window = df['MACD'].iloc[i-250:i+1]
            max_pos = window[window == window.max()].index[-1]  # 找到最近的最大值位置
            df.loc[df.index[i], 'MACD250'] = df.loc[max_pos, 'MACD'] / 2
        else:
            df.loc[df.index[i], 'MACD250'] = df.loc[df.index[i], 'MACD'] / 2
    
    # XG信号和强势区判断
    df['XG'] = (df['MACD120'] != df['MACD120'].shift(1))
    df['强势区'] = (df['MACD'] >= df['MACD250'])
    
    # 主升判断
    df['主升'] = (df['XG'] & 
                (df['XG'] > df['XG'].shift(1)) & 
                df['强势区'] & 
                (df['强势区'] > df['强势区'].shift(1)))
    
    # 填充所有可能的NaN值
    df = df.fillna(0)
    
    return df

def plot_macd_system_new(df, stock_code):
    """
    绘制修改后的MACD指标系统图
    """
    # 设置中文字体 - 改进字体兼容性
    import matplotlib.font_manager as fm
    
    # 强制设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Hiragino Sans GB', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 尝试设置中文字体，按优先级尝试
    font_options = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Hiragino Sans GB', 'Arial Unicode MS', 'DejaVu Sans']
    font_found = False
    
    for font in font_options:
        try:
            test_font = fm.FontProperties(family=font)
            if test_font.get_name() != 'DejaVu Sans':
                font_found = True
                selected_font = font
                break
        except:
            continue
    
    # 如果没有找到中文字体，使用默认字体并添加字体回退
    if not font_found:
        selected_font = 'DejaVu Sans'
    
    # 设置暗色背景样式
    plt.style.use('dark_background')
    
    # 创建连续的数字索引，避免非交易日空缺
    x_index = range(len(df))
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), height_ratios=[1, 1.5])
    fig.patch.set_facecolor('#000000')  # 设置图表背景为黑色
    
    # 设置子图背景
    ax1.set_facecolor('#000000')
    ax2.set_facecolor('#000000')
    
    # 绘制K线图
    ax1.plot(x_index, df['close'], label='收盘价', color='#00FF00', linewidth=1)
    ax1.set_title(f'股票代码 {stock_code} 修改后MACD指标系统', fontsize=12, color='white', 
                  fontproperties=fm.FontProperties(family=selected_font))
    
    # 设置x轴标签为日期，但只显示部分日期以避免重叠
    date_labels = df.index.strftime('%y-%m')
    step = max(1, len(df) // 20)  # 最多显示20个日期标签
    ax1.set_xticks(x_index[::step])
    ax1.set_xticklabels(date_labels[::step], rotation=45)
    
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 1.02), ncol=2, facecolor='#333333')
    ax1.grid(True, color='#333333', linestyle='--', alpha=0.3)
    
    # 绘制MACD
    ax2.plot(x_index, df['DIF'], label='DIF线', color='white', linewidth=1)
    ax2.plot(x_index, df['DEA'], label='DEA线', color='yellow', linewidth=1)
    
    # 绘制零轴线
    ax2.axhline(y=0, color='#666666', linestyle='--', alpha=0.5, label='零轴线')
    
    # 绘制MACD柱状图
    colors = ['#FF4444' if x >= 0 else '#00FF00' for x in df['MACD']]
    ax2.bar(x_index, df['MACD'], color=colors, width=0.8, alpha=0.6)
    
    # 绘制背离柱线和标记
    for i, idx in enumerate(df.index):
        x_pos = x_index[i]  # 使用数字索引位置
        
        # DIF转折标记
        if df.loc[idx, 'DIF顶转折']:
            ax2.vlines(x_pos, df.loc[idx, 'DIF'], df.loc[idx, 'DIF']*1.3,
                      color='#00FF00', linewidth=2, alpha=0.8)
            ax2.annotate('Z', (x_pos, df.loc[idx, 'DIF']*1.2),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom', color='#00FF00')
        
        if df.loc[idx, 'DIF底转折']:
            ax2.vlines(x_pos, df.loc[idx, 'DIF'], df.loc[idx, 'DIF']*1.3,
                      color='yellow', linewidth=2, alpha=0.8)
            ax2.annotate('Z', (x_pos, df.loc[idx, 'DIF']*1.2),
                        xytext=(0, -10), textcoords='offset points',
                        ha='center', va='top', color='yellow')
        
        # 顶背离柱线（绿色）
        if df.loc[idx, '直接顶背离']:
            ax2.vlines(x_pos, df.loc[idx, 'DIF'], df.loc[idx, 'DEA'],
                      color='#00FF00', linewidth=2, alpha=0.8)
        
        # 隔峰顶背离柱线（青色）
        if df.loc[idx, '隔峰顶背离'] and not df.loc[idx, '直接顶背离']:
            ax2.vlines(x_pos, df.loc[idx, 'DIF'], df.loc[idx, 'DEA'],
                      color='#00FFFF', linewidth=2, alpha=0.8)
        
        # 底背离柱线（红色和粉色）
        if df.loc[idx, '直接底背离'] and df.loc[idx, '隔峰底背离'] and df.loc[idx, 'DIF'] < 0:
            ax2.vlines(x_pos, df.loc[idx, 'DIF'], df.loc[idx, 'DEA'],
                      color='#FF4444', linewidth=2, alpha=0.8)
        elif df.loc[idx, '直接底背离'] and not df.loc[idx, '隔峰底背离'] and df.loc[idx, 'DIF'] < 0:
            ax2.vlines(x_pos, df.loc[idx, 'DIF'], df.loc[idx, 'DEA'],
                      color='#FF4444', linewidth=2, alpha=0.8)
        elif not df.loc[idx, '直接底背离'] and df.loc[idx, '隔峰底背离'] and df.loc[idx, 'DIF'] < 0:
            ax2.vlines(x_pos, df.loc[idx, 'DIF'], df.loc[idx, 'DEA'],
                      color='#FF00FF', linewidth=2, alpha=0.8)
        
        # 顶结构白柱线
        if df.loc[idx, 'TG']:
            ax2.vlines(x_pos, 0, df.loc[idx, 'DIF'],
                      color='white', linewidth=2, alpha=0.8)
            # 标记顶结构类型
            if df.loc[idx, '直接TG'] and not df.loc[idx, '隔峰TG']:
                ax2.annotate('直接顶', (x_pos, df.loc[idx, 'DIF']*1.8),
                            xytext=(0, 10), textcoords='offset points',
                            ha='center', va='bottom', color='white')
            elif not df.loc[idx, '直接TG'] and df.loc[idx, '隔峰TG']:
                ax2.annotate('隔峰顶', (x_pos, df.loc[idx, 'DIF']*1.8),
                            xytext=(0, 10), textcoords='offset points',
                            ha='center', va='bottom', color='#00FFFF')
            elif df.loc[idx, '直接TG'] and df.loc[idx, '隔峰TG']:
                ax2.annotate('直隔顶', (x_pos, df.loc[idx, 'DIF']*1.8),
                            xytext=(0, 10), textcoords='offset points',
                            ha='center', va='bottom', color='#00FFFF')
        
        # 底结构白柱线
        if df.loc[idx, 'BG']:
            ax2.vlines(x_pos, 0, df.loc[idx, 'DIF'],
                      color='white', linewidth=2, alpha=0.8)
            # 标记底结构类型
            if df.loc[idx, '直接BG'] and df.loc[idx, '隔峰BG']:
                ax2.annotate('直隔底', (x_pos, df.loc[idx, 'DIF']*1.8),
                            xytext=(0, -10), textcoords='offset points',
                            ha='center', va='top', color='white')
            elif df.loc[idx, '直接BG'] and not df.loc[idx, '隔峰BG']:
                ax2.annotate('直接底', (x_pos, df.loc[idx, 'DIF']*1.8),
                            xytext=(0, -10), textcoords='offset points',
                            ha='center', va='top', color='gray')
            elif not df.loc[idx, '直接BG'] and df.loc[idx, '隔峰BG']:
                ax2.annotate('隔峰底', (x_pos, df.loc[idx, 'DIF']*1.8),
                            xytext=(0, -10), textcoords='offset points',
                            ha='center', va='top', color='yellow')
        
        # 背离消失标记
        '''
        if df.loc[idx, '直接顶背离消失']:
            ax2.annotate('消失1', (x_pos, df.loc[idx, 'DIF']*1.2),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom', color='white')
        elif df.loc[idx, '隔峰顶背离消失']:
            ax2.annotate('消失2', (x_pos, df.loc[idx, 'DIF']*1.5),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom', color='white')
        '''
        # 买卖信号标记
        if df.loc[idx, '低位金叉']:
            ax2.annotate('B', (x_pos, df.loc[idx, 'DIF']*1.4),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom', color='#FF4444')
        elif df.loc[idx, '二次金叉']:
            ax2.annotate('B2', (x_pos, df.loc[idx, 'DIF']*1.4),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom', color='#FF4444')
        
        # 主升段标记
        if df.loc[idx, '主升']:
            ax2.annotate('升', (x_pos, (df.loc[idx, 'DIF']+df.loc[idx, 'DEA'])/2),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom', color='#FF4444')
    
    # 设置图例
    ax2.legend(loc='upper left', bbox_to_anchor=(0, 1.15), ncol=4,
              title='修改后MACD指标系统说明', title_fontsize=10, facecolor='#333333')
    
    # 设置网格
    ax2.grid(True, color='#333333', linestyle='--', alpha=0.3)
    
    # 为下方图表也设置日期标签
    ax2.set_xticks(x_index[::step])
    ax2.set_xticklabels(date_labels[::step], rotation=45)
    
    # 设置坐标轴颜色
    for ax in [ax1, ax2]:
        ax.tick_params(axis='x', rotation=45, colors='white')
        ax.tick_params(axis='y', colors='white')
        for spine in ax.spines.values():
            spine.set_color('#666666')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(f'stock_{stock_code}_macd_chart_new.png', 
                bbox_inches='tight', 
                dpi=300, 
                facecolor='black',
                edgecolor='none')
    plt.close()

def main():
    stock_code = input("请输入股票代码（例如：sh000001）：")
    df = get_stock_data(stock_code)
    if df is None:
        return
    
    df = calculate_macd_indicators_new(df)
    
    # 选择需要输出到Excel的列
    columns_to_export = [
        'close',  # 收盘价
        'DIF',    # DIF线
        'DEA',    # DEA线
        'MACD',   # MACD柱
        'DIF顶转折', 'DIF底转折',  # DIF转折信号
        '金叉',   # 金叉信号
        'M1', 'M2', 'M3',  # 金叉周期
        'CH1', 'CH2', 'CH3',  # 高点价格    
        'DIFH1', 'DIFH2', 'DIFH3',  # 高点DIF值
        'PDIFH1', 'PDIFH2','PDIFH3', #取对数
        'MDIFH1', 'MDIFT2', 'MDIFT3',  # 标准化DIF（顶背离）
        'MDIFH2', 'MDIFH3',  # 标准化高点DIF
        '直接顶背离', '隔峰顶背离',  # 顶背离信号
        'T', # 顶背离信号
        '直接TG', '隔峰TG', 'TG', # 顶背离确认信号
        '顶钝化',
        '直接顶背离消失', '隔峰顶背离消失',
        '死叉',  # 死叉信号
        'N1', 'N2', 'N3',  # 死叉周期
        'CL1', 'CL2', 'CL3',  # 低点价格
        'DIFL1', 'DIFL2', 'DIFL3',  # 低点DIF值
        'PDIFL1', 'PDIFL2','PDIFL3', #取对数
        'MDIFL1', 'MDIFB2', 'MDIFB3',  # 标准化当前DIF（底背离）
        'MDIFL2', 'MDIFL3',  # 标准化低点DIF
        '直接底背离', '隔峰底背离',  # 底背离信号
        'B',  # 底背离信号
        '直接BG', '隔峰BG', 'BG',  # 底背离确认信号
        '底钝化',  # 钝化信号
        '直接底背离消失', '隔峰底背离消失',
        '主升' # 主升信号   
    ]
    
    # 确保所有列都存在
    existing_columns = [col for col in columns_to_export if col in df.columns]
    
    excel_path = f'stock_{stock_code}_macd_analysis_new.xlsx'
    df[existing_columns].to_excel(excel_path)
    print(f"数据已保存到: {excel_path}")
    
    plot_path = f'stock_{stock_code}_macd_chart_new.png'
    plot_macd_system_new(df, stock_code)
    print(f"图表已保存到: {plot_path}")

if __name__ == "__main__":
    main() 