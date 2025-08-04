import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import pytz
from data_fetcher import get_index_stocks, batch_get_stock_data
from strategy_calculator import calculate_macd_indicators, calculate_signal_returns

# 定义指数映射
INDEX_MAPPING = {
    "沪深300": "000300",
    "中证500": "000905"
}

def analyze_stocks(stock_data_dict):
    """
    分析所有股票数据并返回满足条件的结果
    """
    results = []
    
    for stock_code, stock_info in stock_data_dict.items():
        df = stock_info['data']
        stock_name = stock_info['name']
        
        # 计算指标
        df = calculate_macd_indicators(df)
        
        # 计算信号收益率
        returns = calculate_signal_returns(df)
        
        # 检查是否有顶底结构信号
        has_top_signal = df['顶结构'].any()
        has_bottom_signal = df['底结构'].any()
        
        if has_top_signal or has_bottom_signal:
            result = {
                '股票代码': stock_code,
                '股票名称': stock_name,
                '信号类型': []
            }
            
            # 处理顶结构信号
            if has_top_signal and 'last_top' in returns:
                result['信号类型'].append('顶结构')
                result['最近顶结构日期'] = df[df['顶结构']].index[-1].strftime('%Y-%m-%d')
                result['顶结构3日涨跌幅'] = f"{returns['last_top']['3d']:.2f}%" if returns['last_top']['3d'] is not None else "无数据"
                result['顶结构5日涨跌幅'] = f"{returns['last_top']['5d']:.2f}%" if returns['last_top']['5d'] is not None else "无数据"
                result['顶结构10日涨跌幅'] = f"{returns['last_top']['10d']:.2f}%" if returns['last_top']['10d'] is not None else "无数据"
            
            # 处理底结构信号
            if has_bottom_signal and 'last_bottom' in returns:
                result['信号类型'].append('底结构')
                result['最近底结构日期'] = df[df['底结构']].index[-1].strftime('%Y-%m-%d')
                result['底结构3日涨跌幅'] = f"{returns['last_bottom']['3d']:.2f}%" if returns['last_bottom']['3d'] is not None else "无数据"
                result['底结构5日涨跌幅'] = f"{returns['last_bottom']['5d']:.2f}%" if returns['last_bottom']['5d'] is not None else "无数据"
                result['底结构10日涨跌幅'] = f"{returns['last_bottom']['10d']:.2f}%" if returns['last_bottom']['10d'] is not None else "无数据"
            
            result['信号类型'] = ', '.join(result['信号类型'])
            results.append(result)
    
    return pd.DataFrame(results)

def main():
    st.title('股票MACD策略分析系统')
    
    # 添加自动刷新选项
    st.sidebar.header('刷新设置')
    
    # 使用session_state存储自动刷新状态
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False
    if 'refresh_interval' not in st.session_state:
        st.session_state.refresh_interval = 15
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now(pytz.timezone("Asia/Shanghai"))
    
    # 更新自动刷新状态
    st.session_state.auto_refresh = st.sidebar.checkbox('启用自动刷新', 
                                                       value=st.session_state.auto_refresh)
    
    if st.session_state.auto_refresh:
        st.session_state.refresh_interval = st.sidebar.slider('刷新间隔（分钟）', 
                                                            min_value=5, 
                                                            max_value=60, 
                                                            value=st.session_state.refresh_interval)
        
        # 显示下次刷新时间
        next_refresh = st.session_state.last_refresh + timedelta(minutes=st.session_state.refresh_interval)
        st.sidebar.info(f'下次刷新时间: {next_refresh.strftime("%H:%M:%S")}')
    
    # 显示最后更新时间
    placeholder = st.empty()
    with placeholder.container():
        st.info(f'最后更新时间: {datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")}')
    
    # 添加手动刷新按钮
    if st.sidebar.button('立即刷新数据'):
        st.cache_data.clear()
        st.experimental_rerun()
    
    # 侧边栏 - 时间范围选择
    st.sidebar.header('参数设置')
    time_range = st.sidebar.radio(
        "选择时间范围",
        ('近一年', '近两年')
    )
    
    # 计算日期范围
    beijing_tz = pytz.timezone('Asia/Shanghai')
    end_date = datetime.now(beijing_tz)
    if time_range == '近一年':
        start_date = end_date - timedelta(days=365)
    else:  # 近两年
        start_date = end_date - timedelta(days=730)
    
    # 转换为字符串格式
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    
    # 选择指数
    st.sidebar.header('指数选择')
    selected_index = st.sidebar.selectbox(
        '选择要分析的指数',
        list(INDEX_MAPPING.keys()),
        index=0
    )
    
    # 获取选中指数的成分股
    with st.spinner(f'正在获取{selected_index}成分股列表...'):
        index_stocks = get_index_stocks(INDEX_MAPPING[selected_index])
        
        if not index_stocks.empty:
            st.success(f'成功获取{selected_index}成分股列表，共 {len(index_stocks)} 只股票')
            
            # 获取所有股票数据
            with st.spinner('正在获取并分析所有股票数据...'):
                stock_data_dict = batch_get_stock_data(index_stocks, start_date, end_date)
                
                if stock_data_dict:
                    # 分析所有股票数据
                    results_df = analyze_stocks(stock_data_dict)
                    
                    if not results_df.empty:
                        st.success(f'分析完成，发现 {len(results_df)} 只股票出现信号')
                        
                        # 分别展示顶结构和底结构的股票
                        st.subheader('顶结构信号股票')
                        top_signals = results_df[results_df['信号类型'].str.contains('顶结构', na=False)]
                        if not top_signals.empty:
                            st.dataframe(top_signals[[
                                '股票代码', '股票名称', '最近顶结构日期',
                                '顶结构3日涨跌幅', '顶结构5日涨跌幅', '顶结构10日涨跌幅'
                            ]])
                        else:
                            st.write('没有股票出现顶结构信号')
                        
                        st.subheader('底结构信号股票')
                        bottom_signals = results_df[results_df['信号类型'].str.contains('底结构', na=False)]
                        if not bottom_signals.empty:
                            st.dataframe(bottom_signals[[
                                '股票代码', '股票名称', '最近底结构日期',
                                '底结构3日涨跌幅', '底结构5日涨跌幅', '底结构10日涨跌幅'
                            ]])
                        else:
                            st.write('没有股票出现底结构信号')
                    else:
                        st.warning('没有发现满足条件的股票')
                else:
                    st.error('获取股票数据失败')
        else:
            st.error('获取沪深300成分股列表失败')

def auto_refresh_data():
    """自动刷新数据的函数"""
    # 清除缓存
    st.cache_data.clear()
    # 重新运行应用
    st.experimental_rerun()

if __name__ == "__main__":
    main()
    
    # 如果启用了自动刷新，设置定时器
    if 'auto_refresh' in st.session_state and st.session_state.auto_refresh:
        # 获取当前时间
        current_time = datetime.now(pytz.timezone("Asia/Shanghai"))
        
        # 如果session_state中没有上次更新时间，或者已经超过刷新间隔
        if ('last_refresh' not in st.session_state or 
            (current_time - st.session_state.last_refresh).total_seconds() >= 
            st.session_state.refresh_interval * 60):
            
            # 更新上次刷新时间
            st.session_state.last_refresh = current_time
            # 执行刷新
            auto_refresh_data()