import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
import pytz
import warnings
import time
from tqdm import tqdm
import streamlit as st
warnings.filterwarnings('ignore')

def get_stock_data(stock_code, start_date=None, end_date=None):
    """
    获取股票数据
    
    Parameters:
    -----------
    stock_code : str
        股票代码
    start_date : str, optional
        开始日期，格式：'YYYY-MM-DD'
    end_date : str, optional
        结束日期，格式：'YYYY-MM-DD'
    
    Returns:
    --------
    pd.DataFrame or None
        包含股票数据的DataFrame，如果获取失败则返回None
    """
    try:
        # 设置默认时间范围
        beijing_tz = pytz.timezone('Asia/Shanghai')
        now = datetime.now(beijing_tz)
        
        if end_date is None:
            end_date = now.strftime('%Y-%m-%d')
        if start_date is None:
            start_date = '2020-01-01'
        
        # 使用akshare获取指数数据
        df = ak.stock_zh_index_daily(symbol=stock_code)
        
        if df is None or df.empty:
            print(f"无法获取股票 {stock_code} 的数据")
            return None
        
        # 确保date列存在并转换为日期格式
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
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

@st.cache_data(ttl=3600)  # 缓存1小时
def get_index_stocks(index_type="000300"):
    """
    获取指数成分股列表
    
    Parameters:
    -----------
    index_type : str
        指数代码，'000300'为沪深300，'000905'为中证500
    
    Returns:
    --------
    pd.DataFrame
        包含股票代码和名称的DataFrame
    """
    try:
        # 获取指数成分股
        index_stocks = ak.index_stock_cons_weight_csindex(symbol=index_type)
        # 转换股票代码格式
        index_stocks['stock_code'] = index_stocks['成分券代码'].apply(
            lambda x: f"sh{x}" if x.startswith('6') else f"sz{x}"
        )
        
        # 获取指数名称
        index_name = "沪深300" if index_type == "000300" else "中证500"
        print(f"成功获取{index_name}成分股列表")
        
        return index_stocks[['stock_code', '成分券名称']].rename(columns={'成分券名称': 'stock_name'})
    except Exception as e:
        print(f"获取指数成分股失败: {e}")
        return pd.DataFrame(columns=['stock_code', 'stock_name'])

def get_stock_name(stock_code):
    """
    获取股票名称
    
    Parameters:
    -----------
    stock_code : str
        股票代码
    
    Returns:
    --------
    str
        股票名称
    """
    try:
        # 使用akshare获取股票名称
        stock_info = ak.stock_info_a_code_name()
        stock_code_clean = stock_code.replace('sh', '').replace('sz', '')
        stock_name = stock_info[stock_info['code'] == stock_code_clean]['name'].values[0]
        return stock_name
    except:
        return stock_code  # 如果获取失败，返回股票代码

@st.cache_data(ttl=3600)  # 缓存1小时
def batch_get_stock_data(stock_list, start_date=None, end_date=None):
    """
    批量获取股票数据
    
    Parameters:
    -----------
    stock_list : pd.DataFrame
        包含股票代码和名称的DataFrame
    start_date : str, optional
        开始日期，格式：'YYYY-MM-DD'
    end_date : str, optional
        结束日期，格式：'YYYY-MM-DD'
    
    Returns:
    --------
    dict
        股票数据字典，key为股票代码，value为DataFrame
    """
    results = {}
    for _, row in tqdm(stock_list.iterrows(), total=len(stock_list), desc="获取股票数据"):
        stock_code = row['stock_code']
        try:
            df = get_stock_data(stock_code, start_date, end_date)
            if df is not None and not df.empty:
                results[stock_code] = {
                    'data': df,
                    'name': row['stock_name']
                }
            time.sleep(0.5)  # 添加延时避免请求过快
        except Exception as e:
            print(f"获取股票 {stock_code} 数据失败: {e}")
            continue
    return results