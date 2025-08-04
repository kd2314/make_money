# 沪深300 MACD策略分析系统

基于 MACD 指标的沪深300成分股分析系统，用于自动识别顶底结构信号。

## 功能特点

- 自动获取沪深300成分股数据
- MACD 顶底结构信号识别
- 信号后 3/5/10 日涨跌幅追踪
- 支持近一年/两年数据分析
- 实时数据更新

## 在线使用

访问 Streamlit 应用：[应用链接]

## 本地运行

1. 克隆仓库：
```bash
git clone [你的仓库地址]
cd [仓库名]
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 运行应用：
```bash
streamlit run app.py
```

## 技术栈

- Python
- Streamlit
- AKShare
- Pandas
- NumPy

## 注意事项

- 数据来源于 AKShare，请确保网络连接正常
- 首次运行可能需要较长时间获取数据
- 建议使用 Python 3.8 或更高版本