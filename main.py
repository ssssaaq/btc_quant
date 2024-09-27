import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math

import talib as ta
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

def factor_selection_by_correlation(data, fct_list, corr_threshold): #相关系数
    X = data[fct_list[:-1]] 
    y = data['return']
    X_corr_matrix = X.corr()

    factor_list_1 = [i for i in X_corr_matrix.columns]
    factor_list_2 = [i for i in X_corr_matrix.columns]

    for i in range(0, len(factor_list_1), 1):
        fct_1 = factor_list_1[i]
        for j in range(0, i, 1):
            fct_2 = factor_list_1[j]
            corr_value = X_corr_matrix.iloc[i, j]
            if abs(corr_value) > corr_threshold:  
                corr_1 = np.corrcoef(X[fct_1], y)[0, 1]
                corr_2 = np.corrcoef(X[fct_2], y)[0, 1]
                if (abs(corr_1) < abs(corr_2)) and (fct_1 in factor_list_2):
                    factor_list_2.remove(fct_1) 
    
    return factor_list_2

def norm(x):
    factors_data = pd.DataFrame(x, columns=['factor'])
    factors_data = factors_data.replace([np.inf, -np.inf, np.nan], 0.0)
    factors_mean = factors_data.cumsum() / np.arange(1, factors_data.shape[0] + 1)[:, np.newaxis]
    factors_std = factors_data.expanding().std()
    factor_value = (factors_data - factors_mean) / factors_std
    factor_value = factor_value.replace([np.inf, -np.inf, np.nan], 0.0)
    factor_value = factor_value.clip(-6, 6)
    x = np.nan_to_num(factor_value['factor'].values)
    return x

def generate_etime_close_data_divd_time(bgn_date, end_date,code,frequency):
    # 读取数据
    read_file_path = 'D:\quant\crypto/1569754440000_1825.csv'
    kbars = pd.read_csv(read_file_path)
    kbars.rename(columns={'timestamp':'etime'},inplace=True)
    kbars['tdate'] = pd.to_datetime(kbars['etime']).dt.date 
    dt = pd.to_datetime(kbars['etime'], format='%Y-%m-%d %H:%M:%S')
    kbars['etime'] = pd.Series([pd.Timestamp(x).round('s').to_pydatetime() for x in dt])
    kbars['label'] = '-1'
    
    # 日期截取数据
    bgn_date = pd.to_datetime(bgn_date)
    end_date = pd.to_datetime(end_date)
    for i in range(0, len(kbars), 1): # .strftime('%Y-%m-%d %H:%M:%S')
        if (bgn_date <= kbars.loc[i, 'etime']) and (kbars.loc[i, 'etime'] <= end_date):
            kbars.loc[i, 'label'] = '1'

    # 筛选数据并重置索引
    kbars = kbars[kbars['label'] == '1']
    kbars = kbars.reset_index(drop=True)
    etime_close_data = kbars[['etime', 'tdate', 'close']]
    etime_close_data = etime_close_data.reset_index(drop=True)

    return etime_close_data

start_time = time.time()
file_path = 'D:\quant\crypto/1569754440000_1825.csv'
btcusdt = pd.read_csv(file_path).reset_index() # 读取源文件
btcusdt['timestamp'] = pd.to_datetime(btcusdt['timestamp'])
btcusdt = btcusdt.sort_values(by='timestamp', ascending=True) # 1.6s polars
btcusdt = btcusdt.set_index('timestamp')
btcusdt['return'] = btcusdt['close'].shift(-1)/btcusdt['close'] - 1
btcusdt = btcusdt.replace([np.nan], 0.0)

fct_value = pd.DataFrame() 

# 1、ma类
fct_value['ma5'] =  ta.MA(btcusdt['close'], timeperiod = 5 , matype = 0)
fct_value['ma10'] =  ta.MA(btcusdt['close'], timeperiod = 10 , matype = 0)
fct_value['ma20'] =  ta.MA(btcusdt['close'], timeperiod = 20 , matype = 0)
fct_value['ma5diff'] = fct_value['ma5']/btcusdt['close'] - 1
fct_value['ma10diff'] = fct_value['ma10']/btcusdt['close'] - 1
fct_value['ma20diff'] = fct_value['ma20']/btcusdt['close'] - 1 

# 2、bollinger band类
fct_value['h_line'], fct_value['m_line'], fct_value['l_line'] = ta.BBANDS(btcusdt['close'], timeperiod=20, nbdevup=2,nbdevdn=2,matype=0)
fct_value['stdevrate'] = (fct_value['h_line'] - fct_value['l_line']) / (btcusdt['close']*4)

# 3、sar因子
fct_value['sar_index'] = ta.SAR(btcusdt['high'], btcusdt['low'])
fct_value['sar_close'] = (fct_value['sar_index'] - btcusdt['close']) / btcusdt['close']

# 4、aroon
fct_value['aroon_index'] = ta.AROONOSC(btcusdt['high'], btcusdt['low'], timeperiod=14)

# 5、CCI
fct_value['cci_14'] = ta.CCI(btcusdt['close'], btcusdt['high'], btcusdt['low'], timeperiod=14)
fct_value['cci_25'] = ta.CCI(btcusdt['close'], btcusdt['high'], btcusdt['low'], timeperiod=25)
fct_value['cci_55'] = ta.CCI(btcusdt['close'], btcusdt['high'], btcusdt['low'], timeperiod=55)

# 6、CMO
fct_value['cmo_14'] = ta.CMO(btcusdt['close'], timeperiod=14)
fct_value['cmo_25'] = ta.CMO(btcusdt['close'], timeperiod=25)

# 7、MFI
fct_value['mfi_index'] = ta.MFI(btcusdt['high'], btcusdt['low'], btcusdt['close'], btcusdt['volume'])

# 8、MOM
fct_value['mom_14'] = ta.MOM(btcusdt['close'], timeperiod=14)
fct_value['mom_25'] = ta.MOM(btcusdt['close'], timeperiod=25)

# 9、
fct_value['index'] = ta.PPO(btcusdt['close'], fastperiod=12, slowperiod=26, matype=0)

# 10、AD
fct_value['ad_index'] = ta.AD(btcusdt['high'], btcusdt['low'], btcusdt['close'], btcusdt['volume'])
fct_value['ad_real'] = ta.ADOSC(btcusdt['high'], btcusdt['low'], btcusdt['close'], btcusdt['volume'], fastperiod=3, slowperiod=10)

# 11、OBV
fct_value['obv_index'] = ta.OBV(btcusdt['close'],btcusdt['volume'])

# 12、ATR
fct_value['atr_14'] = ta.ATR(btcusdt['high'], btcusdt['low'], btcusdt['close'], timeperiod=14)
fct_value['atr_25'] = ta.ATR(btcusdt['high'], btcusdt['low'], btcusdt['close'], timeperiod=25)
fct_value['atr_60'] = ta.ATR(btcusdt['high'], btcusdt['low'], btcusdt['close'], timeperiod=60)
fct_value['tr_index'] = ta.TRANGE(btcusdt['high'], btcusdt['low'], btcusdt['close'])
fct_value['tr_ma5'] = ta.MA(fct_value['tr_index'], timeperiod=5, matype = 0)/btcusdt['close']
fct_value['tr_ma10'] = ta.MA(fct_value['tr_index'], timeperiod=10, matype = 0)/btcusdt['close']
fct_value['tr_ma20'] = ta.MA(fct_value['tr_index'], timeperiod=20, matype = 0)/btcusdt['close']

# 13、KD
fct_value['kdj_k'], fct_value['kdj_d'] = ta.STOCH(btcusdt['high'], btcusdt['low'], btcusdt['close'], fastk_period=9, slowk_period=5, slowk_matype=1,slowd_period=5, slowd_matype=1)
fct_value['kdj_j'] = fct_value['kdj_k'] - fct_value['kdj_d']

# 14、MACD 
fct_value['macd_dif'],  fct_value['macd_dea'], fct_value['macd_hist'] = ta.MACD(btcusdt['close'], fastperiod=12, slowperiod=26, signalperiod=9)

# 15、RSI 
fct_value['rsi_6'] = ta.RSI(btcusdt['close'], timeperiod=6)
fct_value['rsi_12'] = ta.RSI(btcusdt['close'], timeperiod=12)
fct_value['rsi_25'] = ta.RSI(btcusdt['close'], timeperiod=25)
fct_value = fct_value.replace([np.nan], 0.0)



#因子标准化
factors_mean_2 = fct_value.cumsum() / np.arange(1, fct_value.shape[0] + 1)[:, np.newaxis]
factors_std_2 = fct_value.expanding().std()
factor_value = (fct_value - factors_mean_2) / factors_std_2 
factor_value = factor_value.replace([np.nan], 0.0)
factor_value = factor_value.clip(-6, 6)
factor_value.to_excel('btc1562737860000.xlsx',index=False)

#print(factor_value)
#print(factor_value.info())
#print(factor_value.describe())
#print(factor_value.skew())
#print(factor_value.kurt()) 

# 单位根检验
# from src.transforms.stationary_utils import check_unit_root
# res = check_unit_root(y_unit_root, confidence=0.05)
# print(f"Stationary: {res.stationary} | p-value: {res.results[1]}")

#相关性筛选因子
fct_file = factor_value #pd.read_csv('D:/quant/14/factor_base_0419.csv').set_index('timestamp')
fct_corr = fct_file.corr()
fct_file['return'] = (btcusdt['close'].shift(-1)/btcusdt['close'] - 1).values 
fct_file = fct_file.replace([np.nan], 0.0)
column_list = list(set(fct_file.columns) - set(['return']))
fct_corr_05 = factor_selection_by_correlation(fct_file, column_list, 0.4) #
#print(fct_corr_05)

fct_selected_list=fct_corr_05


# 训练集和测试集
# feed_data = factor_value[fct_corr_05]
feed_data = factor_value[fct_selected_list]
fct_in_use = fct_selected_list
feed_data['y'] = (btcusdt['close'].shift(-1)/btcusdt['close'] - 1).values
feed_data = feed_data.replace([np.nan], 0.0)
feed_data = feed_data.reset_index()
#print(feed_data)

train_set_end_index = feed_data[(feed_data['timestamp'].dt.year == 2024) & (feed_data['timestamp'].dt.month == 2) & (feed_data['timestamp'].dt.day == 10)  & (feed_data['timestamp'].dt.hour == 14) ].index.values[0]
X_train = feed_data[fct_in_use][ : train_set_end_index].values.reshape(-1, len(fct_in_use)) # X_train
y_train = feed_data['y'][ : train_set_end_index].values.reshape(-1, 1)
X_test = feed_data[fct_in_use][train_set_end_index : ].values.reshape(-1, len(fct_in_use))
y_test = feed_data['y'][train_set_end_index : ].values.reshape(-1, 1)

etime_train = feed_data['timestamp'][ : train_set_end_index].values 
etime_test = feed_data['timestamp'][train_set_end_index : ].values
etime_train_test = feed_data['timestamp'].values

# print(etime_train)
# print(etime_test)
# print(etime_train_test)

# 学习
#model = LinearRegression(fit_intercept=True) 
model=LGBMRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train) 

# 训练集预测
y_train_hat = model.predict(X_train)
#y_train_hat = [i[0] for i in y_train_hat]

# 测试集预测
y_test_hat = model.predict(X_test) 
# print(y_test_hat)
#y_test_hat = [i[0] for i in y_test_hat]
print(y_test_hat)

position_size = 1.0 #仓位
clip_num = 2 #杠杆倍数

#持仓净值（训练集）
begin_date_train = pd.to_datetime(str(etime_train[0])).strftime('%Y-%m-%d %H:%M:%S')
end_date_train = pd.to_datetime(str(etime_train[-1])).strftime('%Y-%m-%d %H:%M:%S')
ret_frame_train_total = generate_etime_close_data_divd_time(begin_date_train, end_date_train,'1','1')

start_index = ret_frame_train_total[ret_frame_train_total['etime'] == etime_train[0]].index.values[0]
end_index = ret_frame_train_total[ret_frame_train_total['etime'] == etime_train[-1]].index.values[0]
ret_frame_train_total = ret_frame_train_total.loc[start_index: end_index, :].reset_index(drop=True)  

ret_frame_train_total['position'] = [(i / 0.0005) * position_size for i in y_train_hat]  
ret_frame_train_total['position'] = ret_frame_train_total['position'].clip(-1*clip_num, clip_num) 
ret_frame_train = ret_frame_train_total
ret_frame_train.loc[0, '持仓净值'] = 1 

for i in range(0, len(ret_frame_train), 1):
    # 计算持仓净值
    if i == 0 or ret_frame_train.loc[i - 1, 'position'] == 0:  
        ret_frame_train.loc[i, '持仓净值'] = 1
    else:
        close_2 = ret_frame_train.loc[i, 'close']
        close_1 = ret_frame_train.loc[i - 1, 'close']
        position = abs(ret_frame_train.loc[i - 1, 'position'])  
        
        if ret_frame_train.loc[i - 1, 'position'] > 0: 
            ret_frame_train.loc[i, '持仓净值'] = 1 * (close_2 / close_1) * position + 1 * (1 - position)
        elif ret_frame_train.loc[i - 1, 'position'] < 0:  
            ret_frame_train.loc[i, '持仓净值'] = 1 * (1 - (close_2 / close_1 - 1)) * position + 1 * (1 - position)
            
#滚动测算累计持仓净值
ret_frame_train.loc[0, '持仓净值（累计）'] = 1
for i in range(1, len(ret_frame_train), 1):
    ret_frame_train.loc[i, '持仓净值（累计）'] = ret_frame_train.loc[i - 1, '持仓净值（累计）'] * ret_frame_train.loc[i, '持仓净值']

#持仓净值（测试集）

# 测算周期的起始日期和结束日期
begin_date_test = pd.to_datetime(str(etime_test[0])).strftime('%Y-%m-%d %H:%M:%S')
end_date_test = pd.to_datetime(str(etime_test[-1])).strftime('%Y-%m-%d %H:%M:%S')
ret_frame_test_total = generate_etime_close_data_divd_time(begin_date_test, end_date_test,'1','1')

# 初始化测算持仓净值的预备表格
start_index = ret_frame_test_total[ret_frame_test_total['etime'] == etime_test[0]].index.values[0]
end_index = ret_frame_test_total[ret_frame_test_total['etime'] == etime_test[-1]].index.values[0]


ret_frame_test_total = ret_frame_test_total.loc[start_index: end_index, :].reset_index(drop=True)  # 进一步根据起止时刻筛选数据
ret_frame_test_total['position'] = [(i / 0.0005) * position_size for i in y_test_hat]  
ret_frame_test_total['position'] = ret_frame_test_total['position'].clip(-1*clip_num, clip_num)
ret_frame_test = ret_frame_test_total
ret_frame_test = ret_frame_test.dropna(axis=0).reset_index(drop=True)  # 去除空值并重置索引

#===================== 1：初始化持仓净值 ==============================
ret_frame_test.loc[0, '持仓净值'] = 1

# 2：分周期测算持仓净值
for i in range(0, len(ret_frame_test), 1):
    # 计算持仓净值
    if i == 0 or ret_frame_test.loc[i - 1, 'position'] == 0:  # 如果是第一个时间步或前一个区间的结束时刻为空仓状态
        ret_frame_test.loc[i, '持仓净值'] = 1
    else:
        close_2 = ret_frame_test.loc[i, 'close']
        close_1 = ret_frame_test.loc[i - 1, 'close']
        position = abs(ret_frame_test.loc[i - 1, 'position'])  # 获取仓位大小（上一周期）

        if ret_frame_test.loc[i - 1, 'position'] > 0:  # 如果上一周期开的是多仓
            ret_frame_test.loc[i, '持仓净值'] = 1 * (close_2 / close_1) * position + 1 * (1 - position)
        elif ret_frame_test.loc[i - 1, 'position'] < 0:  # 如果上一周期开的是空仓
            ret_frame_test.loc[i, '持仓净值'] = 1 * (1 - (close_2 / close_1 - 1)) * position + 1 * (1 - position)

# 3：滚动测算累计持仓净值
ret_frame_test.loc[0, '持仓净值（累计）'] = 1
for i in range(1, len(ret_frame_test), 1):
    ret_frame_test.loc[i, '持仓净值（累计）'] = ret_frame_test.loc[i - 1, '持仓净值（累计）'] * ret_frame_test.loc[i, '持仓净值']

# -============  4：测算持仓净值（训练集 + 测试集）===================

# 测算周期的起始日期和结束日期
begin_date_train_test = pd.to_datetime(str(etime_train_test[0])).strftime('%Y-%m-%d %H:%M:%S')
end_date_train_test = pd.to_datetime(str(etime_train_test[-1])).strftime('%Y-%m-%d %H:%M:%S')
ret_frame_train_test_total = generate_etime_close_data_divd_time(begin_date_train_test, end_date_train_test, '510050', '15')

# 初始化测算持仓净值的预备表格
start_index = ret_frame_train_test_total[ret_frame_train_test_total['etime'] == etime_train_test[0]].index.values[0]
end_index = ret_frame_train_test_total[ret_frame_train_test_total['etime'] == etime_train_test[-1]].index.values[0]
ret_frame_train_test_total = ret_frame_train_test_total.loc[start_index: end_index, :].reset_index(drop=True)  # 进一步根据起止时刻筛选数据
ret_frame_train_test_total['position'] = [(i / 0.0005) * position_size for i in y_train_hat] + [(i / 0.0005) * position_size for i in y_test_hat]  # 训练值每间隔0.0005对应仓位变化1% + 预测值每间隔0.0005对应仓位变化1%
ret_frame_train_test_total['position'] = ret_frame_train_test_total['position'].clip(-1*clip_num, clip_num)
ret_frame_train_test = ret_frame_train_test_total
ret_frame_train_test = ret_frame_train_test.dropna(axis=0).reset_index(drop=True)  # 去除空值并重置索引

#================== 1：初始化持仓净值 =============================
ret_frame_train_test.loc[0, '持仓净值'] = 1

#================== 2：分周期测算持仓净值==========================
for i in range(0, len(ret_frame_train_test), 1):
    # 计算持仓净值
    if i == 0 or ret_frame_train_test.loc[i - 1, 'position'] == 0:  # 如果是第一个时间步或前一个区间的结束时刻为空仓状态
        ret_frame_train_test.loc[i, '持仓净值'] = 1
    else:
        close_2 = ret_frame_train_test.loc[i, 'close']
        close_1 = ret_frame_train_test.loc[i - 1, 'close']
        position = abs(ret_frame_train_test.loc[i - 1, 'position'])  # 获取仓位大小（上一周期）
        
        if ret_frame_train_test.loc[i - 1, 'position'] > 0:  # 如果上一周期开的是多仓
            ret_frame_train_test.loc[i, '持仓净值'] = 1 * (close_2 / close_1) * position + 1 * (1 - position)
        elif ret_frame_train_test.loc[i - 1, 'position'] < 0:  # 如果上一周期开的是空仓
            ret_frame_train_test.loc[i, '持仓净值'] = 1 * (1 - (close_2 / close_1 - 1)) * position + 1 * (1 - position)
# =======================3：滚动测算累计持仓净值===============================
ret_frame_train_test.loc[0, '持仓净值（累计）'] = 1
for i in range(1, len(ret_frame_train_test), 1):
    ret_frame_train_test.loc[i, '持仓净值（累计）'] = ret_frame_train_test.loc[i - 1, '持仓净值（累计）'] * ret_frame_train_test.loc[i, '持仓净值']

# ========================训练集验证集测试集数据统计完毕========================
#===================================================================================================================


#    ========================================================================================================================
#    PART 2：单因子风险指标测算
#    ========================================================================================================================

# 0：设置无风险利率和费用
fixed_return = 0.0
fees_rate = 0.004

# 1：初始化
indicators_frame = pd.DataFrame()
year_list = [i for i in ret_frame_train_test['etime'].dt.year.unique()]  # 获取年份列表
indicators_frame['年份'] = year_list + ['样本内', '样本外', '总体']
indicators_frame = indicators_frame.set_index('年份')  # 将年份列表设置为表格索引

# 2：计算风险指标（总体）
start_index = ret_frame_train_test.index[0]  # 获取总体的起始索引
end_index = ret_frame_train_test.index[-1]  # 获取总体的结束索引

# 1：总收益
net_value_2 = ret_frame_train_test.loc[end_index, '持仓净值（累计）']
net_value_1 = ret_frame_train_test.loc[start_index, '持仓净值（累计）']
total_return = net_value_2 / net_value_1 - 1
indicators_frame.loc['总体', '总收益'] = total_return

# 2：年化收益率
date_list = [i for i in ret_frame_train_test['etime'].dt.date.unique()]
run_day_length = len(date_list)  # 计算策略运行天数
annual_return = math.pow(1 + total_return, 252 / run_day_length) - 1

indicators_frame.loc['总体', '年化收益'] = annual_return

# 3：夏普比率、年化波动率
net_asset_value_list = []  # 初始化累计持仓净值列表（日度）
net_asset_value_index = [i for i in ret_frame_train_test.groupby(['tdate']).tail(1).index]  # 获取每日的结束索引

for date_index in net_asset_value_index:
    net_asset_value = ret_frame_train_test.loc[date_index, '持仓净值（累计）']
    net_asset_value_list.append(net_asset_value)  # 附加每日结束时对应的累计持仓净值

net_asset_value_frame = pd.DataFrame({'tdate': date_list, 'nav': net_asset_value_list})  # 构建日度累计持仓净值表格
net_asset_value_frame.loc[0, 'daily_log_return'] = 0  # 初始化对数收益率（日度）
for i in range(1, len(net_asset_value_frame), 1):
    net_asset_value_frame.loc[i, 'daily_log_return'] = math.log(net_asset_value_frame.loc[i, 'nav']) - math.log(net_asset_value_frame.loc[i - 1, 'nav'])  # 计算对数收益率（日度）
annual_volatility = math.sqrt(252) * net_asset_value_frame['daily_log_return'].std()  # 计算年化波动率
sharpe_ratio = (annual_return - fixed_return) / annual_volatility  # 计算夏普比率

indicators_frame.loc['总体', '年化波动率'] = annual_volatility
indicators_frame.loc['总体', '夏普比率'] = sharpe_ratio

# 4：最大回撤率及其对应的起止日（需要利用计算夏普比率过程中构建的日度累计持仓净值表格）
mdd_end_index = np.argmax((np.maximum.accumulate(net_asset_value_list) - net_asset_value_list) / (np.maximum.accumulate(net_asset_value_list)))
# if mdd_end_index == 0:return 0
mdd_end_date = net_asset_value_frame.loc[mdd_end_index, 'tdate']  # 最大回撤起始日
mdd_start_index = np.argmax(net_asset_value_list[: mdd_end_index])
mdd_start_date = net_asset_value_frame.loc[mdd_start_index, 'tdate']  # 最大回撤结束日
maximum_drawdown = (net_asset_value_list[mdd_start_index] - net_asset_value_list[mdd_end_index]) / (net_asset_value_list[mdd_start_index])  # 计算最大回撤率

indicators_frame.loc['总体', '最大回撤率'] = maximum_drawdown
indicators_frame.loc['总体', '最大回撤起始日'] = mdd_start_date
indicators_frame.loc['总体', '最大回撤结束日'] = mdd_end_date

# 5：卡尔玛比率（基于夏普比率以及最大回撤率）
calmar_ratio = (annual_return - fixed_return) / maximum_drawdown  # 计算卡尔玛比率

indicators_frame.loc['总体', '卡尔玛比率'] = calmar_ratio

# 6：总交易次数、交易胜率、交易盈亏比
total_trading_times = len(ret_frame_train_test)  # 计算总交易次数
win_times = 0  # 初始化盈利次数
win_lose_frame = pd.DataFrame()  # 初始化盈亏表格

for i in range(1, len(ret_frame_train_test), 1):
    delta_value = ret_frame_train_test.loc[i, '持仓净值（累计）'] - ret_frame_train_test.loc[i - 1, '持仓净值（累计）']  # 计算每次交易过程中累计持仓净值的变化量
    win_lose_frame.loc[i, 'delta_value'] = delta_value
    if delta_value > 0:
        win_times = win_times + 1

gain_amount = abs(win_lose_frame[win_lose_frame['delta_value'] > 0]['delta_value'].sum())  # 计算总盈利额
loss_amount = abs(win_lose_frame[win_lose_frame['delta_value'] < 0]['delta_value'].sum())  # 计算总亏损额
winning_rate = win_times / total_trading_times  # 计算胜率
gain_loss_ratio = gain_amount / loss_amount  # 计算盈亏比

indicators_frame.loc['总体', '总交易次数'] = total_trading_times
indicators_frame.loc['总体', '胜率'] = winning_rate
indicators_frame.loc['总体', '盈亏比'] = gain_loss_ratio


# 3：计算风险指标（分年度）
for year in year_list:
    data_demo = ret_frame_train_test[ret_frame_train_test['etime'].dt.year == year]  # 提取数据
    data_demo = data_demo.reset_index(drop=True)  # 重置索引
    data_demo['持仓净值（累计）'] = data_demo['持仓净值（累计）'] / data_demo.loc[0, '持仓净值（累计）']  # 缩放区间内部累计持仓净值

    start_index = data_demo.index[0]  # 获取当年的起始索引
    end_index = data_demo.index[-1]  # 获取当年的结束索引
    # 1：总收益
    net_value_2 = data_demo.loc[end_index, '持仓净值（累计）']
    net_value_1 = data_demo.loc[start_index, '持仓净值（累计）']
    total_return = net_value_2 / net_value_1 - 1

    indicators_frame.loc[year, '总收益'] = total_return

    # 2：年化收益率
    date_list = [i for i in data_demo['etime'].dt.date.unique()]
    run_day_length = len(date_list)  # 计算策略运行天数
    annual_return = math.pow(1 + total_return, 252 / run_day_length) - 1

    indicators_frame.loc[year, '年化收益'] = annual_return

    # 3：夏普比率、年化波动率
    net_asset_value_list = []  # 初始化累计持仓净值列表（日度）
    net_asset_value_index = [i for i in data_demo.groupby(['tdate']).tail(1).index]  # 获取每日的结束索引

    for date_index in net_asset_value_index:
        net_asset_value = data_demo.loc[date_index, '持仓净值（累计）']
        net_asset_value_list.append(net_asset_value)  # 附加每日结束时对应的累计持仓净值
    
    net_asset_value_frame = pd.DataFrame({'tdate': date_list, 'nav': net_asset_value_list})  # 构建日度累计持仓净值表格
    net_asset_value_frame.loc[0, 'daily_log_return'] = 0  # 初始化对数收益率（日度）

    for i in range(1, len(net_asset_value_frame), 1):
        net_asset_value_frame.loc[i, 'daily_log_return'] = math.log(net_asset_value_frame.loc[i, 'nav']) - math.log(net_asset_value_frame.loc[i - 1, 'nav'])  # 计算对数收益率（日度）
    
    annual_volatility = math.sqrt(252) * net_asset_value_frame['daily_log_return'].std()  # 计算年化波动率
    sharpe_ratio = (annual_return - fixed_return) / annual_volatility  # 计算夏普比率

    indicators_frame.loc[year, '年化波动率'] = annual_volatility
    indicators_frame.loc[year, '夏普比率'] = sharpe_ratio

    # 4：最大回撤率及其对应的起止日（需要利用计算夏普比率过程中构建的日度累计持仓净值表格）
    mdd_end_index = np.argmax((np.maximum.accumulate(net_asset_value_list) - net_asset_value_list) / (np.maximum.accumulate(net_asset_value_list)))
    # if mdd_end_index == 0:return 0
    mdd_end_date = net_asset_value_frame.loc[mdd_end_index, 'tdate']  # 最大回撤起始日
    mdd_start_index = np.argmax(net_asset_value_list[: mdd_end_index])
    mdd_start_date = net_asset_value_frame.loc[mdd_start_index, 'tdate']  # 最大回撤结束日
    maximum_drawdown = (net_asset_value_list[mdd_start_index] - net_asset_value_list[mdd_end_index]) / (net_asset_value_list[mdd_start_index])  # 计算最大回撤率

    indicators_frame.loc[year, '最大回撤率'] = maximum_drawdown
    indicators_frame.loc[year, '最大回撤起始日'] = mdd_start_date
    indicators_frame.loc[year, '最大回撤结束日'] = mdd_end_date

    # 5：卡尔玛比率（基于夏普比率以及最大回撤率）
    calmar_ratio = (annual_return - fixed_return) / maximum_drawdown  # 计算卡尔玛比率

    indicators_frame.loc[year, '卡尔玛比率'] = calmar_ratio

    # 6：总交易次数、交易胜率、交易盈亏比
    total_trading_times = len(data_demo)  # 计算总交易次数
    
    win_times = 0  # 初始化盈利次数
    win_lose_frame = pd.DataFrame()  # 初始化盈亏表格
    
    for i in range(1, len(data_demo), 1):
        delta_value =  data_demo.loc[i, '持仓净值（累计）'] - data_demo.loc[i - 1, '持仓净值（累计）']  # 计算每次交易过程中累计持仓净值的变化量
        win_lose_frame.loc[i, 'delta_value'] = delta_value
        if delta_value > 0:
            win_times = win_times + 1
    
    gain_amount = abs(win_lose_frame[win_lose_frame['delta_value'] > 0]['delta_value'].sum())  # 计算总盈利额
    loss_amount = abs(win_lose_frame[win_lose_frame['delta_value'] < 0]['delta_value'].sum())  # 计算总亏损额

    winning_rate = win_times / total_trading_times  # 计算胜率
    gain_loss_ratio = gain_amount / loss_amount  # 计算盈亏比

    indicators_frame.loc[year, '总交易次数'] = total_trading_times
    indicators_frame.loc[year, '胜率'] = winning_rate
    indicators_frame.loc[year, '盈亏比'] = gain_loss_ratio

# -=====================4：计算风险指标（样本内）=======================================
start_index = ret_frame_train.index[0]  # 获取训练集的起始索引
end_index = ret_frame_train.index[-1]  # 获取训练集的结束索引

# 1：总收益
net_value_2 = ret_frame_train.loc[end_index, '持仓净值（累计）']
net_value_1 = ret_frame_train.loc[start_index, '持仓净值（累计）']
total_return = net_value_2 / net_value_1 - 1

indicators_frame.loc['样本内', '总收益'] = total_return

# 2：年化收益率
date_list = [i for i in ret_frame_train['etime'].dt.date.unique()]
run_day_length = len(date_list)  # 计算策略运行天数
annual_return = math.pow(1 + total_return, 252 / run_day_length) - 1

indicators_frame.loc['样本内', '年化收益'] = annual_return

# 3：夏普比率、年化波动率
net_asset_value_list = []  # 初始化累计持仓净值列表（日度）
net_asset_value_index = [i for i in ret_frame_train.groupby(['tdate']).tail(1).index]  # 获取每日的结束索引

for date_index in net_asset_value_index:
    net_asset_value = ret_frame_train.loc[date_index, '持仓净值（累计）']
    net_asset_value_list.append(net_asset_value)  # 附加每日结束时对应的累计持仓净值

net_asset_value_frame = pd.DataFrame({'tdate': date_list, 'nav': net_asset_value_list})  # 构建日度累计持仓净值表格
net_asset_value_frame.loc[0, 'daily_log_return'] = 0  # 初始化对数收益率（日度）

for i in range(1, len(net_asset_value_frame), 1):
    net_asset_value_frame.loc[i, 'daily_log_return'] = math.log(net_asset_value_frame.loc[i, 'nav']) - math.log(net_asset_value_frame.loc[i - 1, 'nav'])  # 计算对数收益率（日度）

annual_volatility = math.sqrt(252) * net_asset_value_frame['daily_log_return'].std()  # 计算年化波动率
sharpe_ratio = (annual_return - fixed_return) / annual_volatility  # 计算夏普比率

indicators_frame.loc['样本内', '年化波动率'] = annual_volatility
indicators_frame.loc['样本内', '夏普比率'] = sharpe_ratio

# 4：最大回撤率及其对应的起止日（需要利用计算夏普比率过程中构建的日度累计持仓净值表格）
mdd_end_index = np.argmax((np.maximum.accumulate(net_asset_value_list) - net_asset_value_list) / (np.maximum.accumulate(net_asset_value_list)))
# if mdd_end_index == 0:return 0
mdd_end_date = net_asset_value_frame.loc[mdd_end_index, 'tdate']  # 最大回撤起始日
mdd_start_index = np.argmax(net_asset_value_list[: mdd_end_index])
mdd_start_date = net_asset_value_frame.loc[mdd_start_index, 'tdate']  # 最大回撤结束日
maximum_drawdown = (net_asset_value_list[mdd_start_index] - net_asset_value_list[mdd_end_index]) / (net_asset_value_list[mdd_start_index])  # 计算最大回撤率

indicators_frame.loc['样本内', '最大回撤率'] = maximum_drawdown
indicators_frame.loc['样本内', '最大回撤起始日'] = mdd_start_date
indicators_frame.loc['样本内', '最大回撤结束日'] = mdd_end_date

# 5：卡尔玛比率（基于夏普比率以及最大回撤率）
calmar_ratio = (annual_return - fixed_return) / maximum_drawdown  # 计算卡尔玛比率

indicators_frame.loc['样本内', '卡尔玛比率'] = calmar_ratio

# 6：总交易次数、交易胜率、交易盈亏比
total_trading_times = len(ret_frame_train)  # 计算总交易次数
win_times = 0  # 初始化盈利次数
win_lose_frame = pd.DataFrame()  # 初始化盈亏表格

for i in range(1, len(ret_frame_train), 1):
    delta_value = ret_frame_train.loc[i, '持仓净值（累计）'] - ret_frame_train.loc[i - 1, '持仓净值（累计）']  # 计算每次交易过程中累计持仓净值的变化量
    win_lose_frame.loc[i, 'delta_value'] = delta_value
    if delta_value > 0:
        win_times = win_times + 1

gain_amount = abs(win_lose_frame[win_lose_frame['delta_value'] > 0]['delta_value'].sum())  # 计算总盈利额
loss_amount = abs(win_lose_frame[win_lose_frame['delta_value'] < 0]['delta_value'].sum())  # 计算总亏损额
winning_rate = win_times / total_trading_times  # 计算胜率
gain_loss_ratio = gain_amount / loss_amount  # 计算盈亏比

indicators_frame.loc['样本内', '总交易次数'] = total_trading_times
indicators_frame.loc['样本内', '胜率'] = winning_rate
indicators_frame.loc['样本内', '盈亏比'] = gain_loss_ratio

#==========================5：计算风险指标（样本外）===========================
start_index = ret_frame_test.index[0]  # 获取测试集的起始索引
end_index = ret_frame_test.index[-1]  # 获取测试集的结束索引

# 1：总收益
net_value_2 = ret_frame_test.loc[end_index, '持仓净值（累计）']
net_value_1 = ret_frame_test.loc[start_index, '持仓净值（累计）']
total_return = net_value_2 / net_value_1 - 1

indicators_frame.loc['样本外', '总收益'] = total_return

# 2：年化收益率
date_list = [i for i in ret_frame_test['etime'].dt.date.unique()]
run_day_length = len(date_list)  # 计算策略运行天数
annual_return = math.pow(1 + total_return, 252 / run_day_length) - 1

indicators_frame.loc['样本外', '年化收益'] = annual_return

# 3：夏普比率、年化波动率
net_asset_value_list = []  # 初始化累计持仓净值列表（日度）
net_asset_value_index = [i for i in ret_frame_test.groupby(['tdate']).tail(1).index]  # 获取每日的结束索引

for date_index in net_asset_value_index:
    net_asset_value = ret_frame_test.loc[date_index, '持仓净值（累计）']
    net_asset_value_list.append(net_asset_value)  # 附加每日结束时对应的累计持仓净值

net_asset_value_frame = pd.DataFrame({'tdate': date_list, 'nav': net_asset_value_list})  # 构建日度累计持仓净值表格
net_asset_value_frame.loc[0, 'daily_log_return'] = 0  # 初始化对数收益率（日度）

for i in range(1, len(net_asset_value_frame), 1):
    net_asset_value_frame.loc[i, 'daily_log_return'] = math.log(net_asset_value_frame.loc[i, 'nav']) - math.log(net_asset_value_frame.loc[i - 1, 'nav'])  # 计算对数收益率（日度）

annual_volatility = math.sqrt(252) * net_asset_value_frame['daily_log_return'].std()  # 计算年化波动率
sharpe_ratio = (annual_return - fixed_return) / annual_volatility  # 计算夏普比率

indicators_frame.loc['样本外', '年化波动率'] = annual_volatility
indicators_frame.loc['样本外', '夏普比率'] = sharpe_ratio

# 4：最大回撤率及其对应的起止日（需要利用计算夏普比率过程中构建的日度累计持仓净值表格）
mdd_end_index = np.argmax((np.maximum.accumulate(net_asset_value_list) - net_asset_value_list) / (np.maximum.accumulate(net_asset_value_list)))
# if mdd_end_index == 0:return 0
mdd_end_date = net_asset_value_frame.loc[mdd_end_index, 'tdate']  # 最大回撤起始日
mdd_start_index = np.argmax(net_asset_value_list[: mdd_end_index])
mdd_start_date = net_asset_value_frame.loc[mdd_start_index, 'tdate']  # 最大回撤结束日
maximum_drawdown = (net_asset_value_list[mdd_start_index] - net_asset_value_list[mdd_end_index]) / (net_asset_value_list[mdd_start_index])  # 计算最大回撤率

indicators_frame.loc['样本外', '最大回撤率'] = maximum_drawdown
indicators_frame.loc['样本外', '最大回撤起始日'] = mdd_start_date
indicators_frame.loc['样本外', '最大回撤结束日'] = mdd_end_date

# 5：卡尔玛比率（基于夏普比率以及最大回撤率）
calmar_ratio = (annual_return - fixed_return) / maximum_drawdown  # 计算卡尔玛比率

indicators_frame.loc['样本外', '卡尔玛比率'] = calmar_ratio

# 6：总交易次数、交易胜率、交易盈亏比
total_trading_times = len(ret_frame_test)  # 计算总交易次数

win_times = 0  # 初始化盈利次数
win_lose_frame = pd.DataFrame()  # 初始化盈亏表格

for i in range(1, len(ret_frame_test), 1):
    delta_value = ret_frame_test.loc[i, '持仓净值（累计）'] - ret_frame_test.loc[i - 1, '持仓净值（累计）']  # 计算每次交易过程中累计持仓净值的变化量
    win_lose_frame.loc[i, 'delta_value'] = delta_value
    if delta_value > 0:
        win_times = win_times + 1

gain_amount = abs(win_lose_frame[win_lose_frame['delta_value'] > 0]['delta_value'].sum())  # 计算总盈利额
loss_amount = abs(win_lose_frame[win_lose_frame['delta_value'] < 0]['delta_value'].sum())  # 计算总亏损额
winning_rate = win_times / total_trading_times  # 计算胜率
gain_loss_ratio = gain_amount / loss_amount  # 计算盈亏比

indicators_frame.loc['样本外', '总交易次数'] = total_trading_times
indicators_frame.loc['样本外', '胜率'] = winning_rate
indicators_frame.loc['样本外', '盈亏比'] = gain_loss_ratio

print(indicators_frame)

plot_output = ret_frame_test['持仓净值（累计）']
close_output = ret_frame_test['close']/ret_frame_test['close'].iloc[0]

plot_output.index = etime_test
close_output.index = etime_test

plt.figure(figsize=(8,6))

# 绘制持仓净值（累计）
plt.plot(plot_output, 'b-', label='持仓净值（累计）')

# 绘制 close 值
plt.plot(close_output, 'r-', label='Close 值')

# 添加图例、网格、标签等
plt.legend()
plt.grid()
plt.xlabel('Model_output')
plt.ylabel('Return on lvg')

# 显示图形
plt.show()

end_time = time.time()
print('time cost:      ', end_time-start_time)
