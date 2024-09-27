import requests
import time
import pandas as pd

base_url='https://api.binance.com/api/v1/klines'
symbol='BTCUSDT'
interval='1h'
limit='1000'
# kline_url=base_url+'?'+'symbol='+symbol+'&'+'interval='+interval+'&'+'limit='+limit
# data=requests.get(kline_url)
# df=pd.DataFrame(data.json())
# df.columns=['open_time','open','high','low','close','volume','close_time','base_ass_volume','num_trades','taker_buy_volume','base_taker','ignore']
# print(df)

end_time=int(time.time()//60*60*1000)
start_time=int(end_time-1000*60*60*1000)
print(start_time)

days=365*5 #获取数据时长（天）
start=end_time-days*24*60*60*1000
raw_data=pd.DataFrame() #存放总数据
while start_time>start:
    kline_url=base_url+'?'+'symbol='+symbol+'&'+'interval='+interval+'&'+'startTime='+str(start_time)+'&endTime='+str(end_time)+'&limit='+limit
    data=requests.get(kline_url)
    df=pd.DataFrame(data.json())
    df.columns=['open_time','open','high','low','close','volume','close_time','base_ass_volume','num_trades','taker_buy_volume','base_taker','ignore']
    raw_data=pd.concat([df,raw_data],ignore_index=True)
    start_time-=1000*60*60*1000
    end_time-=1000*60*60*1000

raw_data['timestamp']=pd.to_datetime(raw_data['open_time'],unit='ms',origin=pd.Timestamp('1970-01-01'))
raw_data.set_index('timestamp',inplace=True)

raw_data.to_csv(str(start)+'_'+str(days)+'.csv') #无特定时区时间