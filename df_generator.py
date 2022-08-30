import os
import sys
import wget
import pandas as pd
import numpy as np
import time
import datetime
from statistics import mean, stdev
import zipfile
from talib import RSI, ULTOSC, ADX, MINUS_DI, PLUS_DI, MFI, HT_TRENDLINE, MIDPOINT, BBANDS, MACD, AD, OBV, ADOSC, APO, AROONOSC, STOCHRSI, CMO, BOP, TRANGE, SAR, PPO, WILLR, ROC, MAMA, ATR, TRIMA
import ta
from sklearn import preprocessing

from get_data import get_data

def check_df(seconds, dir, itv, tckr, futures):
  try:
    if 'google.colab' in sys.modules:
      full_path='/content/drive/MyDrive/RLtrader'+dir+itv+'_data/'+tckr+'/'+tckr+'.csv'
    else:
      full_path=os.getcwd()+dir+itv+'_data/'+tckr+'/'+tckr+'.csv'
    tckr_df = pd.read_csv(full_path)
    #tckr_df['Opened'] = pd.to_datetime(tckr_df['Opened'])
    timestamp_last = pd.to_datetime(tckr_df.iloc[-1]['Opened']).value // 10 ** 9
    diff=time.time()-timestamp_last
    print(f' avg gain/loss data is of by {diff} timestamps')
    # checking how many seconds old the local df is
    if diff>seconds:
      raise FileNotFoundError("Out of date dataframe (1day)")
    return tckr_df
  except FileNotFoundError:
    return get_data(ticker=tckr, interval=itv, futures=futures)

def get_levels(ticker='BTCUSDT', interval='1m',  futures=True, decimals=0, current_price_offset=15,volume_above_avg=False):
  ### get data ###
  if futures:
    directory='/data/binance_data_futures/'
  else:
    directory='/data/binance_data_spot/'
  tckr_df = check_df(86400, directory, interval, ticker, futures)

  ### searchin for lvls ###
  vol_array=tckr_df['Volume'][1:].to_numpy()
  avg_vol=np.mean(vol_array)
  tckr_lvls={}
  for open, close, volume, close_next in zip(tckr_df['Open'][:-1].to_numpy(), tckr_df['Close'][:-1].to_numpy(), vol_array, tckr_df['Close'][1:].to_numpy()):
    #if (not volume_above_avg or volume>=avg_vol) and band_top>close>band_bottom:
    if (close>open>close_next) or (close<open<close_next):
      close=round(close, decimals)
      if close in tckr_lvls:
        tckr_lvls[close]+=1
      else:
        tckr_lvls[close]=1
      if volume>=avg_vol:
        tckr_lvls[close]+=1
      #print(row)
  tckr_lvls={k: v for k, v in sorted(tckr_lvls.items(), key=lambda item: item[1], reverse=True)}
  #print(tckr_lvls)
  return tckr_lvls


def get_avg_changes(ticker='BTCUSDT', interval='1m', futures=True):
  ### get data ###
  if futures:
    directory='/data/binance_data_futures/'
  else:
    directory='/data/binance_data_spot/'
  tckr_df = check_df(86400, directory, interval, ticker, futures)

  open_array=tckr_df['Open'].to_numpy()
  close_array=tckr_df['Close'].to_numpy()
  #low_array=tckr_df['Low'].to_numpy()
  #high_array=tckr_df['High'].to_numpy()
  #vol_array=tckr_df['Volume'].to_numpy()

  gain = [ (close/open-1)*100 for open, close in zip(open_array, close_array) if close>open]
  loss = [ (open/close-1)*100 for open, close in zip(open_array, close_array) if close<open]
  #HL_distance = [ (high/low-1)*100 for low, high in zip(low_array, high_array)]

  #avg_vol=np.mean(vol_array)
  #stdev_vol=np.std(vol_array)
  #print(f'interval: {interval} avg_gain: {mean(gain)} stdev: {stdev(gain)} avg_loss: {mean(loss)} stdev: {stdev(loss)}')
  #print(f'avg_volume: {avg_vol} stdev: {stdev_vol} avg_High_Low_distance: {mean(HL_distance)} stdev: {stdev(HL_distance)}')
  return mean(gain),stdev(gain),mean(loss),stdev(loss)

def get_weekday_changes(ticker='BTCUSDT', interval='1m',  futures=True):
    ### get data ###
    if futures:
      directory='/data/binance_data_futures/'
    else:
      directory='/data/binance_data_spot/'
    tckr_df = check_df(86400, directory, interval, ticker, futures)

    np_open=tckr_df['Open'].to_numpy()
    np_close=tckr_df['Close'].to_numpy()
    weekday_np=pd.to_datetime(tckr_df['Opened']).dt.dayofweek.to_numpy()
    CO_chng=(((np_close/np_open)-1)*100)
    h_dict={}
    for i in range(0,7):
      h_dict[i]=np.mean([chng for chng, day in zip(CO_chng, weekday_np) if day==i])
    return h_dict

def get_hour_changes(ticker='BTCUSDT', interval='1m',  futures=True):
  ### get data ###
  if futures:
    directory='/data/binance_data_futures/'
  else:
    directory='/data/binance_data_spot/'
  tckr_df = check_df(86400, directory, interval, ticker, futures)

  np_open=tckr_df['Open'].to_numpy()
  np_close=tckr_df['Close'].to_numpy()
  hour_np=pd.to_datetime(tckr_df['Opened']).dt.hour.to_numpy()
  CO_chng=(np_close/np_open-1)*100
  h_dict={}
  for i in range(0,24):
    h_dict[i]=np.mean([chng for chng, hour in zip(CO_chng, hour_np) if hour==i])
  return h_dict

############################# SIGNAL GENERATORS ######################################
######################################################################################
def Hour_strat(hour_col, interval='1m',  futures=True):
  h_dict=get_hour_changes(ticker='BTCUSDT', interval='1m',  futures=True)
  Hour_sig=[]
  for i in range(len(hour_col)):
    Hour_sig.append(h_dict[hour_col.iloc[i]])
  return Hour_sig

def Weekday_strat(column, interval='1m', futures=True):
  wd_dict=get_weekday_changes(ticker='BTCUSDT', interval=interval, futures=futures)
  Weekday_sig=[]
  for i in range(len(column)):
    Weekday_sig.append(wd_dict[column.iloc[i]])
  return Weekday_sig

def ULT_RSI_strat(column):
  signals=[0]
  for i in range(1,len(column)):
    # Overbought
    if column.iloc[i-1]>65.0:
      signals.append(-1)
    # Oversold
    elif column.iloc[i-1]<35.0:
      signals.append(1)
    # Sell singal
    elif column.iloc[i-1]>65.0 and column.iloc[i]<65.0:
      signals.append(-2)
    # Buy singal
    elif column.iloc[i-1]<35.0 and column.iloc[i]>35.0:
      signals.append(2)
    else:
      signals.append(0)
  return signals

def ADX_strat(adx_col,minus_DI,plus_DI):
  signals=[0]
  for i in range(1,len(adx_col)):
    # Strong BUY signal
    if plus_DI.iloc[i]>minus_DI.iloc[i] and plus_DI.iloc[i-1]<minus_DI.iloc[i-1] and adx_col.iloc[i]>25.0:
      signals.append(4)
    # BUY signal
    elif plus_DI.iloc[i]>minus_DI.iloc[i] and plus_DI.iloc[i-1]<minus_DI.iloc[i-1] and adx_col.iloc[i]>20.0:
      signals.append(3)

    # Strong SELL signal
    elif plus_DI.iloc[i]<minus_DI.iloc[i] and plus_DI.iloc[i-1]>minus_DI.iloc[i-1] and adx_col.iloc[i]>25.0:
      signals.append(-4)
    # SELL signal
    elif plus_DI.iloc[i]<minus_DI.iloc[i] and plus_DI.iloc[i-1]>minus_DI.iloc[i-1] and adx_col.iloc[i]>20.0:
      signals.append(-3)
    
    # strong heading toward BUY
    elif plus_DI.iloc[i]>plus_DI.iloc[i-1] and minus_DI.iloc[i]<minus_DI.iloc[i-1] and plus_DI.iloc[i]<minus_DI.iloc[i] and adx_col.iloc[i]>25.0:
      signals.append(2)
    # strong heading toward SELL
    elif plus_DI.iloc[i]<plus_DI.iloc[i-1] and minus_DI.iloc[i]>minus_DI.iloc[i-1] and plus_DI.iloc[i]>minus_DI.iloc[i] and adx_col.iloc[i]>25.0:
      signals.append(-2)

    # heading toward BUY
    elif plus_DI.iloc[i]>plus_DI.iloc[i-1] and minus_DI.iloc[i]<minus_DI.iloc[i-1] and plus_DI.iloc[i]<minus_DI.iloc[i] and adx_col.iloc[i]>20.0:
      signals.append(1)
    # heading toward SELL
    elif plus_DI.iloc[i]<plus_DI.iloc[i-1] and minus_DI.iloc[i]>minus_DI.iloc[i-1] and plus_DI.iloc[i]<minus_DI.iloc[i] and adx_col.iloc[i]>20.0:
      signals.append(-1)
    else:
      signals.append(0)
  return signals

def ADX_trend(column):
  signals=[0]
  for i in range(1,len(column)):
    # Strong trend
    if column.iloc[i]>25.0:
      signals.append(1)
    # Weak trend
    elif column.iloc[i]<20.0:
      signals.append(-1)
    else:
      signals.append(0)
  return signals

def MFI_strat(mfi_col):
  signals=[0]
  for i in range(1,len(mfi_col)):
    # Strong BUY signal
    if mfi_col.iloc[i]>90:
      signals.append(2)
    # Strong SELL signal
    elif mfi_col.iloc[i]<10:
      signals.append(-2)
    # BUY signal
    elif mfi_col.iloc[i]>80:
      signals.append(1)
    # SELL signal
    elif mfi_col.iloc[i]<20:
      signals.append(-1)
    else:
      signals.append(0)
  return signals

def MFI_divergence(mfi_col,close_col):
  signals=[0]
  for i in range(1,len(mfi_col)):
    # BUY signal
    if (mfi_col.iloc[i-1]<20 and mfi_col.iloc[i]>20) and (close_col.iloc[i]<close_col.iloc[i-1]):
      signals.append(1)
    # SELL signal
    elif (mfi_col.iloc[i-1]>80 and mfi_col.iloc[i]<80) and (close_col.iloc[i]>close_col.iloc[i-1]):
      signals.append(-1)
    else:
      signals.append(0)
  return signals

def MACD_cross(macd_col,signal_col):
  signals=[0]
  for i in range(1,len(macd_col)):
    # Buy signal
    if macd_col.iloc[i]>signal_col.iloc[i] and macd_col.iloc[i-1]<signal_col.iloc[i-1]:
      signals.append(1)
    # Sell signal
    elif macd_col.iloc[i]<signal_col.iloc[i] and macd_col.iloc[i-1]>signal_col.iloc[i-1]:
      signals.append(-1)
    else:
      signals.append(0)
  return signals

def MACDhist_reversal(macdhist_col):
  signals=[0,0]
  for i in range(2,len(macdhist_col)):
    # Buy signal
    if macdhist_col.iloc[i]>macdhist_col.iloc[i-1] and (macdhist_col.iloc[i-1]<macdhist_col.iloc[i-2]<macdhist_col.iloc[i-3]):
      signals.append(1)
    # Sell signal
    elif macdhist_col.iloc[i]<macdhist_col.iloc[i-1] and (macdhist_col.iloc[i-1]>macdhist_col.iloc[i-2]>macdhist_col.iloc[i-3]):
      signals.append(-1)
    else:
      signals.append(0)
  return signals

def MACD_zerocross(macd_col,signal_col):
  signals=[0]
  for i in range(1,len(macd_col)):
    # Buy signal
    if macd_col.iloc[i]>0 and macd_col.iloc[i-1]<0:
      signals.append(1)
    # Buy signal
    elif signal_col.iloc[i]>0 and signal_col.iloc[i-1]<0:
      signals.append(1)
    # Sell signal
    elif macd_col.iloc[i]<0 and macd_col.iloc[i-1]>0:
      signals.append(-1)
    # Sell signal
    elif signal_col.iloc[i]<0 and signal_col.iloc[i-1]>0:
      signals.append(-1)
    else:
      signals.append(0)
  return signals

def TP_sig(mid, close):
  return mid-close

def BB_sig(mid, up_x1, low_x1, up_x2, low_x2, up_x3, low_x3, close):
  signals=[]
  for i in range(len(close)):
    if close[i]>up_x3[i]:
      signals.append(-4)
    elif close[i]<low_x3[i]:
      signals.append(4)
    elif up_x2[i]<close[i]<up_x3[i]:
      signals.append(-3)
    elif low_x3[i]<close[i]<low_x2[i]:
      signals.append(3)
    elif up_x1[i]<close[i]<up_x2[i]:
      signals.append(-2)
    elif low_x2[i]<close[i]<low_x1[i]:
      signals.append(2)
    elif mid[i]<close[i]<up_x1[i]:
      signals.append(-1)
    elif low_x1[i]<close[i]<mid[i]:
      signals.append(1)
    else:
      signals.append(0)
  return signals

def TMA_sig(np_close, np_TriMA, np_ATR_75):
  return (np_TriMA-np_close)/np_ATR_75

def price_levels(close_col, interval='1m'):
  signals=[]
  lvls=get_levels(futures=True, interval=interval, decimals=0)
  for i in range(0,len(close_col)):
    try:
      signals.append( lvls[round(close_col.iloc[i], 0)]  )
    except:
      signals.append(0)
  return signals

def move_prob(open_col_np, close_col_np, futures=True, interval='1m'):
  avg_gain,gain_stdev,avg_loss,loss_stdev,*_=get_avg_changes(ticker='BTCUSDT', interval=interval, futures=futures)
  signals=[ ((((close/open)-1)*100)-avg_gain)+((((close/open)-1)*100)/gain_stdev) if ((close/open)-1)>0 else ((((close/open)-1)*100)-avg_loss)+((((close/open)-1)*100)/loss_stdev) for open, close in zip(open_col_np,close_col_np) ]
  return signals

def vol_prob(vol_col_np):
  avg_vol=np.mean(vol_col_np)
  stdev_vol=np.std(vol_col_np)
  signals=(vol_col_np-avg_vol)/stdev_vol
  return signals

def scaleColumns(df, scaler):
    for col in df.columns:
      if col not in ['Open time', 'Close time', 'Open', 'High', 'Low', 'Close']:
        #print(col)
        #caler.fit(df[[col]])
        df[col] = scaler.fit_transform(df[[col]])
    return df
######################################################################################
######################################################################################

def add_AT_features(df, suffix='_'):
  print(f'     adding features to {suffix}...')
  np_vol=df['Volume'].to_numpy()
  np_open=df['Open'].to_numpy()
  np_close=df['Close'].to_numpy()
  np_high=df['High'].to_numpy()
  np_low=df['Low'].to_numpy()
  if suffix=='1m': 
    df['weekday']=pd.to_datetime(df['Open time'], unit='ms').dt.dayofweek
    df['hour']=pd.to_datetime(df['Open time'], unit='ms').dt.hour
    df['hour_sig']=Hour_strat(df['hour'], suffix)
    df['weekday_sig']=Weekday_strat(df['weekday'], suffix)
    df.drop(columns=['weekday',	'hour'], inplace=True)
    df['lvls_count']=price_levels(df['Close'], suffix)
    df['move_prob']=move_prob(np_open, np_close, suffix)
    df['vol_prob']=vol_prob(np_vol)
    suffix=''
  else:
    df['lvls_count'+suffix]=price_levels(df['Close'], interval=suffix)
    df['move_prob'+suffix]=move_prob(np_open,np_close,interval=suffix)
    df['vol_prob'+suffix]=vol_prob(np_vol)
  open=df['Open'] 
  high=df['High']
  low=df['Low']
  close=df['Close']
  volume=df['Volume']
  #print(suffix)
  #df=ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
  df['RSI'+suffix] =                                                   RSI(close, timeperiod=10)
  df['ULT'+suffix] =                                                   ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
  df['ADX'+suffix] =                                                   ADX(high, low, close, timeperiod=14)
  df['-DI'+suffix] =                                                   MINUS_DI(high, low, close, timeperiod=14)
  df['+DI'+suffix] =                                                   PLUS_DI(high, low, close, timeperiod=14)
  df['MFI'+suffix] =                                                   MFI(high, low, close, volume, timeperiod=14)
  #df['Hilbert'+suffix] =                                               HT_TRENDLINE(close)
  #df['MIDPOINT'+suffix] =                                              MIDPOINT(close, timeperiod=14)
  df['macd'+suffix],df['macdsignal'+suffix],df['macdhist'+suffix] =    MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
  df['ATR'+suffix] =                                                   ATR(high, low, close, timeperiod=14)
  df['ADOSC'+suffix] =                                                 ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
  df['APO'+suffix] =                                                   APO(close, fastperiod=12, slowperiod=26, matype=0)
  df['AROONOSC'+suffix] =                                              AROONOSC(high, low, timeperiod=14)
  df['STOCHRSIfastk'+suffix], df['STOCHRSIfastd'+suffix] =             STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
  df['CMO'+suffix] =                                                   CMO(close, timeperiod=14)
  df['BOP'+suffix] =                                                   BOP(open, high, low, close)
  df['TRANGE'+suffix] =                                                TRANGE(high, low, close)
  df['PPO'+suffix] =                                                   PPO(close, fastperiod=12, slowperiod=26, matype=0)
  df['WILLR'+suffix] =                                                 WILLR(high, low, close, timeperiod=14)
  df['KST'+suffix] =                                                  ta.trend.kst_sig(close)
  df['Vortex'+suffix] =                                               ta.trend.VortexIndicator(high, low, close).vortex_indicator_diff()
  df['STC'+suffix] =                                                  ta.trend.STCIndicator(close).stc()
  df['PVO'+suffix] =                                                  ta.momentum.PercentageVolumeOscillator(volume).pvo()
  df['AO'+suffix] =                                                   ta.momentum.AwesomeOscillatorIndicator(high, low).awesome_oscillator()
  df['up_x1'], df['mid'+suffix], df['low_x1'] =                       BBANDS(close, timeperiod=5, nbdevup=1, nbdevdn=1, matype=0)
  df['up_x2'], _, df['low_x2'] =                                      BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
  df['up_x3'], _, df['low_x3'] =                                      BBANDS(close, timeperiod=5, nbdevup=3, nbdevdn=3, matype=0)
  df['TriMA'+suffix] =                                                TRIMA(close, timeperiod=35)
  ### Signals ###
  df['RSI_sig'+suffix]=ULT_RSI_strat(df['RSI'+suffix])
  df['ULT_sig'+suffix]=ULT_RSI_strat(df['ULT'+suffix])
  df['ADX_sig'+suffix]=ADX_strat(df['ADX'+suffix], df['-DI'+suffix], df['+DI'+suffix])
  df['ADX_trend'+suffix]=ADX_trend(df['ADX'+suffix])
  df['MFI_strat'+suffix]=MFI_strat(df['MFI'+suffix])
  df['MFI_divergence'+suffix]=MFI_divergence(df['MFI'+suffix], df['Close'])
  df['MACD_cross'+suffix] = MACD_cross(df['macd'+suffix],df['macdsignal'+suffix])
  df['MACDhist_reversal'+suffix] = MACDhist_reversal(df['macdhist'+suffix])
  df['MACD_zerocross'+suffix] = MACD_zerocross(df['macd'+suffix],df['macdsignal'+suffix])
  df['TP_sig'+suffix] = TP_sig(df['mid'+suffix].to_numpy(), np_close)
  df['BB_sig'+suffix] = BB_sig(df['mid'+suffix].to_numpy(), df['up_x1'].to_numpy(), df['low_x1'].to_numpy(), df['up_x2'].to_numpy(), df['low_x2'].to_numpy(), df['up_x3'].to_numpy(), df['low_x3'].to_numpy(), np_close)
  df['TMA_sig'+suffix] = TMA_sig(np_close, TRIMA(close, timeperiod=35).to_numpy(), ATR(high, low, close, timeperiod=75).to_numpy())

  # OHLC simple features
  df['C-O'+suffix]=np_close-np_open
  df['H-L'+suffix]=np_high-np_low
  # down candle wicks
  df['H-O'+suffix]=np_high-np_open
  df['C-L'+suffix]=np_open-np_low
  # up candle wicks
  df['O-L'+suffix]=np_open-np_low
  df['H-C'+suffix]=np_high-np_close

  df = df.drop(columns=['MFI'+suffix, 'up_x1', 'low_x1', 'up_x2', 'low_x2', 'up_x3', 'low_x3'])
  #df = df.drop(columns=['RSI'+suffix, 'macd'+suffix, 'macdsignal'+suffix, 'macdhist'+suffix])
  #df = df.drop(columns=['RSI'+suffix, 'ULT'+suffix, 'ADX'+suffix, '-DI'+suffix, '+DI'+suffix, 'MFI'+suffix, 'Hilbert'+suffix, 'MIDPOINT'+suffix, 'macd'+suffix, 'macdsignal'+suffix, 'macdhist'+suffix])
  return scaleColumns(df, preprocessing.MinMaxScaler())

def get_df(interval_list, month_list):
  print(f' generating df with {interval_list} intervals, from {month_list} months...')
  dfs=[]
  for mth in month_list:
    for itv in interval_list:
      # check if data for given interval and month exists inside data folder
      path=os.getcwd()+'\data\get_df'
      file_name='BTCUSDT-'+itv+'-2022-'+mth+'.zip'
      full_path=path+'\\'+file_name
      if os.path.exists(full_path):
        pass
      else:
        os.makedirs(path)
        print(f' donwloading {file_name}')
        link ='https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/'+itv+'/'+file_name
        wget.download(link, full_path)
        with zipfile.ZipFile(full_path, 'r') as zip_ref:
            zip_ref.extractall(path)

      print(f' month {mth}', end=' ')
      if itv=='1m':
        df=pd.read_csv(full_path)
        df.columns = ['Open time',	'Open',	'High',	'Low',	'Close',	'Volume',	'Close time',	'Quote asset volume',	'Number of trades',	'Taker buy base asset volume',	'Taker buy quote asset volume',	'Ignore']
        df.drop(columns=['Quote asset volume',	'Number of trades',	'Taker buy base asset volume',	'Taker buy quote asset volume',	'Ignore'],inplace=True)
        df=add_AT_features(df, itv)
      else:
        _df=pd.read_csv(full_path)
        _df.columns = ['Open time',	'Open',	'High',	'Low',	'Close',	'Volume',	'Close time',	'Quote asset volume',	'Number of trades',	'Taker buy base asset volume',	'Taker buy quote asset volume',	'Ignore']
        _df.drop(columns=['Quote asset volume',	'Number of trades',	'Taker buy base asset volume',	'Taker buy quote asset volume',	'Ignore'],inplace=True)
        _df=add_AT_features(_df, itv)
        suff='_'+itv
        df=pd.merge_asof(df,_df, on='Close time',suffixes=('', suff))
        df.drop(columns=['Open time_'+itv,'Open_'+itv, 'High_'+itv, 'Low_'+itv, 'Close_'+itv, 'Volume_'+itv,],inplace=True)
      #print(df[73:93])
    dfs.append(df)
  df=pd.concat(dfs, ignore_index=True)
  df.fillna(method='ffill',inplace=True)
  df.dropna(inplace=True)
  df.drop(columns=['Close time'], inplace=True)
  return df
