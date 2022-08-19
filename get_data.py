import os
import sys
import shutil
import pandas as pd
from binance_data import DataClient


def get_data(ticker='BTCUSDT', interval='1h', futures=True, decimals=0, progress_statements=True,):
  if futures:
    directory='/data/binance_data_futures/'
  else:
    directory='/data/binance_data_spot/'

  ### If inside colab notebook ###
  if 'google.colab' in sys.modules:
    drive_path='/content/drive/MyDrive/RLtrader'+directory+interval+'_data/'+ticker
    if not os.path.exists(drive_path):
        os.makedirs(drive_path)
        print(f'created dir {drive_path}')

    # move already downloaded data from google drive to working dir
    local_dir = '/content'+directory+interval+'_data/'
    shutil.move(drive_path, local_dir)

    # refill with newer data 
    print('\n updating data...')
    store_data = DataClient(futures=futures).kline_data([ticker.upper()], interval, storage=['csv', os.getcwd()+directory], progress_statements=progress_statements)
    local_dir_updated=os.getcwd()+directory+interval+'_data/'+ticker
    shutil.move(local_dir_updated, drive_path)

    tckr_df = pd.read_csv(drive_path+'/'+ticker+'.csv', header=0)
    return tckr_df
  else:
    local_dir=os.getcwd()+directory+interval+'_data/'+ticker
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
        print(f'created dir {local_dir}')
    print('\n updating data...')
    store_data = DataClient(futures=futures).kline_data([ticker.upper()], interval, storage=['csv', os.getcwd()+directory], progress_statements=progress_statements)
    local_dir_updated=os.getcwd()+directory+interval+'_data/'+ticker

    tckr_df = pd.read_csv(local_dir_updated+'/'+ticker+'.csv', header=0)
    return tckr_df

if __name__=='__main__':
    get_data(str(sys.argv[1]), str(sys.argv[2]), sys.argv[3])