import pandas as pd 
import numpy as np
from collections import deque
import random
from statistics import mean, stdev
import time

from visualize import TradingGraph

class CustomEnv:
    def __init__(self, df, initial_balance=100, init_postition_size=10, leverage=125, max_steps=0, lookback_window_size=240, Render_range=120, visualize=False):
        self.df = df.dropna().reset_index()
        self.df.drop(columns=['index'], inplace=True)
        print('Feature list in df: ', end=' ')
        for col in self.df.columns:
          print(col, end=", ")
        print()
        self.df_total_steps = len(self.df)-1
        self.max_steps = max_steps
        self.initial_balance = initial_balance
        self.balance = initial_balance-init_postition_size
        self.position_size = init_postition_size
        self.init_postition_size = init_postition_size
        self.start_postition_size = init_postition_size
        self.leverage = leverage
        self.lookback_window_size = lookback_window_size
        self.Render_range = Render_range
        self.visualize = visualize

        # features(col names) to exclude from env state
        self.exclude_list=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']

        # Action space from 0 to 3, 0 is hold, 1 is buy, 2 is sell
        self.action_space = np.array([0, 1, 2])
        self.action_space_n = len(self.action_space)

        self.orders_history = deque(maxlen=self.lookback_window_size)
        self.market_history = deque(maxlen=self.lookback_window_size)

        # State size contains Market+Orders history for the last lookback_window_size steps
        self.state_size = (self.lookback_window_size, len(self.df.columns)+7)

    # Reset the state of the environment to an initial state
    def reset(self, env_steps_size = 0):
        if self.max_steps>0: env_steps_size=self.max_steps
        if self.visualize: 
          self.trades = []
          self.visualization = TradingGraph(Render_range=self.Render_range) # init visualization
        self.position_size=0
        self.init_postition_size=self.start_postition_size
        self.balance=self.initial_balance
        self.qty=0
        self.enter_price=None
        self.in_position=0
        self.in_position_log=[self.in_position, self.in_position]
        self.in_position_counter=0
        self.pnl=0
        self.episode_orders = 0
        self.good_trades_count=1
        self.good_trades=[]
        self.bad_trades_count=1
        self.bad_trades=[]
        self.pnl_list=[self.pnl,self.pnl]
        self.cumulative_pnl=0
        self.balance_history=[self.balance,self.balance]
        self.reward=0

        if env_steps_size > 0:
            self.start_step = random.randint(self.lookback_window_size, self.df_total_steps - env_steps_size)
            self.end_step = self.start_step + env_steps_size
        else:
            self.start_step = random.randint(self.lookback_window_size, self.df_total_steps)
            self.end_step = self.df_total_steps
        self.current_step = self.start_step

        for i in reversed(range(self.lookback_window_size)):
            current_step = self.current_step - i
            self.orders_history.append([self.position_size, self.balance, self.in_position, self.qty, self.cumulative_pnl, self.pnl, (self.df.loc[self.current_step, 'Close']/self.enter_price)-1 if self.enter_price!=None else None])
            self.market_history.append([self.df.loc[self.current_step, col_name] for col_name in self.df.columns if col_name not in self.exclude_list])
            #print(i, end=': ')
            #print()
            #print(self.orders_history[-1])
            #print(self.market_history[-1])

        state = np.concatenate((self.market_history, self.orders_history), axis=1)
        #print(state)
        return state

    # Get the data points for the given current_step
    def _next_observation(self):
      self.market_history.append([self.df.loc[self.current_step, col_name] for col_name in self.df.columns if col_name not in self.exclude_list])
      obs = np.concatenate((self.market_history, self.orders_history), axis=1)
      #print(obs)
      #time.sleep(120)
      return obs
    
    def _calculate_reward(self):
      ###### Profit closing ######
      if (not self.in_position) and self.in_position_log[-2]==True and (self.position_size>round(self.init_postition_size*0.999, 2)):
        if self.cumulative_pnl>0:
          self.reward=self.cumulative_pnl
        elif self.cumulative_pnl<0:
          self.cumulative_pnl=(-1)*self.cumulative_pnl
        #self.reward=(self.balance/self.balance_history[-2]-1)+abs(self.cumulative_pnl)*1
        #self.reward=2+(self.bad_trades_count/self.good_trades_count)+self.orders_history[-2][-2]
        #self.reward=3+(self.bad_trades_count/self.good_trades_count)+self.cumulative_pnl
        self.reward=0.5
      ###### Loss closing ######
      elif (not self.in_position) and self.in_position_log[-2]==True and (self.position_size<round(self.init_postition_size*0.999, 2)):
        if self.cumulative_pnl<0:
          self.reward=self.cumulative_pnl
        elif self.cumulative_pnl>0:
          self.cumulative_pnl=(-1)*self.cumulative_pnl
        #self.reward=(self.balance_history[-2]/self.balance-1)*(-1)-abs(self.cumulative_pnl)*1
        #self.reward=1/abs(self.cumulative_pnl)
        #self.reward=-3-(self.bad_trades_count/self.good_trades_count)+self.cumulative_pnl
        self.reward=-0.5
      ###### Holding position ######
      #elif self.pnl<0 and self.pnl_list[-2]>0:
        #if self.cumulative_pnl<0:
          #self.cumulative_pnl=self.cumulative_pnl
          #self.reward=self.cumulative_pnl
        #elif self.cumulative_pnl>0:
          #self.cumulative_pnl=(-1)*self.cumulative_pnl
          #self.cumulative_pnl=self.cumulative_pnl
          #self.reward=self.cumulative_pnl
        #self.reward=self.pnl
        #self.reward=self.pnl-self.pnl_list[-2]
      #elif self.pnl>0 and self.pnl_list[-2]<0:
        #self.reward=self.pnl*10
        #self.cumulative_pnl=0
        #self.reward=self.pnl
        #self.reward=self.pnl-self.pnl_list[-2]
      elif self.in_position:
        '''if self.pnl>0:
          if self.cumulative_pnl>0:
            self.reward=self.cumulative_pnl
          elif self.cumulative_pnl<0:
            self.cumulative_pnl=(-1)*self.cumulative_pnl
            self.reward=self.cumulative_pnl
        elif self.pnl<0:
          if self.cumulative_pnl<0:
            self.reward=self.cumulative_pnl
          elif self.cumulative_pnl>0:
            self.cumulative_pnl=(-1)*self.cumulative_pnl
            self.reward=self.cumulative_pnl
        else:
          self.reward=self.pnl'''
        #self.reward=0
        self.reward=self.cumulative_pnl/self.in_position_counter
        '''if self.pnl>0:
          if self.cumulative_pnl>0:
            self.reward=self.cumulative_pnl/self.in_position_counter
          elif self.cumulative_pnl<0:
            self.cumulative_pnl=(-1)*self.cumulative_pnl
            self.cumulative_pnl=self.cumulative_pnl
            self.reward=self.cumulative_pnl/self.in_position_counter
        elif self.pnl<0:
          if self.cumulative_pnl<0:
            self.reward=self.cumulative_pnl
          elif self.cumulative_pnl>0:
            self.cumulative_pnl=(-1)*self.cumulative_pnl
            self.cumulative_pnl=self.cumulative_pnl
            self.reward=self.cumulative_pnl
        else:
          self.reward=self.pnl'''
      else:
        #self.reward=self.cumulative_pnl
        #self.reward=self.pnl
        self.reward=0
      return self.reward

    def _get_pnl(self, current_price):
      self.pnl=(((current_price/self.enter_price)-1)*self.leverage)*self.qty
      self.pnl_list.append(self.pnl)
      if self.cumulative_pnl!=0 and len(self.pnl_list)>1:
        self.cumulative_pnl+=(self.pnl-self.pnl_list[-2])
      else:
        self.cumulative_pnl+=self.pnl
      #self.cumulative_pnl+=self.pnl
      return self.pnl

    def _finish_episode(self):
      self.orders_history.append([self.position_size, self.balance, self.in_position, self.qty, self.cumulative_pnl, self.pnl, (self.df.loc[self.current_step, 'Close']/self.enter_price)-1 if self.enter_price!=None else None])
      self._calculate_reward()
      done=True
      obs = self._next_observation()
      if len(self.balance_history)>0 and len(self.good_trades)>0 and len(self.bad_trades)>0:
        print('')
        print(f' Koncowy balans: ${self.balance:.2f} Sredni balans: ${mean(self.balance_history):.2f} Max balans: ${max(self.balance_history):.2f}', end='  ')
        print(f'zyskownych:{self.good_trades_count:} ({mean(self.good_trades)*100:.1f}%) stratnych:{self.bad_trades_count} ({mean(self.bad_trades)*100:.1f}%)')
      else:
        print('')
        print(f' Koncowy balans: ${self.balance:.2f} Max balans: ${max(self.balance_history):.2f} zyskownych:{self.good_trades_count} stratnych:{self.bad_trades_count}')
      info = {'action': 0,
              'reward': self.reward,
              'step': self.current_step}
      #return obs, (mean(self.balance_history)-self.initial_balance)*self.leverage, done, info
      return obs, 0, done, info

    def _open_position(self, side, price):
      self.in_position=1
      self.in_position_log.append(self.in_position)
      self.episode_orders += 1
      self.enter_price=price
      self.in_position_counter+=1
      #print(f' BEFORE self.position_size: {self.position_size}')
      #print(f' BEFORE self.balance: {self.balance}')
      self.position_size = round(self.init_postition_size*0.999, 2)
      self.balance-=self.init_postition_size
      #print(f' AFTER self.position_size: {self.position_size}')
      #print(f' AFTER self.balance: {self.balance}')
      self.balance_history.append(self.balance)
      if side=='long':
        self.qty = 1  #*(self.position_size/self.enter_price)
      elif side=='short':
        self.qty= -1  #(self.position_size/self.enter_price)

    def _close_position(self):
      self.qty = 0
      self.in_position=0
      self.in_position_log.append(self.in_position)
      self.in_position_counter=0
      self.position_size += round((self.position_size*self.pnl)*0.999, 2)
      if self.pnl>0:
        self.good_trades_count += 1
        self.good_trades.append(self.position_size/round(self.init_postition_size*0.999, 2)-1)
      elif self.pnl<0:
        self.bad_trades_count += 1
        self.bad_trades.append(self.position_size/round(self.init_postition_size*0.999, 2)-1)
      #print(f' self.pnl: {self.pnl}')
      #print(f' BEFORE self.position_size: {self.position_size}')
      #print(f' BEFORE self.balance: {self.balance}')
      self.balance += self.position_size
      if self.balance>self.initial_balance: self.init_postition_size=round(self.balance/10, 2)
      #print(f' AFTER self.position_size: {self.position_size}')
      #print(f' AFTER self.balance: {self.balance}')
      self.balance_history.append(self.balance)
      self.in_position_counter=0
      self.pnl = 0
      self.cumulative_pnl = 0
    
    # Execute one time step within the environment
    def step(self, action):
        self.current_step += 1
        if self.current_step==self.end_step:
          if self.in_position: self._close_position()
          return self._finish_episode()
        done=False
        self.in_position_log.append(self.in_position)
        close=self.df.loc[self.current_step, 'Close']
        current_price = random.uniform(round(close*1.0002, 2), round(close*0.9998, 2))
        #current_price=self.df.loc[self.current_step, 'Close']
        #current_price = random.uniform(self.df.loc[self.current_step, 'High'], self.df.loc[self.current_step, 'Low'])
        ########################## VISUALIZATION ###############################
        if self.visualize:
          Date = self.df.loc[self.current_step, 'Open time']
          High = self.df.loc[self.current_step, 'High']
          Low = self.df.loc[self.current_step, 'Low']
        ########################################################################
        if self.in_position:
          self._get_pnl(current_price)
          if self.pnl<=-0.9:
            self._close_position()
            #return self._finish_episode()
          else:
            ##CLOSING POSTIONS OR PASS##
            if action == 0:
              #print('IN POSITION PASS')
              pass
            # Close SHORT or LONG qty={-1,0,1}
            elif (action==1 and self.qty<0) or (action==2 and self.qty>0):
              #print('CLOSING POSITION')
              self._close_position()
              ########################## VISUALIZATION ###############################
              if self.visualize and action==1:  self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low, 'total': self.qty, 'type': "close_short"})
              elif self.visualize and action==2:  self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low, 'total': self.qty, 'type': "close_long"})
              ########################################################################
              if self.balance*0.999<=self.init_postition_size:
                return self._finish_episode()
        ##OPENING POSTIONS OR PASS##
        else:
          if action == 0:
            #print('NO POSITION PASS')
            pass
          # OPEN LONG
          elif self.balance*0.999<=self.init_postition_size:
            return self._finish_episode()
          elif action == 1:
            #print('OPENING LONG')
            self._open_position('long', current_price)
            if self.visualize: self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low, 'total': self.qty, 'type': "open_long"})
          # OPEN SHORT
          elif action == 2:
            #print('OPENING SHORT')
            self._open_position('short', current_price)
            if self.visualize: self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low, 'total': self.qty, 'type': "open_short"})
        
        self.orders_history.append([self.position_size, self.balance, self.in_position, self.qty, self.cumulative_pnl, self.pnl, (self.df.loc[self.current_step, 'Close']/self.enter_price)-1 if self.enter_price!=None else None])
        if self.in_position: self.in_position_counter+=1
        #Write_to_file(Date, self.orders_history[-1])
        self._calculate_reward()
        obs = self._next_observation()
        info = {'action': action,
                'reward': self.reward,
                'step': self.current_step}
        return obs, self.reward, done, info

    # render environment
    def render(self, visualize=False, *args, **kwargs):
      if visualize or self.visualize:
        Date = self.df.loc[self.current_step, 'Open time']
        Open = self.df.loc[self.current_step, 'Open']
        Close = self.df.loc[self.current_step, 'Close']
        High = self.df.loc[self.current_step, 'High']
        Low = self.df.loc[self.current_step, 'Low']
        Volume = self.reward
        # Render the environment to the screen
        self.visualization.render(Date, Open, High, Low, Close, Volume, self.balance, self.trades)