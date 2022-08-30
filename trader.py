from fileinput import filename
import os
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy
from rl.memory import SequentialMemory

from env import CustomEnv
import df_generator

if __name__=="__main__":
    try:
        df=pd.read_csv(os.getcwd()+'/data/df.csv', index_col=[0])
        df_test=pd.read_csv(os.getcwd()+'/data/df_test.csv', index_col=[0])
    except:
        intervals=['1m', '5m', '15m', '1h']

        months=['02','03','04','05']
        df=df_generator.get_df(intervals, months)
        path=os.getcwd()+'/data/df.csv'
        df.to_csv(path)
        print(f'df saved to {path}')
        test_months=['06', '07']
        df_test=df_generator.get_df(intervals, test_months)
        path=os.getcwd()+'/data/df_test.csv'
        df_test.to_csv(path)
        print(f'df_test saved to {path}')

    def build_model(states, actions, neurons, n_layers, window_length, lookback_window_size):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(window_length,lookback_window_size,states)))
        #model.add(Reshape((lookback_window_size, states), input_shape=(window_length, lookback_window_size, states)))
        #model.add(LSTM(neurons, return_sequences = True, input_shape=(lookback_window_size, states), activation = 'relu'))
        #model.add(Dense(states, activation='relu'))
        #model.add(Flatten(input_shape=(window_length,lookback_window_size,states)))
        for k in layers_set.values():
            model.add(tf.keras.layers.Dense(k, activation='relu'))
        model.add(tf.keras.layers.Dense(actions, activation='linear'))
        #model.summary()
        return model
    def build_agent(model, actions, window_length, lookback_window_size, policy=EpsGreedyQPolicy()):
        #policy = BoltzmannQPolicy()
        test_policy=policy
        memory = SequentialMemory(limit=100*window_length, window_length=window_length)
        dqn = DQNAgent(model=model, memory=memory, policy=policy, test_policy=test_policy,
                    nb_actions=actions, nb_steps_warmup=3600, target_model_update=1e-3, enable_double_dqn=True, enable_dueling_network=True, dueling_type='avg')
        return dqn



    policy=EpsGreedyQPolicy()
    scores_l=[]
    while True:
      lookback_size=random.choice(list(range(1,60,1)))
      n_count=int((lookback_size*(len(df.columns)+7)+3)//2)
      n_counts=list(range(n_count//32,n_count,n_count//32))
      n_layers=list(range(1,6))
      layers_set={}
      for i in range(random.choice(n_layers)):
        layers_set[i]=random.choice(n_counts)
      n_windows=random.choice(list(range(1,2)))
      print("####################################################")
      print(f'n_windows: {n_windows}', end='  ')
      #print(f'layers_count: {layers_set.keys()}', end='  ')
      print(f'layers_size: {layers_set.values()}', end='  ')
      print(f'lookback_size: {lookback_size}')

      train_env = CustomEnv(df, lookback_window_size=lookback_size, max_steps=4320, visualize=True, initial_balance=20, init_postition_size=2.0, leverage=125)
      test_env = CustomEnv(df_test, lookback_window_size=lookback_size, visualize=True, initial_balance=20, init_postition_size=2.0, leverage=125)

      model = build_model(len(train_env.reset()[-1]), train_env.action_space_n, n_count, layers_set, n_windows, lookback_size)
      #model.summary()
      dqn = build_agent(model, train_env.action_space_n, n_windows, lookback_size, policy=policy)
      dqn.compile(tf.keras.optimizers.Adam(learning_rate=1e-5), metrics=[tf.keras.metrics.RootMeanSquaredError()])
      ## Callbacks:
      #tfboard = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
      #dqn.fit(train_env, nb_steps=100000, visualize=False, callbacks=[tfboard], verbose=2, log_interval=250)
      dqn.fit(train_env, nb_steps=12_960, visualize=True, verbose=2, log_interval=250)
  
      scores = dqn.test(test_env, nb_episodes=1, visualize=True)
      scores_l.append([n_windows,n_layers,n_count,scores])
      cv = lambda x: abs(np.std(x, ddof=1) / np.mean(x)) * 100
      median=np.median(scores.history['episode_reward'])
      mean_r=np.mean(scores.history['episode_reward'])
      zm=cv(scores.history['episode_reward'])
      print(f'n_windows: {n_windows}', end='  ')
      #print(f'layers_count: {layers_set.keys()}', end='  ')
      print(f'layers_size: {layers_set.values()}', end='  ')
      print(f'lookback_size: {lookback_size}')
      print(f'mediana: {median}', end='  ')
      print(f'srednia: {mean_r}', end=' ')
      print(f'wsp. zmiennosci: {zm:.1f}%')
      if zm<=50 and median>0 and mean_r>0:
        model_dir=f'/content/drive/MyDrive/RLtrader/models/WspZm-{zm:.0f}/'
        os.makedirs(model_dir)
        filepath=f'Me-{median:.0f}window-{n_windows}look-{lookback_size}Layers-{layers_set.values()}.h5'
        dqn.save_weights(model_dir+filepath, overwrite=False)
        print(f"#### Zapisano model: {filepath} #####")
      print("####################################################")
    '''n_windows=1
    #n_count=8192
    n_layers=5
    policy=EpsGreedyQPolicy()
    window_sizes=list(range(5,240,5))
    scores_l=[]
    for window in window_sizes:
        train_env = CustomEnv(df, lookback_window_size=window, max_steps=1440, visualize=True, initial_balance=20, init_postition_size=2.0, leverage=125)
        test_env = CustomEnv(df_test, lookback_window_size=window, max_steps=20_000, visualize=True, initial_balance=20, init_postition_size=2.0, leverage=125)
        n_count=int((window*len(df.columns)+train_env.action_space_n)/2)
        #n_windows=window

        model = build_model(len(train_env.reset()[-1]), train_env.action_space_n, n_count, n_layers, n_windows, window)
        #model.summary()
        dqn = build_agent(model, train_env.action_space_n, n_windows, window, policy=policy)
        dqn.compile(tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=[tf.tf.keras.metrics.RootMeanSquaredError()])

        ## Callbacks:
        #tfboard = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
        #dqn.fit(train_env, nb_steps=100000, visualize=False, callbacks=[tfboard], verbose=2, log_interval=250)
        
        
        dqn.fit(train_env, nb_steps=15_000, visualize=True, verbose=2, log_interval=1000)
        print("####################################################")
        scores = dqn.test(test_env, nb_episodes=10, visualize=False)
        scores_l.append([n_windows,n_layers,n_count,scores])
        print(f'n_windows: {n_windows}', end='  ')
        print(f'n_layers: {n_layers}', end='  ')
        print(f'n_count: {n_count}', end='  ')
        print(f'window: {window}')
        cv = lambda x: abs(np.std(x, ddof=1) / np.mean(x)) * 100 
        print('mediana: ', end='  ')
        print(np.median(scores.history['episode_reward']), end='  ')
        print('srednia: ', end=' ')
        print(np.mean(scores.history['episode_reward']), end='  ')
        zm=cv(scores.history['episode_reward'])
        print(f'wsp. zmiennosci: {zm:.1f}%')
        print("####################################################")'''
