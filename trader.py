import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
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
        test_months=['06']
        df_test=df_generator.get_df(intervals, test_months)
        path=os.getcwd()+'/data/df_test.csv'
        df_test.to_csv(path)
        print(f'df_test saved to {path}')

    def build_model(states, actions, neurons, n_layers, window_length, lookback_window_size):
        model = tf.keras.models.Sequential()
        model.add(keras.layers.Flatten(input_shape=(window_length,lookback_window_size,states)))
        #model.add(Reshape((lookback_window_size, states), input_shape=(window_length, lookback_window_size, states)))
        #model.add(LSTM(neurons, return_sequences = True, input_shape=(lookback_window_size, states), activation = 'relu'))
        #model.add(Dense(states, activation='relu'))
        #model.add(Flatten(input_shape=(window_length,lookback_window_size,states)))
        for _ in range(n_layers):
            model.add(keras.layers.Dense(neurons, activation='relu'))
        model.add(keras.layers.Dense(actions, activation='linear'))
        #model.summary()
        return model
    def build_agent(model, actions, window_length, lookback_window_size, policy=EpsGreedyQPolicy()):
        #policy = BoltzmannQPolicy()
        test_policy=policy
        memory = SequentialMemory(limit=100*window_length, window_length=window_length)
        dqn = DQNAgent(model=model, memory=memory, policy=policy, test_policy=test_policy,
                    nb_actions=actions, nb_steps_warmup=3600, target_model_update=1e-3, enable_double_dqn=True, enable_dueling_network=True, dueling_type='avg')
        return dqn

    n_windows=1
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
        dqn = build_agent(model, train_env.action_space_n, n_windows, window, policy=BoltzmannQPolicy)
        dqn.compile(keras.optimizers.RMSprop(learning_rate=1e-5), metrics=[tf.keras.metrics.RootMeanSquaredError()])

        ## Callbacks:
        #tfboard = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
        #dqn.fit(train_env, nb_steps=100000, visualize=False, callbacks=[tfboard], verbose=2, log_interval=250)
        
        
        dqn.fit(train_env, nb_steps=15_000, visualize=False, verbose=2, log_interval=1000)
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
        print("####################################################")