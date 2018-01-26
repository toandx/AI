from env4 import TicTacToeEnv as env
from AI import AI
import gym
import numpy as np
import pandas as pd
def arr_to_csv(x):
 np.savetxt('data.csv',x,delimiter=',')
def csv_to_arr():
 df=pd.read_csv('data.csv',sep=',',header=None)
 return(df.values)
env=env()
player1=AI(env,2)
player2=AI(env,1)
for i in range(1000000):
 env.reset()
 player1.reset()
 player2.reset()
 while (env.done==False):
  player1.nextState()
  if (env.done):
   break
  player2.nextState()
arr_to_csv(player1.q)




