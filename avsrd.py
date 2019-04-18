import librosa
import numpy as np
import os


for i in [6]:
  index_str=str(i)
  y,_=librosa.core.load('/home/tiancheng/Desktop/DATA/P'+index_str+'/LENA/CHA/p'+index_str+'.wav.wav')
  y_average=sum(y)/len(y)
  y_std=np.var(y)
  print((y_average))
  print((y_std))
  print('/n')