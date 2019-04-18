import librosa
import numpy as np
import os

filename = './p2.txt'
session = 'p2'


data = []
line_no = 1
with open(filename) as f:
    for line in f:
        if line.startswith('*FAN:'):
            temp = line.split('_')
            start = temp[3]
            finish = temp[4][:-4]
            data.append([float(start) / 1000, float(finish) / 1000])

for ind, item in enumerate(data):
    y, sr = librosa.core.load('P2.wav', offset = item[0], duration = item[1] - item[0])
    librosa.output.write_wav('./output/P2/' + session + '_' + str(ind) + '.wav', y, sr)
    


#data_noise = [[1 , 10], [2371, 2375], [841, 845], [955, 960], [24330, 24335]]
#data_mother  = [[66, 70], [35, 36], [828, 829], [11207, 11209], [11230, 11231.3], [11755, 11758]]




folder = './output/'
length = []

#y, sr = librosa.core.load('p2.wav')

for file in os.listdir(folder):
    if file.startswith('p2'):
        #print(file)
        #print(folder + file)
        y, sr = librosa.load(folder + file)
        length.append(librosa.get_duration(y=y, sr=sr))
print(np.mean(length), np.std(length))

