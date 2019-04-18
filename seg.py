import librosa
import numpy as np
import os

no_cha_flag=0
no_wav_flag=0
index_of_no_cha=[]
index_of_no_wav=[]
buffer_mean_var_info= [[0 for i in range(3)] for j in range(28)]
counter_valid=0
for i in [4]:

    index_str = str(i)
    # filename = './p2.txt'
    filename = '/home/tiancheng/Desktop/DATA/P'+index_str+'/LENA/CHA/p'+index_str+'.cha'
    session = 'p' + index_str

    data = []
    line_no = 1

    try:
        y_try_1=open(filename)
    except FileNotFoundError:
        index_of_no_cha=np.append(index_of_no_cha,i)
        no_cha_flag=1

    if no_cha_flag==1:
        no_cha_flag=0
        continue

    with open(filename) as f:
        for line in f:
            print('reach2')
            if line.startswith('*FAN:'):
                temp = line.split('_')
                lengg=len(temp)
                start = temp[lengg-2]
                print(start)
                finish = temp[lengg-1][:-1]
                finish=finish[:(len(finish)-1)]
                print('/n')
                print(finish)
                data.append([float(start) / 1000, float(finish) / 1000])
    print('reachdown')
    try:
       y_try,_=librosa.core.load('/home/tiancheng/Desktop/DATA/P'+index_str+'/LENA/CHA/p'+index_str+'.wav.wav',offset=0,duration=1)
    except FileNotFoundError:
                       print('reach3')
                       index_of_no_wav=np.append(index_of_no_wav,i)
                       no_wav_flag=1

    if no_wav_flag==1:
        no_wav_flag=0
        continue

    indi_dir='/home/tiancheng/Desktop/DATA/output/P'+index_str
    os.mkdir(indi_dir)

    for ind, item in enumerate(data):
        print('reach1')
        y, sr = librosa.core.load('/home/tiancheng/Desktop/DATA/P'+index_str+'/LENA/CHA/p'+index_str+'.wav.wav',offset=item[0], duration=item[1] - item[0])
        librosa.output.write_wav(indi_dir+'/'+ session + '_' + str(ind) + '.wav', y, sr)

    # data_noise = [[1 , 10], [2371, 2375], [841, 845], [955, 960], [24330, 24335]]
    # data_mother  = [[66, 70], [35, 36], [828, 829], [11207, 11209], [11230, 11231.3], [11755, 11758]]

    folder = './output/' + 'P' + index_str + '/'
    length = []

    # y, sr = librosa.core.load('p2.wav')
    indi_dir=indi_dir+'/'

    for file in os.listdir(indi_dir):
        if file.startswith('p' + index_str):
            # print(file)
            # print(folder + file)
            print('reach')
            y, sr = librosa.load(indi_dir+file)
            length.append(librosa.get_duration(y=y, sr=sr))
    print(np.mean(length), np.std(length))
    buffer_mean_var_info[i][:]=[i,np.mean(length),np.str(length)]
    counter_valid=counter_valid+1

buffer_mean_var_info=np.matrix(buffer_mean_var_info)
buffer_mean_var_info.tofile("result.txt",sep=" ",format="%d")
counter_valid=np.array([counter_valid])
counter_valid.tofile("num_valid.txt",sep=" ",format="%d")

