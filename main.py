import librosa
import numpy as np
import sklearn
from sklearn.multiclass import  OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

y_baby,sr=librosa.load('baby.wav')
#y_baby=(np.matrix(y_baby))
len_baby=len(y_baby)
mfcc_baby=librosa.feature.mfcc(y=y_baby,sr=sr)
mfcc_baby=(np.matrix(mfcc_baby)).transpose()

y_mom,sr=librosa.load('mother.wav')
#y_mom=(np.matrix(y_mom))
len_mother=len(y_mom)
mfcc_mom=librosa.feature.mfcc(y=y_mom,sr=sr)
mfcc_mom=(np.matrix(mfcc_mom)).transpose()

y_noise,_=librosa.load('noise.wav')
y_1_noise,_=librosa.load('noise_1.wav')
y_2_noise,_=librosa.load('noise_2.wav')
#y_noise=(np.matrix(y_noise))
#y_noise=np.concatenate((np.concatenate((y_noise,y_1_noise)),y_2_noise))
y_noise=y_1_noise
len_noise=len(y_2_noise)
mfcc_noise=librosa.feature.mfcc(y=y_noise,sr=sr)
mfcc_noise=(np.matrix(mfcc_noise)).transpose()

data_train=np.concatenate((np.concatenate((mfcc_noise,mfcc_baby)),mfcc_mom))

bin_noise=(np.matrix(np.zeros(int(mfcc_noise.size / 20)))).transpose()
bin_baby=(np.matrix(np.ones(int(mfcc_baby.size / 20)))).transpose()
bin_mother=((np.matrix(np.ones(int(mfcc_mom.size / 20)))).transpose())*2

print(data_train.size)

bin_info=np.concatenate((np.concatenate((bin_noise,bin_baby)),bin_mother))

print(bin_info.size)


model_1=OneVsOneClassifier(RandomForestClassifier())
#model_1=OneVsOneClassifier(RandomForestClassifier())
#model_test=model_1.fit(data_train,bin_info)
print(data_train.size)

X_train, X_test, y_train, y_test = train_test_split(data_train, bin_info, test_size=0.1, random_state=0)
model_test=model_1.fit(X_train,y_train)
result=model_test.score(X_test,y_test)
print(result)

y_exam,sr=librosa.load('exam_2.wav')
len_exam=len(y_exam)
mfcc_exam=librosa.feature.mfcc(y=y_exam,sr=sr,n_mfcc=20)
mfcc_exam=(np.matrix(mfcc_exam)).transpose()

prediction=model_test.predict(mfcc_exam)
prediction=np.matrix(prediction)
prediction=np.array(prediction)
prediction=prediction.tofile("result.txt",sep=" ",format="%d")



