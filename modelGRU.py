from cProfile import label
from msilib import sequence
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
import os
from keras.models import Sequential
from keras.layers import LSTM,GRU, Dense,Dropout,TimeDistributed
from keras.callbacks import TensorBoard
from tensorflow.python.util.tf_export import keras_export


#print(label_map)

DATA_PATH = os.path.join('MP_DATA')

# hanh dong
actions = np.array(['A','B','BAN','BIET','BUON','C','CAM ON','CHAP NHAN','D','DEL','E','G','GAP LAI','H','HEN','HIEU',
                        'I','K','KHOE','KHONG BIET','KHONG HIEU','L','M','N','NGAC NHIEN','NONG TINH','O','P','Q','R','RAT VUI DUOC GAP BAN'
                        ,'S','SO','T','TEN LA','THAT TUYET VOI','THUONG','TINH CAM','TOI','U','V','VO TAY','X','XAU HO','XIN LOI','Y'])

# thu thap 30 chuoi video
no_sequences = 30

label_map = {label:num for num , label in enumerate(actions)}
# co the la chuyen video thanh 30 frame
sequences_leght = 30

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequences_leght):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# np.array(sequences).shape (90,30,1662)

# np.array(labels).shape  (90)

X = np.array(sequences)
X.shape

#print(X.shape)

# Chuan bi du lieu de train 
y = to_categorical(labels).astype(int)
#print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

"""
from keras.preprocessing.image import ImageDataGenerator
# generater
datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    validation_split=0.2)

datagen.fit(X_train)
"""



log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Layer
"""
model = Sequential()
model.add(GRU(64, return_sequences=True, activation='relu'), input_shape=(30,126)))
model.add(GRU(128, return_sequences=True,activation='relu')))
model.add(GRU(64, return_sequences=False,activation='relu')))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu')) 
model.add(Dense(actions.shape[0], activation='softmax'))
"""
model = Sequential()
model.add(GRU(64, return_sequences=True, input_shape=(30,126)))
model.add(Dropout(0.2))
model.add(GRU(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(64))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu')) 
model.add(Dropout(0.2))
model.add(Dense(actions.shape[0], activation='softmax'))

# Model GRU overfit 150
# Model GRU MY AI 210epch 0.9891304347826086 chay thi hoi loi
# model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662))) cai nay co face
# Model Gru Drop on dinh 0.9827898550724637
#Mode GRU MIAI DENSE 0.9764492753623188 loss kha nhiu Chay thuc nghiem kha thanh con hen GRUMIAI 
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

history = model.fit(X_train, y_train ,epochs=210, callbacks=[tb_callback])

model.summary()

#Test
res = model.predict(X_test)


# res => giong cai res o tren
# res = [.7,0.2,0.1]  # => 0
# actions[np.argmax(res)] # => 'hello'

model.save('ModelGruMIAI.h5')
model.load_weights('ModelGruMIAI.h5')


from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
# Train test acc
yhat = model.predict(X_train)


ytrue = np.argmax(y_train, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print(accuracy_score(ytrue, yhat))


import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 210), history.history["loss"], label="Mất mát khi trainning")
plt.plot(np.arange(0, 210), history.history["categorical_accuracy"], label="Độ chính xác khi trainning")
plt.title("Biểu đồ hiển thị mất mát trong Training và độ chính xác")
plt.xlabel("Epoch #")
plt.ylabel("Mất mát/Độ chính xác")
plt.legend()
plt.show()