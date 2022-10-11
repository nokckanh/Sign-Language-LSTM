from cProfile import label
from msilib import sequence
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
import os

actions = np.array(['hello', 'thks','iloveu'])

label_map = {label:num for num , label in enumerate(actions)}

#print(label_map)

DATA_PATH = os.path.join('MP_DATA')

# hanh dong
actions = np.array(['hello', 'thks','iloveu'])

# thu thap 30 chuoi video
no_sequences = 30

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

# Chuan bi du lieu de train 
y = to_categorical(labels).astype(int)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05) 

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Layer
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
 
# muc dich cua actions.shape[0] => 3 
# vi du 


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])

model.summary()

#Test
res = model.predict(X_test)

# res => giong cai res o tren
# res = [.7,0.2,0.1]  # => 0
# actions[np.argmax(res)] # => 'hello'

model.save('sign.h5')

del model

model.load_weights('sign.h5')

# Danh gia model

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

yhat = model.predict(X_test)


ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()


multilabel_confusion_matrix(ytrue, yhat)

accuracy_score(ytrue, yhat)
