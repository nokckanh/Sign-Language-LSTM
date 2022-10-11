from cProfile import label
from curses import window
from msilib import sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
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
             res = np.load(os.path.join(DATA_PATH, action, str(sequence), "()пpу" ,format(frame_num)))
        sequences.append(window)
        labels.append(label_map[action])



