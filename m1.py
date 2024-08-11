from keras.models import load_model, Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
from music21 import converter, note, chord, instrument
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle

# Define all possible notes and chords
notes = [
    'C', 'C#', 'Db', 'D', 'D#', 'Eb', 'E', 'Fb', 'F', 'F#', 'Gb', 'G', 'G#', 'Ab', 'A', 'A#', 'Bb', 'B', 'Cb',
    'C1', 'C#1', 'Db1', 'D1', 'D#1', 'Eb1', 'E1', 'Fb1', 'F1', 'F#1', 'Gb1', 'G1', 'G#1', 'Ab1', 'A1', 'A#1', 'Bb1', 'B1', 'Cb1',
    'C2', 'C#2', 'Db2', 'D2', 'D#2', 'Eb2', 'E2', 'Fb2', 'F2', 'F#2', 'Gb2', 'G2', 'G#2', 'Ab2', 'A2', 'A#2', 'Bb2', 'B2', 'Cb2',
    'C3', 'C#3', 'Db3', 'D3', 'D#3', 'Eb3', 'E3', 'Fb3', 'F3', 'F#3', 'Gb3', 'G3', 'G#3', 'Ab3', 'A3', 'A#3', 'Bb3', 'B3', 'Cb3',
    'C4', 'C#4', 'Db4', 'D4', 'D#4', 'Eb4', 'E4', 'Fb4', 'F4', 'F#4', 'Gb4', 'G4', 'G#4', 'Ab4', 'A4', 'A#4', 'Bb4', 'B4', 'Cb4',
    'C5', 'C#5', 'Db5', 'D5', 'D#5', 'Eb5', 'E5', 'Fb5', 'F5', 'F#5', 'Gb5', 'G5', 'G#5', 'Ab5', 'A5', 'A#5', 'Bb5', 'B5', 'Cb5',
    'C6', 'C#6', 'Db6', 'D6', 'D#6', 'Eb6', 'E6', 'Fb6', 'F6', 'F#6', 'Gb6', 'G6', 'G#6', 'Ab6', 'A6', 'A#6', 'Bb6', 'B6', 'Cb6',
    'C7', 'C#7', 'Db7', 'D7', 'D#7', 'Eb7', 'E7', 'Fb7', 'F7', 'F#7', 'Gb7', 'G7', 'G#7', 'Ab7', 'A7', 'A#7', 'Bb7', 'B7', 'Cb7',
    'C8'
]

# Load the model
model = load_model("music_generation_model_retrained_complex_V1.h5")

# Load the label encoder
encoder_filename = 'label_encoder_retrained_complex_V1.pkl'
with open(encoder_filename, 'rb') as file:
    encoder = pickle.load(file)

# Load the MusicXML file (for retraining or new data)
score = converter.parse('Chopin_-_Nocturne_Op_9_No_1_B_Flat_Minor.mxl')

# Extract notes, chords, and their attributes
sample_sequence = []
for element in score.flat.notesAndRests:
    if isinstance(element, note.Note):
        sample_sequence.append(f"{element.nameWithOctave}_{element.quarterLength}_{element.volume.velocity}")
    elif isinstance(element, chord.Chord):
        chord_str = '.'.join(n.nameWithOctave for n in element.notes)
        sample_sequence.append(f"{chord_str}_{element.quarterLength}_{element.volume.velocity}")
    elif isinstance(element, note.Rest):
        sample_sequence.append(f"Rest_{element.quarterLength}")

# Combine all unique elements
unique_elements = list(set(sample_sequence))

# Encode the sequence
encoder.fit(unique_elements + encoder.classes_.tolist())
encoded_sequence = encoder.transform(sample_sequence)

# Prepare input and output sequences
sequence_length = 16  # Increased sequence length for more context
X = []
y = []

for i in range(len(encoded_sequence) - sequence_length):
    X.append(encoded_sequence[i:i + sequence_length])
    y.append(encoded_sequence[i + sequence_length])

X = np.array(X)
y = np.array(y)
y = to_categorical(y, num_classes=len(encoder.classes_))

# Reshape X for LSTM
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Define the model with added complexity
model = Sequential()
model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dense(len(encoder.classes_), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Retrain the model
model.fit(X, y, epochs=500, batch_size=64)

# Save the retrained model
model.save("music_generation_model_retrained_complex_V2.h5")

# Save the updated encoder
encoder_filename = 'label_encoder_retrained_complex_V2.pkl'
with open(encoder_filename, 'wb') as file:
    pickle.dump(encoder, file)