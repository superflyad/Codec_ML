import pickle
from keras.utils import to_categorical
from music21 import instrument, note, chord, stream
from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import ModelCheckpoint
from music21 import converter, note, chord, stream

# Load the MusicXML file
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

unique_elements = set(sample_sequence)

# Define notes and chords
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

chords = [
    'Cmaj', 'Cm', 'C7', 'Cm7', 'Cmaj7',
    'C#maj', 'Dbmaj', 'C#m', 'Dbm', 'C#7', 'Db7', 'C#m7', 'Dbm7', 'C#maj7', 'Dbmaj7',
    'Dmaj', 'Dm', 'D7', 'Dm7', 'Dmaj7',
    'D#maj', 'Ebmaj', 'D#m', 'Ebm', 'D#7', 'Eb7', 'D#m7', 'Ebm7', 'D#maj7', 'Ebmaj7',
    'Emaj', 'Em', 'E7', 'Em7', 'Emaj7',
    'Fmaj', 'Fm', 'F7', 'Fm7', 'Fmaj7',
    'F#maj', 'Gbmaj', 'F#m', 'Gbm', 'F#7', 'Gb7', 'F#m7', 'Gbm7', 'F#maj7', 'Gbmaj7',
    'Gmaj', 'Gm', 'G7', 'Gm7', 'Gmaj7',
    'G#maj', 'Abmaj', 'G#m', 'Abm', 'G#7', 'Ab7', 'G#m7', 'Abm7', 'G#maj7', 'Abmaj7',
    'Amaj', 'Am', 'A7', 'Am7', 'Amaj7',
    'A#maj', 'Bbmaj', 'A#m', 'Bbm', 'A#7', 'Bb7', 'A#m7', 'Bbm7', 'A#maj7', 'Bbmaj7',
    'Bmaj', 'Bm', 'B7', 'Bm7', 'Bmaj7'
]

durations = [
    '0.0625', '0.125', '0.1875', '0.25', '0.3333', '0.375', '0.5', '0.6667', '0.75',
    '1.0', '1.5', '2.0', '3.0', '4.0', '6.0', '8.0'
]
velocities = [str(i) for i in range(0, 128)]  # MIDI velocities

# Combine elements
elements = notes + chords + durations + velocities + list(unique_elements) + ['None']

# Encode elements
encoder = LabelEncoder()
encoder.fit(elements)

# Ensure all elements in sample_sequence are in the encoder's classes
for element in sample_sequence:
    if element not in encoder.classes_:
        print(f"Unrecognized element: {element}")

# Encode the sequence
encoded_sequence = encoder.transform(sample_sequence)

# Prepare input and output sequences
sequence_length = 64
X = []
y = []

for i in range(len(encoded_sequence) - sequence_length):
    X.append(encoded_sequence[i:i + sequence_length])
    y.append(encoded_sequence[i + sequence_length])

X = np.array(X)
y = np.array(y)
y = to_categorical(y, num_classes=len(elements))

# Reshape X for LSTM
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Check shapes of X and y
print("Shape of X:", X.shape)  # Should be (number_of_samples, sequence_length, 1)
print("Shape of y:", y.shape)  # Should be (number_of_samples, num_classes)

# Define the model with increased complexity
model = Sequential()
model.add(Bidirectional(LSTM(256, return_sequences=True), input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(256)))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(elements), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define the checkpoint
filepath = "model_weights_genI_complex_V1.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Train the model
model.fit(X, y, epochs=1000, batch_size=64, callbacks=callbacks_list)

# Save the model
model.save("music_generation_model_genI_complex_V1.h5")
encoder_filename = 'label_encoder_genI_complex_V1.pkl'
with open(encoder_filename, 'wb') as file:
    pickle.dump(encoder, file) #end