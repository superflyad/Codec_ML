from fractions import Fraction
import pickle
from keras.models import load_model
from keras.utils import to_categorical
from music21 import instrument, note, chord, stream, converter
from sklearn.preprocessing import LabelEncoder
import numpy as np

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
model = load_model("music_generation_model_super_complex_V1.h5")

# Load the label encoder
encoder_filename = 'label_encoder_super_complex_V1.pkl'
with open(encoder_filename, 'rb') as file:
    encoder = pickle.load(file)

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

# Encode the sequence
encoded_sequence = encoder.transform(sample_sequence)

# Prepare input and output sequences
sequence_length = 64  # Increased sequence length for more context
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

# Generate music from a seed
def generate_music(model, seed, length=1000):  # Increased length for a longer MIDI file
    generated = []
    current_input = seed
    for _ in range(length):
        prediction = model.predict(current_input)
        predicted_index = np.argmax(prediction)
        predicted_element = encoder.inverse_transform([predicted_index])[0]
        generated.append(predicted_element)
        current_input = np.append(current_input[:, 1:, :], [[[predicted_index]]], axis=1)
    return generated

seed = encoded_sequence[:sequence_length]
seed = np.reshape(seed, (1, sequence_length, 1))
generated_sequence = generate_music(model, seed)

# Convert the generated sequence to MIDI
output_notes = []
for element in generated_sequence:
    parts = element.split('_')
    if len(parts) == 3:
        note_name, duration_str, velocity = parts[0], parts[1], int(parts[2]) if parts[2] != 'None' else 64
        try:
            duration = float(duration_str)  # Try to convert directly to float
        except ValueError:
            duration = float(Fraction(duration_str))  # Convert fractional string to float
        if note_name in notes:
            new_note = note.Note(note_name)
            new_note.quarterLength = duration
            new_note.volume.velocity = velocity
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        elif '.' in note_name:  # Chord
            chord_notes = note_name.split('.')
            chord_notes_obj = [note.Note(n) for n in chord_notes]
            new_chord = chord.Chord(chord_notes_obj)
            new_chord.quarterLength = duration
            new_chord.volume.velocity = velocity
            new_chord.storedInstrument = instrument.Piano()
            output_notes.append(new_chord)
    elif parts[0] == "Rest":
        try:
            duration = float(parts[1])  # Try to convert directly to float
        except ValueError:
            duration = float(Fraction(parts[1]))  # Convert fractional string to float
        new_rest = note.Rest()
        new_rest.quarterLength = duration
        output_notes.append(new_rest)

midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='generated_music_longer_complex_V4.mid')