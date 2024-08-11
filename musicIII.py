import os
from music21 import converter, instrument, note, chord, stream
import pickle
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense, Activation, BatchNormalization
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf

# Ensure TensorFlow is using the GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"Using GPU: {physical_devices[0]}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Using CPU.")

def process_midi_file(file_path):
    notes = []
    try:
        midi = converter.parse(file_path)
        parts = instrument.partitionByInstrument(midi)
        if parts:  # If there are instrument parts
            for part in parts.parts:
                if 'Piano' in str(part):  # Only consider piano parts
                    for element in part.flat.notes:
                        if isinstance(element, note.Note):
                            notes.append(str(element.pitch))
                        elif isinstance(element, chord.Chord):
                            notes.append('.'.join(str(n) for n in element.normalOrder))
        else:
            for element in midi.flat.notes:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    return notes

def extract_notes_from_midi_folder(folder_path):
    notes = []
    files = []
    for root, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith(".mid") or filename.endswith(".midi"):
                files.append(os.path.join(root, filename))

    with ThreadPoolExecutor(max_workers=10) as executor:
        for result in tqdm(executor.map(process_midi_file, files), total=len(files)):
            notes.extend(result)

    return notes

notes_file = 'notes_master.pkl'
folder_path = 'lmd_matched'

# Check if notes file exists
if os.path.exists(notes_file):
    # Load notes from file
    with open(notes_file, 'rb') as file:
        notes = pickle.load(file)
    print(f'Loaded {len(notes)} notes from {notes_file}')
else:
    # Extract notes from MIDI files in the 'new' folder and all its subfolders
    notes = extract_notes_from_midi_folder(folder_path)

    # Save notes to a file
    with open(notes_file, 'wb') as file:
        pickle.dump(notes, file)
    print(f'Extracted {len(notes)} notes and saved to {notes_file}')

# Verify notes are loaded correctly
if not notes:
    raise ValueError("No notes found. Please check the MIDI files in the 'new' folder.")

# Prepare the dataset
sequence_length = 100  # Reduce sequence length for faster processing
unique_notes = list(set(notes))
note_to_int = dict((note, number) for number, note in enumerate(unique_notes))
int_to_note = dict((number, note) for note, number in note_to_int.items())

# Debugging: Check the number of unique notes
num_unique_notes = len(unique_notes)
print(f'Number of unique notes: {num_unique_notes}')

input_sequences = []
output_notes = []
for i in range(len(notes) - sequence_length):
    input_seq = notes[i:i + sequence_length]
    output_note = notes[i + sequence_length]
    input_sequences.append([note_to_int[note] for note in input_seq])
    output_notes.append(note_to_int[output_note])

# Ensure sequences and outputs are generated correctly
print(f'Number of input sequences: {len(input_sequences)}')
print(f'Number of output notes: {len(output_notes)}')

# Check if input_sequences and output_notes have been correctly populated
if not input_sequences or not output_notes:
    raise ValueError("No sequences or output notes found. Please check the preprocessing steps.")

# Reshape for LSTM input
X = np.reshape(input_sequences, (len(input_sequences), sequence_length, 1))
X = X / float(len(unique_notes))
y = to_categorical(output_notes, num_classes=num_unique_notes)

print(f'Shape of X: {X.shape}')
print(f'Shape of y: {y.shape}')

# Function to build the model
def build_model(input_shape, output_units):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(LSTM(128))
    model.add(Dense(64))
    model.add(Dropout(0.3))
    model.add(Dense(output_units))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Function to check model compatibility
def check_model_compatibility(model, num_unique_notes):
    if model.layers[-1].output_shape[-1] != num_unique_notes:
        print("Output shape mismatch. Rebuilding the model.")
        return False
    return True

# Build or load the model
model_path = 'model_O.h5'
rebuild_model = False

if os.path.exists(model_path):
    model = load_model(model_path)
    if not check_model_compatibility(model, num_unique_notes):
        rebuild_model = True
else:
    rebuild_model = True

if rebuild_model:
    model = build_model((X.shape[1], X.shape[2]), num_unique_notes)
    print("Built new model.")
else:
    print("Loaded model from disk.")

# Train the model
epochs = 100  # Reduced epochs for faster training
batch_size = 64  # Increased batch size for faster training
model.fit(X, y, epochs=epochs, batch_size=batch_size)
model.save(model_path)
print(f"Model trained for {epochs} epochs and saved to {model_path}")

# Generate music
def generate_music(model, int_to_note, sequence_length, unique_notes, num_notes=5000):
    start = np.random.randint(0, len(input_sequences) - 1)
    pattern = input_sequences[start]
    generated_notes = []

    for i in range(num_notes):  # Generate 5000 notes
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(len(unique_notes))
        
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        generated_notes.append(result)
        
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return generated_notes

def create_midi(prediction_output, file_name='outputIIII.mid'):
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        # If it's a chord
        if ('.' in pattern) or pattern.isdigit():
            chord_notes = pattern.split('.')
            notes = []
            for current_note in chord_notes:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # Increase offset each iteration to ensure notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=file_name)

generated_notes = generate_music(model, int_to_note, sequence_length, unique_notes, num_notes=200)  # Reduced notes for faster generation
create_midi(generated_notes, 'outputIIII.mid')
print("Generated music saved to outputIIII.mid")