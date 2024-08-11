import os
import warnings
from keras.callbacks import EarlyStopping, ModelCheckpoint
from music21 import converter, instrument, note, chord, stream
from multiprocessing import Pool
from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from tqdm import tqdm
import pickle

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress specific music21 warnings
warnings.filterwarnings('ignore', category=UserWarning, module='music21')

# Function to parse a single MIDI file
def parse_single_midi(file):
    notes = []
    try:
        midi = converter.parse(file)
        parts = instrument.partitionByInstrument(midi)
        if parts:  # File has instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else:  # File has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    except Exception as e:
        print(f"Error parsing {file}: {e}")
    return notes

# Function to collect all MIDI files in the directory
def collect_midi_files(root_dir):
    midi_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.mid'):
                midi_files.append(os.path.join(root, file))
    return midi_files

# Function to filter piano ballads (simplified example)
def filter_piano_ballads(midi_files):
    piano_ballads = []
    for file in midi_files:
        try:
            midi = converter.parse(file)
            instruments = [str(instr) for instr in midi.getInstruments()]
            if any('Piano' in instr for instr in instruments):
                piano_ballads.append(file)
        except Exception as e:
            print(f"Error filtering {file}: {e}")
    return piano_ballads

# Function to parse MIDI files in parallel with a progress bar
def parse_midi_files_in_parallel(midi_files, num_workers=6):
    with Pool(num_workers) as pool:
        all_notes = []
        for result in tqdm(pool.imap_unordered(parse_single_midi, midi_files), total=len(midi_files)):
            all_notes.append(result)
    # Flatten the list of lists
    return [note for sublist in all_notes for note in sublist]

def main():
    # Define the path to the MIDI files and the output file for parsed data
    midi_dir = 'new'
    parsed_data_file = 'parsed_notesI.pkl'

    # Collect and filter MIDI files for piano ballads
    midi_files = collect_midi_files(midi_dir)
    piano_ballad_files = filter_piano_ballads(midi_files)

    if os.path.exists(parsed_data_file):
        print(f"Loading parsed data from {parsed_data_file}...")
        with open(parsed_data_file, 'rb') as f:
            notes = pickle.load(f)
    else:
        # Parse filtered piano ballad MIDI files
        num_workers = os.cpu_count()
        notes = parse_midi_files_in_parallel(piano_ballad_files, num_workers)
        print(f"Parsed {len(notes)} notes/chords.")
        
        # Save the parsed data to a file
        print(f"Saving parsed data to {parsed_data_file}...")
        with open(parsed_data_file, 'wb') as f:
            pickle.dump(notes, f)

    # Encode the notes to integers
    encoder = LabelEncoder()
    integer_notes = encoder.fit_transform(notes)

    # Create sequences and labels
    sequence_length = 100
    network_input = []
    network_output = []

    for i in range(len(integer_notes) - sequence_length):
        seq_in = integer_notes[i:i + sequence_length]
        seq_out = integer_notes[i + sequence_length]
        network_input.append(seq_in)
        network_output.append(seq_out)

    # Use a smaller subset of data for training
    subset_size = 20000  # Adjust as needed
    network_input = network_input[:subset_size]
    network_output = network_output[:subset_size]

    # Print shapes for debugging
    print(f"Shape of network_input: {np.shape(network_input)}")
    print(f"Shape of network_output: {np.shape(network_output)}")

    # Reshape and normalize input
    n_patterns = len(network_input)
    n_vocab = len(set(integer_notes))
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab)
    network_output = to_categorical(network_output, num_classes=n_vocab)

    # Print unique classes and shapes after reshaping
    print(f"Number of unique classes (notes): {n_vocab}")
    print(f"Shape of network_input after reshaping: {np.shape(network_input)}")
    print(f"Shape of network_output after one-hot encoding: {np.shape(network_output)}")

    print(f"Prepared {n_patterns} sequences for training.")

    # Define and compile a simpler model
    model = Sequential()
    model.add(LSTM(128, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(128))
    model.add(Dense(64))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='loss', patience=10)

    # Model checkpoint callback
    checkpoint = ModelCheckpoint('model_checkpoint.h5', save_best_only=True, monitor='loss', mode='min')

    # Train the model with fewer epochs
    model.fit(network_input, network_output, epochs=100, batch_size=128, callbacks=[early_stopping, checkpoint])

    # Function to generate notes
    def generate_notes(model, network_input, n_vocab):
        start = np.random.randint(0, len(network_input) - 1)
        pattern = network_input[start]

        prediction_output = []

        for note_index in range(500):
            prediction_input = np.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(n_vocab)

            prediction = model.predict(prediction_input, verbose=0)
            index = np.argmax(prediction)
            result = encoder.inverse_transform([index])[0]
            prediction_output.append(result)

            pattern = np.append(pattern, index)
            pattern = pattern[1:len(pattern)]

        return prediction_output

    # Function to create a MIDI file from the generated notes
    def create_midi(prediction_output):
        offset = 0
        output_notes = []

        for pattern in prediction_output:
            if ('.' in pattern) or pattern.isdigit():
                notes_in_chord = pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
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

            offset += 0.5

        midi_stream = stream.Stream(output_notes)
        midi_stream.write('midi', fp='output.mid')

    # Generate and save the MIDI file
    prediction_output = generate_notes(model, network_input, n_vocab)
    create_midi(prediction_output)

if __name__ == '__main__':
    main()