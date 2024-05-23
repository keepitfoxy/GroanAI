import os
import numpy as np
from music21 import converter, instrument, note, chord

def parse_midi_files(midi_folder):
    # Inicjalizacja listy na nuty
    notes = []

    # Iteracja przez wszystkie pliki w folderze
    for file in os.listdir(midi_folder):
        # Sprawdzenie, czy plik jest plikiem MIDI
        if file.endswith(".midi"):
            # Konwersja pliku MIDI na strukturę danych music21
            midi = converter.parse(os.path.join(midi_folder, file))

            print(f"Parsing {file}")

            notes_to_parse = None

            try:  
                # Jeśli plik MIDI ma części instrumentów, podziel go na części
                s2 = instrument.partitionByInstrument(midi)
                # Wybierz nuty do analizy z pierwszej części
                notes_to_parse = s2.parts[0].recurse() 
            except TypeError:  
                # Jeśli plik MIDI nie ma części instrumentów, wybierz wszystkie nuty
                notes_to_parse = midi.flat.notes

            # Iteracja przez wybrane nuty
            for element in notes_to_parse:
                # Jeśli element jest nutą, dodaj jej wysokość do listy nut
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                # Jeśli element jest akordem, dodaj jego normalną kolejność do listy nut
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))

    # Zwróć listę nut
    return notes

def prepare_sequences(notes, n_vocab):
    # Sprawdź, czy n_vocab jest większe od 0
    if n_vocab <= 0:
        raise ValueError("n_vocab musi być większe od 0")

    sequence_length = 100

    # Pobierz wszystkie nazwy nut
    pitchnames = sorted(set(item for item in notes))

    # Stwórz słownik do mapowania nut na liczby całkowite
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # Stwórz sekwencje wejściowe i odpowiadające im wyjścia
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # Przekształć wejście do formatu kompatybilnego z warstwami LSTM
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # Normalizuj wejście
    network_input = network_input / float(n_vocab)

    return network_input, network_output