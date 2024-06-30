### Dokumentacja techniczna do projektu: Groan AI

#### Przegląd projektu
Celem tego projektu jest generowanie muzyki za pomocą sztucznej inteligencji. Implementacja została wykonana w Google Colab, a poniżej znajduje się szczegółowa dokumentacja kodu i procesu generowania muzyki.

#### Plik: Generator.ipynb

##### Importowanie bibliotek i montowanie Dysku Google
Na początku notebooka importowane są niezbędne biblioteki oraz montowany jest Dysk Google, aby uzyskać dostęp do plików MIDI:

```python
from google.colab import drive
import os
import random
from music21 import converter, stream, note

drive.mount('/content/drive')

midi_folder_path = '/content/drive/MyDrive/MidiFiles1/'
```

##### Funkcja do ładowania plików MIDI
Funkcja `load_midi_files` przeszukuje podany folder w poszukiwaniu plików MIDI i zwraca ich listę:

```python
def load_midi_files(folder_path):
    midi_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.mid') or filename.endswith('.midi'):
            midi_files.append(os.path.join(folder_path, filename))
    return midi_files

midi_files = load_midi_files(midi_folder_path)
```

##### Funkcja do generowania losowej muzyki
Funkcja `generate_random_music` generuje nowy utwór muzyczny poprzez losowe wybieranie fragmentów z dostępnych plików MIDI:

```python
def generate_random_music(midi_files, num_pieces=3, min_measures=2, max_measures=4):
    random_notes = stream.Stream()

    for _ in range(num_pieces):
        random.shuffle(midi_files)

        for midi_file in midi_files:
            midi_stream = converter.parse(midi_file)

            start_measure = random.randint(0, len(midi_stream.parts[0].getElementsByClass(stream.Measure)) - max_measures)
            end_measure = start_measure + random.randint(min_measures, max_measures)

            fragment_notes = stream.Stream()
            for measure in midi_stream.parts[0].measures(start_measure, end_measure).flat.notes:
                fragment_notes.append(measure)

            random_notes.append(fragment_notes)

    output_midi_file = '/content/drive/MyDrive/WygenerowanyUtwor/wygenerowany_utwor.mid'
    random_notes.write('midi', fp=output_midi_file)

    return output_midi_file
```

##### Interakcja z użytkownikiem
Kod zawiera również interaktywne elementy do wprowadzania danych przez użytkownika:

```python
try:
    num_pieces = int(input("Podaj liczbę plików MIDI, które mają zostać uwzględnione w generowanym utworze: "))
    min_measures = int(input("Podaj minimalną liczbę taktów dla każdego wybranego fragmentu: "))
    max_measures = int(input("Podaj maksymalną liczbę taktów dla każdego wybranego fragmentu: "))

    generated_music_file = generate_random_music(midi_files, num_pieces, min_measures, max_measures)
    if generated_music_file:
        print("Wygenerowano utwór: ", generated_music_file)

except ValueError:
    print("Błąd: Wprowadzono nieprawidłowe wartości. Upewnij się, że wprowadzasz liczby całkowite.")
```

#### Plik: Model_prototyp.ipynb

Teraz przeanalizuję zawartość notebooka `Model_prototyp.ipynb`.

### Dokumentacja techniczna do projektu: Generowanie muzyki z wykorzystaniem AI

#### Przegląd projektu
Celem tego projektu jest generowanie muzyki za pomocą sztucznej inteligencji. Implementacja została wykonana w Google Colab, a poniżej znajduje się szczegółowa dokumentacja kodu i procesu generowania muzyki.

#### Plik: Generator.ipynb

##### Importowanie bibliotek i montowanie Dysku Google
Na początku notebooka importowane są niezbędne biblioteki oraz montowany jest Dysk Google, aby uzyskać dostęp do plików MIDI:

```python
from google.colab import drive
import os
import random
from music21 import converter, stream, note

drive.mount('/content/drive')

midi_folder_path = '/content/drive/MyDrive/MidiFiles1/'
```

##### Funkcja do ładowania plików MIDI
Funkcja `load_midi_files` przeszukuje podany folder w poszukiwaniu plików MIDI i zwraca ich listę:

```python
def load_midi_files(folder_path):
    midi_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.mid') or filename.endswith('.midi'):
            midi_files.append(os.path.join(folder_path, filename))
    return midi_files

midi_files = load_midi_files(midi_folder_path)
```

##### Funkcja do generowania losowej muzyki
Funkcja `generate_random_music` generuje nowy utwór muzyczny poprzez losowe wybieranie fragmentów z dostępnych plików MIDI:

```python
def generate_random_music(midi_files, num_pieces=3, min_measures=2, max_measures=4):
    random_notes = stream.Stream()

    for _ in range(num_pieces):
        random.shuffle(midi_files)

        for midi_file in midi_files:
            midi_stream = converter.parse(midi_file)

            start_measure = random.randint(0, len(midi_stream.parts[0].getElementsByClass(stream.Measure)) - max_measures)
            end_measure = start_measure + random.randint(min_measures, max_measures)

            fragment_notes = stream.Stream()
            for measure in midi_stream.parts[0].measures(start_measure, end_measure).flat.notes:
                fragment_notes.append(measure)

            random_notes.append(fragment_notes)

    output_midi_file = '/content/drive/MyDrive/WygenerowanyUtwor/wygenerowany_utwor.mid'
    random_notes.write('midi', fp=output_midi_file)

    return output_midi_file
```

##### Interakcja z użytkownikiem
Kod zawiera również interaktywne elementy do wprowadzania danych przez użytkownika:

```python
try:
    num_pieces = int(input("Podaj liczbę plików MIDI, które mają zostać uwzględnione w generowanym utworze: "))
    min_measures = int(input("Podaj minimalną liczbę taktów dla każdego wybranego fragmentu: "))
    max_measures = int(input("Podaj maksymalną liczbę taktów dla każdego wybranego fragmentu: "))

    generated_music_file = generate_random_music(midi_files, num_pieces, min_measures, max_measures)
    if generated_music_file:
        print("Wygenerowano utwór: ", generated_music_file)

except ValueError:
    print("Błąd: Wprowadzono nieprawidłowe wartości. Upewnij się, że wprowadzasz liczby całkowite.")
```

#### Plik: Model_prototyp.ipynb

##### Instalacja i importowanie bibliotek
Na początku notebooka instalowane są niezbędne biblioteki oraz montowany jest Dysk Google:

```python
!pip install mido

from google.colab import drive
drive.mount('/content/drive')
```

##### Sprawdzanie i ładowanie plików MIDI
Kod sprawdza, czy folder z plikami MIDI istnieje, a następnie ładuje te pliki:

```python
import os

midi_folder = '/content/drive/MyDrive/MidiFiles1'
if os.path.exists(midi_folder):
    print("Folder exists")
    files = os.listdir(midi_folder)
    print(f"Files in folder: {files}")
else:
    print("Folder does not exist")

midi_files = [os.path.join(midi_folder, f) for f in files if f.endswith('.mid')]
print(f"MIDI files: {midi_files}")
```

##### Przetwarzanie plików MIDI
Funkcja `midi_to_notes` konwertuje pliki MIDI na nuty:

```python
import mido

def midi_to_notes(file_path):
    midi = mido.MidiFile(file_path)
    notes = []
    for track in midi.tracks:
        for msg in track:
            if not msg.is_meta and msg.type == 'note_on' and msg.velocity > 0:
                note = msg.note
                notes.append(note)
    return notes

# Przykład przetwarzania jednego pliku MIDI
if midi_files:
    notes = midi_to_notes(midi_files[0])
    print(f"Notes: {notes}")
    print(f"Number of notes: {len(notes)}")
else:
    print("No MIDI files found")
```

##### Przygotowanie danych
Funkcja `prepare_data` przygotowuje dane do trenowania modelu:

```python
def prepare_data(midi_files, seq_length):
    data = []
    for midi in midi_files:
        notes = midi_to_notes(midi)
        data.extend(notes)
```

##### Tworzenie modelu sieci neuronowej
Kod definiuje i tworzy model sieci neuronowej LSTM do generowania muzyki:

```python
import torch
import torch.nn as nn
import torch.optim as optim

class PianoLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(PianoLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

input_size = 4  # start, end, pitch, velocity
hidden_size = 128
output_size = 4

model = PianoLSTM(input_size, hidden_size, output_size)
print(model)
```

##### Przygotowanie danych treningowych
Funkcje `create_sequences` i `sequences_to_tensors` przygotowują dane do trenowania modelu:

```python
def create_sequences(notes, seq_length=50):
    sequences = []
    for i in range(len(notes) - seq_length):
        seq = notes[i:i+seq_length]
        target = notes[i+seq_length]
        sequences.append((seq, target))
    return sequences

notes = [
    {'pitch': 60, 'velocity': 80, 'time': 0.0},
    {'pitch': 64, 'velocity': 64, 'time': 0.5},
]

seq_length = 50
sequences = create_sequences(notes, seq_length)

# Podział na dane treningowe i testowe
train_size = int(0.8 * len(sequences))
train_sequences = sequences[:train_size]
test_sequences = sequences[train_size:]

# Przekształcanie na tensory PyTorch
def sequences_to_tensors(sequences):
    inputs = []
    targets = []
    for seq, target in sequences:
        inputs.append(torch.tensor(seq, dtype=torch.float32))
        targets.append(torch.tensor(target, dtype=torch.float32))
    return torch.stack(inputs), torch.stack(targets)

train_inputs, train_targets = sequences_to_tensors(train_sequences)
test_inputs, test_targets = sequences_to_tensors(test_sequences)
```

##### Trening modelu
Funkcja `train_model` trenuje model na danych treningowych:

```python
def train_model(model, train_inputs, train_targets, num_epochs=100, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        outputs = model(train_inputs)
        loss = criterion(outputs, train_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

num_epochs = 100
train_model(model, train_inputs, train_targets, num_epochs)
```