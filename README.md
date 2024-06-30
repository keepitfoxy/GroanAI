# GroanAI
Generator of Random Oddball Audio Nonsense

## Opis projektu
Projekt polega na generowaniu muzyki za pomocą sztucznej inteligencji. Wykorzystano Google Colab do implementacji algorytmów oraz trenowania modeli. Projekt składa się z dwóch głównych części: generatora muzyki oraz modelu LSTM do predykcji nut.

## Pliki projektu
- `Generator.ipynb` - Notebook zawierający kod do generowania losowej muzyki z plików MIDI.
- `Model_prototyp.ipynb` - Notebook zawierający kod do przetwarzania plików MIDI oraz trenowania modelu LSTM.

## Funkcjonalności
### Generator.ipynb
1. **Ładowanie plików MIDI** - Funkcja `load_midi_files(folder_path)` przeszukuje podany folder w poszukiwaniu plików MIDI i zwraca ich listę.
2. **Generowanie losowej muzyki** - Funkcja `generate_random_music(midi_files, num_pieces, min_measures, max_measures)` generuje nowy utwór muzyczny poprzez losowe wybieranie fragmentów z dostępnych plików MIDI.
3. **Interakcja z użytkownikiem** - Program prosi użytkownika o podanie liczby plików MIDI oraz zakresu taktów do wygenerowania losowego utworu muzycznego.

### Model_prototyp.ipynb
1. **Przetwarzanie plików MIDI** - Funkcja `midi_to_notes(file_path)` konwertuje pliki MIDI na listę nut.
2. **Przygotowanie danych** - Funkcja `prepare_data(midi_files, seq_length)` przygotowuje dane do trenowania modelu.
3. **Tworzenie modelu sieci neuronowej** - Klasa `PianoLSTM` definiuje model sieci neuronowej LSTM do generowania muzyki.
4. **Trening modelu** - Funkcja `train_model(model, train_inputs, train_targets, num_epochs, learning_rate)` trenuje model na danych treningowych.

## Wymagania
- Google Colab
- Biblioteki: `music21`, `mido`, `torch`

## Instrukcja uruchomienia
1. **Generator muzyki**:
    - Otwórz [Generator.ipynb](https://colab.research.google.com/drive/1B4pAXS9-jzwWrTLpOZ0tKHMBG0LoEBHh#scrollTo=q3a6zIqNrKuM) w Google Colab.
    - Zamontuj swój Dysk Google.
    - Uruchom wszystkie komórki, aby wygenerować nowy utwór muzyczny.

2. **Model LSTM**:
    - Otwórz [Model_prototyp.ipynb](https://colab.research.google.com/drive/1VOoD7d8BsoQsgswf-VFj9oke6KxfWQJT?usp=sharing) w Google Colab.
    - Zamontuj swój Dysk Google.
    - Uruchom wszystkie komórki, aby przetworzyć pliki MIDI i wytrenować model LSTM.

## Źródła
1. **Pliki MIDI:** [Link do dysku z plikami MIDI](https://drive.google.com/drive/folders/19gxwtwRyicWi4KOZC7LbM9-sz20s__eZ)
2. **Wygenerowana muzyka:** [Link do wygenerowanych plików](https://drive.google.com/drive/folders/1UR9vKEspERbCipf3xYm9FS1ahk9ZnLt9)

## Autorzy
- Katarzyna Lisiecka 54163
- Zuzanna Kroczak 52718
- Oskar Krzysztofek 52720
- Kacper Lewicki 54162
- Łukasz Kuliński 52725
