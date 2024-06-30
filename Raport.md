### RAPORT Z REALIZACJI PROJEKTU: Groan AI

#### 1. Opis zaimplementowanych funkcji

Projekt polegał na stworzeniu aplikacji generującej muzykę z wykorzystaniem sztucznej inteligencji. Poniżej znajduje się szczegółowy opis zaimplementowanych funkcji i kroków realizacji projektu.

##### Generator.ipynb

###### Funkcje
1. **Ładowanie plików MIDI**
    - Funkcja `load_midi_files(folder_path)` przeszukuje podany folder w poszukiwaniu plików MIDI i zwraca ich listę.

2. **Generowanie losowej muzyki**
    - Funkcja `generate_random_music(midi_files, num_pieces, min_measures, max_measures)` generuje nowy utwór muzyczny poprzez losowe wybieranie fragmentów z dostępnych plików MIDI.

3. **Interakcja z użytkownikiem**
    - Program prosi użytkownika o podanie liczby plików MIDI oraz zakresu taktów do wygenerowania losowego utworu muzycznego.

##### Model_prototyp.ipynb

###### Funkcje
1. **Przetwarzanie plików MIDI**
    - Funkcja `midi_to_notes(file_path)` konwertuje pliki MIDI na listę nut.

2. **Przygotowanie danych**
    - Funkcja `prepare_data(midi_files, seq_length)` przygotowuje dane do trenowania modelu.
    - Funkcja `create_sequences(notes, seq_length)` tworzy sekwencje nut używane do trenowania modelu.
    - Funkcja `sequences_to_tensors(sequences)` konwertuje sekwencje na tensory PyTorch.

3. **Tworzenie modelu sieci neuronowej**
    - Klasa `PianoLSTM` definiuje model sieci neuronowej LSTM do generowania muzyki.

4. **Trening modelu**
    - Funkcja `train_model(model, train_inputs, train_targets, num_epochs, learning_rate)` trenuje model na danych treningowych.

#### 2. Użyte narzędzia, biblioteki i frameworki

##### Frontend:
- **Google Colab:** Używany jako środowisko programistyczne.
- **Music21:** Biblioteka do analizy i manipulacji plikami muzycznymi MIDI.

##### Backend (model AI):
- **PyTorch:** Framework do budowy i trenowania modeli sieci neuronowych.
- **Mido:** Biblioteka do przetwarzania plików MIDI.

#### 3. Opis napotkanych problemów

1. **Obsługa błędów podczas ładowania plików MIDI:**
    - Konieczność prawidłowego obsługiwania sytuacji, w których pliki MIDI są uszkodzone lub niekompletne.

2. **Optymalizacja parametrów modelu:**
    - Dobór odpowiednich parametrów modelu LSTM (takich jak liczba warstw, rozmiar warstwy ukrytej) w celu uzyskania najlepszych wyników generowania muzyki.

#### 4. Propozycje dalszych rozszerzeń funkcjonalności projektu

1. **Rozbudowa modelu AI:**
    - Eksperymentowanie z różnymi architekturami sieci neuronowych w celu poprawy jakości generowanej muzyki.

2. **Interfejs użytkownika:**
    - Stworzenie przyjaznego interfejsu użytkownika umożliwiającego łatwiejszą interakcję z aplikacją.

3. **Wykorzystanie dodatkowych źródeł danych:**
    - Dodanie możliwości importu i analizy plików muzycznych z różnych źródeł (np. plików audio, notacji muzycznej).

#### Odnośniki:
1. **Pliki MIDI:** [Link do dysku z plikami MIDI](https://drive.google.com/drive/folders/19gxwtwRyicWi4KOZC7LbM9-sz20s__eZ)
2. **Wygenerowana muzyka:** [Link do wygenerowanych plików](https://drive.google.com/drive/folders/1UR9vKEspERbCipf3xYm9FS1ahk9ZnLt9)
3. **Generator:** [Link do notebooka z generatorem](https://colab.research.google.com/drive/1B4pAXS9-jzwWrTLpOZ0tKHMBG0LoEBHh#scrollTo=q3a6zIqNrKuM)
4. **Model:** [Link do notebooka z modelem](https://colab.research.google.com/drive/1VOoD7d8BsoQsgswf-VFj9oke6KxfWQJT?usp=sharing)

### Źródła
- https://www.youtube.com/watch?v=RLYoEyIHL6A&ab_channel=DogaOzgon
- https://www.youtube.com/watch?v=Wpkn4l5uTZY&ab_channel=Nobody%26TheComputer
- https://github.com/flowese/UdioWrapper
- https://www.youtube.com/watch?v=QUT1VHiLmmI&ab_channel=freeCodeCamp.org
- https://stackoverflow.com/questions/76657708/ai-music-generator-plays-the-same-note-over-and-over





#### Przygotowali:

- Katarzyna Lisiecka 54163
- Zuzanna Kroczak 52718
- Oskar Krzysztofek 52720
- Kacper Lewicki 54162
- Łukasz Kuliński 52725