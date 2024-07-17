import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import pretty_midi
from midiutil import MIDIFile
from mingus.core import chords

# Directorio para almacenar archivos generados
DIRECTORIO_MUSICA_GENERADA = 'musica_generada'

# Crear el directorio si no existe
if not os.path.exists(DIRECTORIO_MUSICA_GENERADA):
    os.makedirs(DIRECTORIO_MUSICA_GENERADA)

# Convertir acordes a secuencias de notas
def chord_to_notes(chord):
    return chords.from_shorthand(chord)

# Convertir una secuencia de acordes a una secuencia de notas
def chords_to_note_sequences(chord_progression):
    note_sequences = []
    for chord in chord_progression:
        note_sequences.extend(chord_to_notes(chord))
    return note_sequences

# Cargar archivos MIDI y extraer secuencias de notas
def cargar_archivos_midi(ruta_directorio):
    secuencias_notas = []
    archivos = os.listdir(ruta_directorio)
    archivos_midi = [archivo for archivo in archivos if archivo.endswith('.mid')]

    for archivo_midi in archivos_midi:
        ruta_completa = os.path.join(ruta_directorio, archivo_midi)
        try:
            midi_data = pretty_midi.PrettyMIDI(ruta_completa)
            if any(track.is_drum for track in midi_data.instruments):
                print(f"Advertencia: El archivo {archivo_midi} contiene eventos en pistas no válidas (tipo 0 o tipo 1).")
                continue

            notas = midi_data.instruments[0].notes
            secuencia_notas = [(pretty_midi.note_number_to_name(nota.pitch), nota.start, nota.end) for nota in notas]
            secuencias_notas.append(secuencia_notas)
        except Exception as e:
            print(f"Error al procesar {archivo_midi}: {str(e)}")

    return secuencias_notas

# Ejemplo de uso para cargar archivos MIDI
ruta_directorio_midi = 'data/midi'
datos_midi = cargar_archivos_midi(ruta_directorio_midi)
print(f"Se han cargado {len(datos_midi)} archivos MIDI válidos.")

# Preparar datos de entrenamiento
note_indices = [pretty_midi.note_name_to_number(nota[0]) for secuencia in datos_midi for nota in secuencia]

# Crear secuencias de entrenamiento para la red neuronal
sequence_length = 10
X = []
y = []

for i in range(len(note_indices) - sequence_length):
    X.append(note_indices[i:i + sequence_length])
    y.append(note_indices[i + sequence_length])

X = np.array(X)
y = np.array(y)

# Normalizar los datos
X = X / 127.0  # Normalizar los valores de las notas MIDI (0-127)
y = y / 127.0

# Construir el modelo de red neuronal LSTM
model = Sequential([
    LSTM(128, input_shape=(sequence_length, 1), return_sequences=True),
    LSTM(128),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
X = np.expand_dims(X, axis=-1)  # Agregar dimensión para LSTM
model.fit(X, y, epochs=50, batch_size=32)

# Generar música aleatoria
def generate_random_music(model, start_sequence, length=50):
    generated_notes = list(start_sequence)
    current_sequence = list(start_sequence)

    for _ in range(length):
        X_input = np.expand_dims(np.array(current_sequence[-sequence_length:]) / 127.0, axis=0)
        prediction = model.predict(X_input)
        next_note = int(prediction * 127.0)
        generated_notes.append(next_note)
        current_sequence.append(next_note)

    return generated_notes

# Generar música aleatoria desde un punto de partida aleatorio
start_index = np.random.randint(0, len(note_indices) - sequence_length)
start_sequence = note_indices[start_index:start_index + sequence_length]
generated_note_sequence = generate_random_music(model, start_sequence)

# Convertir la secuencia generada a un archivo MIDI
def create_midi_file(note_sequence, filename="generated_music.mid"):
    MyMIDI = MIDIFile(1)
    track = 0
    channel = 0
    time = 0
    duration = 1
    tempo = 120
    volume = 100

    MyMIDI.addTempo(track, time, tempo)

    for i, pitch in enumerate(note_sequence):
        MyMIDI.addNote(track, channel, pitch, time + i, duration, volume)

    with open(os.path.join(DIRECTORIO_MUSICA_GENERADA, filename), "wb") as output_file:
        MyMIDI.writeFile(output_file)

# Crear archivo MIDI con la secuencia generada y guardar en el directorio especificado
create_midi_file(generated_note_sequence, f"generated_music_{np.random.randint(1000)}.mid")
