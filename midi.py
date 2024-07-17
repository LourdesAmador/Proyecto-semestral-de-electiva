import os
import pretty_midi


def cargar_archivos_midi(ruta_directorio):
    secuencias_notas = []

    archivos = os.listdir(ruta_directorio)
    archivos_midi = [archivo for archivo in archivos if archivo.endswith('.mid')]

    for archivo_midi in archivos_midi:
        ruta_completa = os.path.join(ruta_directorio, archivo_midi)
        try:
            midi_data = pretty_midi.PrettyMIDI(ruta_completa)

            # Verifica si hay eventos de cambio de tempo, clave o firma de tiempo en pistas no cero
            if any(track.is_drum for track in midi_data.instruments):
                print(
                    f"Advertencia: El archivo {archivo_midi} contiene eventos en pistas no válidas (tipo 0 o tipo 1).")
                continue

            notas = midi_data.instruments[0].notes
            secuencia_notas = [(nota.pitch, nota.start, nota.end) for nota in notas]
            secuencias_notas.append(secuencia_notas)
        except Exception as e:
            print(f"Error al procesar {archivo_midi}: {str(e)}")

    return secuencias_notas


# Ejemplo de uso
ruta_directorio_midi = 'data/midi'
datos_midi = cargar_archivos_midi(ruta_directorio_midi)
print(f"Se han cargado {len(datos_midi)} archivos MIDI válidos.")

