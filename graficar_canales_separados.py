#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import os

# ================= CONFIGURACIÓN =================
ARCHIVO = "recording_20251104_022518_eeg.csv"  # Cambia si quieres otro
FILTRO = True                 # True = filtrar 1-40 Hz
FS = 256                      # Hz (Muse S)
LOW = 1
HIGH = 40
# ================================================

# Verificar archivo
if not os.path.exists(ARCHIVO):
    print(f"Archivo no encontrado: {ARCHIVO}")
    exit(1)

print(f"Cargando: {ARCHIVO}...")
df = pd.read_csv(ARCHIVO)

# Mapeo de canales genéricos a nombres Muse
# Orden estándar Muse S: TP9, AF7, AF8, TP10
CANAL_MAP = {
    'ch0': 'TP9',
    'ch1': 'AF7', 
    'ch2': 'AF8',
    'ch3': 'TP10'
}

# Renombrar columnas si existen
for old_name, new_name in CANAL_MAP.items():
    if old_name in df.columns:
        df.rename(columns={old_name: new_name}, inplace=True)

# Verificar que tenemos timestamp (sin 's')
if 'timestamp' not in df.columns:
    if 'timestamps' in df.columns:
        df.rename(columns={'timestamps': 'timestamp'}, inplace=True)
    else:
        print("ERROR: No se encontró columna 'timestamp' o 'timestamps'")
        exit(1)

# Tiempo relativo
df['time'] = df['timestamp'] - df['timestamp'].iloc[0]

# Filtrar
def bandpass(data):
    nyq = 0.5 * FS
    low = LOW / nyq
    high = HIGH / nyq
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, data)

canales = ['TP9', 'AF7', 'AF8', 'TP10']
colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Azul, Naranja, Verde, Rojo

# Aplicar filtro si está activado
if FILTRO:
    for canal in canales:
        if canal in df.columns:
            df[canal] = bandpass(df[canal])

# ================= GRÁFICO PROFESIONAL =================
fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)

for i, (canal, color) in enumerate(zip(canales, colores)):
    ax = axes[i]
    if canal in df.columns:
        ax.plot(df['time'], df[canal], color=color, linewidth=1)
        ax.set_ylabel(f'{canal} (µV)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(df[canal].min() * 1.1, df[canal].max() * 1.1)
    else:
        ax.text(0.5, 0.5, f'{canal} no encontrado', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='red')
        ax.set_ylabel(canal, fontsize=12)

axes[-1].set_xlabel('Tiempo (s)', fontsize=12)
duracion = df['time'].iloc[-1]
fig.suptitle(f'EEG Muse S - {os.path.basename(ARCHIVO)}\n'
             f'Filtrado {LOW}-{HIGH} Hz | Duración: {duracion:.1f}s | 256 Hz',
             fontsize=16, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

print(f"¡GRÁFICA LISTA! {len(df)} muestras | {duracion:.1f} segundos")

