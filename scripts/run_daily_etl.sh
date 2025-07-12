#!/bin/bash

# Ordner des Skripts selbst ermitteln (z.B. scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# .env Datei aus dem Projektroot laden und Conda-Umgebung laden 
set -o allexport
source "$PROJECT_ROOT/.env" || { echo "ERROR: .env file not found!"; exit 1; }
set +o allexport

# Conda initialisieren
if [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/opt/anaconda3/etc/profile.d/conda.sh"
else
    echo "ERROR: conda.sh not found! Please check your conda installation."
    exit 1
fi

# Conda Environment aktivieren
conda activate "$CONDA_ENV_NAME"

# Ausgabe der Aktiven conda umgebung zur kontrolle 
echo "Aktive Conda-Umgebung: $CONDA_DEFAULT_ENV"
which python


# In Projektverzeichnis wechseln
cd "$PROJECT_PATH" || { echo "ERROR: Projektverzeichnis nicht gefunden!"; exit 1; }

# Python-Skript starten mit optionalem Python-Pfad aus .env (falls gesetzt)
# Python-Binary festlegen
PYTHON_BIN="${CONDA_PYTHON_PATH:-python}"

# Python-Skript ausführen
PYTHONPATH=. "$PYTHON_BIN" scripts/run_extract.py
