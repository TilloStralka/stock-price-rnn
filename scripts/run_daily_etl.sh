#!/bin/bash

# Projektpfad und Conda-Umgebung aus .env laden
set -o allexport
source ./.env
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
cd "$PROJECT_PATH"

# Python-Skript starten mit optionalem Python-Pfad aus .env (falls gesetzt)
if [ -n "$CONDA_PYTHON_PATH" ]; then
  PYTHON_BIN="$CONDA_PYTHON_PATH"
else
  PYTHON_BIN="python"
fi

PYTHONPATH=. "$PYTHON_BIN" scripts/run_extract.py
