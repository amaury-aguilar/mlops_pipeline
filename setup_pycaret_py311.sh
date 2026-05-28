#!/usr/bin/env bash

# Script para preparar un entorno compatible con PyCaret sin tocar el entorno actual.
# Requiere que python3.11 este instalado en la maquina.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="$PROJECT_ROOT/mlops-venv-py311"

if ! command -v python3.11 >/dev/null 2>&1; then
  echo "ERROR: python3.11 no esta instalado en esta maquina."
  echo "Instala Python 3.11 y vuelve a ejecutar este script."
  exit 1
fi

echo "[1/5] Creando entorno virtual Python 3.11 en: $VENV_PATH"
python3.11 -m venv "$VENV_PATH"

echo "[2/5] Activando entorno"
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

echo "[3/5] Actualizando pip/setuptools/wheel"
pip install --upgrade pip setuptools wheel

echo "[4/5] Instalando dependencias compatibles de PyCaret"
pip install -r "$PROJECT_ROOT/requirements-pycaret-py311.txt"

echo "[5/5] Ejecutando entrenamiento y evaluacion"
python "$PROJECT_ROOT/src/model_training_evaluation.py"

echo "Listo. Entorno preparado en: $VENV_PATH"
