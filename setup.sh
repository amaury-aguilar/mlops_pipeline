#!/bin/bash

# Fallamos inmediatamente si algun comando retorna error.
set -euo pipefail

# Definimos ruta raiz del proyecto para construir rutas absolutas reproducibles.
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Definimos el path del entorno unico del proyecto.
ENV_PATH="$PROJECT_ROOT/mlops-venv"

# Definimos el manifiesto declarativo del entorno.
ENV_FILE="$PROJECT_ROOT/environment.yml"

# Definimos binario de conda (puede sobrescribirse via variable de entorno CONDA_BIN).
CONDA_BIN="${CONDA_BIN:-/Users/amaury/miniconda3/bin/conda}"

# Informamos el objetivo del script.
echo ""
echo "=== Reconstruccion del entorno del proyecto (100% conda) ==="
echo ""

# Validamos existencia del ejecutable conda.
if [ ! -x "$CONDA_BIN" ]; then
	echo "ERROR: No se encontro conda en: $CONDA_BIN"
	echo "Define CONDA_BIN con la ruta correcta, por ejemplo:"
	echo "  CONDA_BIN=/ruta/a/conda bash setup.sh"
	exit 1
fi

# Validamos existencia del manifiesto declarativo del entorno.
if [ ! -f "$ENV_FILE" ]; then
	echo "ERROR: No se encontro el archivo de entorno: $ENV_FILE"
	echo "Asegurate de tener environment.yml en la raiz del proyecto."
	exit 1
fi

# Eliminamos el entorno anterior si existe para recrearlo de forma limpia y reproducible.
if [ -d "$ENV_PATH" ]; then
	echo "Eliminando entorno previo: $ENV_PATH"
	rm -rf "$ENV_PATH"
fi

# Creamos el entorno unico a partir del manifiesto declarativo.
echo "Creando entorno conda en: $ENV_PATH"
"$CONDA_BIN" env create -y -p "$ENV_PATH" -f "$ENV_FILE"

# Mostramos version activa de Python dentro del entorno para trazabilidad.
echo ""
echo "Version de Python del entorno:"
"$CONDA_BIN" run -p "$ENV_PATH" python --version

# Registramos el kernel para notebooks de Jupyter.
echo ""
echo "Registrando kernel de Jupyter ..."
"$CONDA_BIN" run -p "$ENV_PATH" python -m ipykernel install --user --name="mlops-venv" --display-name="mlops-venv"

# Cerramos con comandos sugeridos para ejecutar scripts de forma reproducible.
echo ""
echo "Setup completado correctamente."
echo ""
echo "Ejecuta scripts con:"
echo "  $CONDA_BIN run -p \"$ENV_PATH\" python src/ft_engineering.py"
echo "  $CONDA_BIN run -p \"$ENV_PATH\" python src/model_training_evaluation.py"
