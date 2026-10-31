#!/usr/bin/env bash
# ============================================================
# Universal Genomics v23.5 — Setup de reproduction
# Usage: bash setup.sh
# ============================================================
set -euo pipefail

PYTHON_MIN="3.10"
VENV=".venv-genomics"

echo "🐍 Vérification de Python >= $PYTHON_MIN..."
if ! command -v python3 &>/dev/null; then
  echo "❌ Python 3 requis. Installez-le depuis https://python.org"
  exit 1
fi
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "   Version détectée : $PY_VER"

echo "📦 Création du venv '$VENV'..."
python3 -m venv "$VENV"

echo "⬆️  Mise à jour pip..."
"$VENV/bin/pip" install --upgrade pip wheel setuptools

echo "📥 Installation des dépendances..."
"$VENV/bin/pip" install -r requirements.txt

echo ""
echo "✅ Installation terminée."
echo ""
echo "Prochaines étapes :"
echo "  1. Placer genomics_v23_5.py dans le répertoire courant"
echo "  2. Lancer :"
echo "     source $VENV/bin/activate"
echo "     python reproduce.py               # Analyse complète (~10 min)"
echo ""
echo "  3. Pour Streamlit interactif :"
echo "     streamlit run genomics_v23_5.py"
echo ""
echo "  4. Pour générer le notebook Jupyter :"
echo "     jupytext --to notebook reproduce.py"
echo "     jupyter notebook reproduce.ipynb"
