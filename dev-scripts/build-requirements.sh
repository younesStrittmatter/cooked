#!/bin/bash

pipreqs . --force --ignore .venv

# Add gunicorn if it's missing
if ! grep -qi '^gunicorn' requirements.txt; then
  echo "gunicorn==21.2.0" >> requirements.txt
  echo "✅ Added gunicorn to requirements.txt"
else
  echo "✅ gunicorn already in requirements.txt"
fi