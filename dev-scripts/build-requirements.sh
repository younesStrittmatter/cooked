#!/bin/bash

pipreqs . --force --ignore .venv

# Add gunicorn if it's missing
if ! grep -qi '^gunicorn' requirements.txt; then
  echo "gunicorn==21.2.0" >> requirements.txt
  echo "✅ Added gunicorn to requirements.txt"
else
  echo "✅ gunicorn already in requirements.txt"
fi

# add google-generativeai if it's missing
if ! grep -qi '^google-generativeai' requirements.txt; then
  echo "google-generativeai==0.2.0" >> requirements.txt
  echo "✅ Added google-generativeai to requirements.txt"
else
  echo "✅ google-generativeai already in requirements.txt"
fi

if ! grep -qi '^google-genai' requirements.txt; then
  echo "google-genai==1.12.1" >> requirements.txt
  echo "✅ Added google-genai to requirements.txt"
else
  echo "✅ google-genai already in requirements.txt"
fi

if ! grep -qi '^gevent' requirements.txt; then
  echo "gevent==23.9.1" >> requirements.txt
  echo "✅ Added gevent to requirements.txt"
fi

if ! grep -qi '^flask-socketio' requirements.txt; then
  echo "flask-socketio==5.3.6" >> requirements.txt
  echo "✅ Added flask-socketio to requirements.txt"
fi

if ! grep -qi '^eventlet' requirements.txt; then
  echo "eventlet==0.33.3" >> requirements.txt
  echo "✅ Added eventlet to requirements.txt"
fi


sed -i '' '/^protobuf==/d' requirements.txt
echo "protobuf==4.23.4" >> requirements.txt