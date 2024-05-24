#!/bin/sh

echo "Running startup script to install dependencies"
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

echo "Starting the application"
exec gunicorn --timeout 3600 app:server