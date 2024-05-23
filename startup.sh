#!/bin/sh
echo "Installing libgl1 and libglib2.0-0"
apt-get update
apt-get install -y libgl1-mesa-glx libglib2.0-0

# Start the Gunicorn server
exec gunicorn --timeout 600 app:server
