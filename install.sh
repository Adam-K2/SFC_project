#!/bin/bash

# Warn if script is run with bash instead of source
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Note: run this script with 'source install.sh' to keep the environment activated"
fi

INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
URL="https://repo.anaconda.com/miniconda/$INSTALLER"
INSTALL_DIR="$PWD/miniconda"
ENV="environment.yml"

# Check if env file exists
if [ ! -f "$ENV" ]; then
  echo "Environment file $ENV does not exist."
  return 1
fi

ENV_NAME=$(grep -E '^name:' "$ENV" | awk '{print $2}')
if [ -z "$ENV_NAME" ]; then
  echo "Environment name not found in $ENV."
  return 1
fi

echo "Environment name: $ENV_NAME"

# Download and install Miniconda
if [ ! -d "$INSTALL_DIR" ]; then
  wget -q "$URL" -O "$INSTALLER"
  bash "$INSTALLER" -b -p "$INSTALL_DIR"
  if [ $? -ne 0 ]; then
      echo "Miniconda installation failed."
      rm -f "$INSTALLER"
      return 1
  fi
else
  echo "Miniconda already installed in $INSTALL_DIR"
fi

export PATH="$INSTALL_DIR/bin:$PATH"
eval "$(conda shell.bash hook)"
if [ $? -ne 0 ]; then
    echo "Error: Conda initialization failed."
    return 1
fi

# Create environment
if ! conda env list | awk '{print $1}' | grep -q "^$ENV_NAME$"; then
  echo "Creating conda environment '$ENV_NAME'..."
  conda env create -f "$ENV"
else
  echo "Conda environment '$ENV_NAME' already exists."
fi

echo "Activating base environment..."
conda activate base

# Cleanup
if [ -f "$INSTALLER" ]; then
  echo "Cleaning up installer..."
  rm -f "$INSTALLER"
fi

echo "Setup complete."
