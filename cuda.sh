
#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to check if a command exists
command_exists() {
    command -v "$@" > /dev/null 2>&1
}

# Ensure the script is run as root
if [ "$(id -u)" -ne 0 ]; then
    echo "This script must be run as root."
    exit 1
fi

# Step 1: Ensure NVIDIA driver is already installed
if ! command_exists nvidia-smi; then
    echo "Error: NVIDIA driver is not installed. Please install the driver version 535.230.02 first."
    exit 1
else
    echo "NVIDIA driver detected:"
    nvidia-smi
fi

# Step 2: Install Docker if not already installed
if ! command_exists docker; then
    echo "Installing Docker..."
    apt-get update
    apt-get install -y \
        ca-certificates \
        curl \
        gnupg \
        lsb-release

    mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg

    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
else
    echo "Docker is already installed."
    docker --version
fi

# Step 3: Install NVIDIA Container Toolkit
if ! command_exists nvidia-container-toolkit; then
    echo "Installing NVIDIA Container Toolkit..."
    distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/ubuntu$(lsb_release -r -s)/$(dpkg --print-architecture)/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    apt-get update
    apt-get install -y nvidia-container-toolkit
else
    echo "NVIDIA Container Toolkit is already installed."
fi

# Step 4: Configure Docker to use the NVIDIA runtime
echo "Configuring Docker to use NVIDIA runtime..."
nvidia-ctk runtime configure --runtime=docker

# Step 5: Restart Docker to apply changes
echo "Restarting Docker..."
systemctl restart docker

# Final Step: Verify Installation
echo "Verifying GPU support in Docker..."
docker run --rm --gpus all nvidia/cuda:12.2-base nvidia-smi

echo "All done! Docker is now configured to use your GPU."
