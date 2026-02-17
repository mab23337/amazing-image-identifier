# Docker (TR-33)

Docker is not installed on the Raspberry Pi used for development (docker command not found).

However, the repository includes a Dockerfile for consistent deployment. To build on any machine with Docker installed:

docker build -t amazing-image-identifier:latest .

If Docker is installed on the Pi later:
sudo apt install docker.io -y
sudo systemctl enable --now docker
