FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install necessary dependencies including supervisor and desktop environment
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    python3 \
    python3-pip \
    supervisor \
    xfce4 \
    xfce4-terminal \
    xrdp \
    desktop-file-utils \
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Create user and set up home directory
ARG REMOTE_DESKTOP_USER=user
ARG REMOTE_DESKTOP_PASSWORD

# Create user with sudo access
RUN useradd -m -s /bin/bash ${REMOTE_DESKTOP_USER} && \
    usermod -aG sudo ${REMOTE_DESKTOP_USER} && \
    echo "${REMOTE_DESKTOP_USER}:${REMOTE_DESKTOP_PASSWORD}" | chpasswd && \
    echo "${REMOTE_DESKTOP_USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set up supervisor directories
RUN mkdir -p /var/log/supervisor /var/run/supervisor /etc/supervisor/conf.d && \
    chown -R ${REMOTE_DESKTOP_USER}:${REMOTE_DESKTOP_USER} /var/log/supervisor /var/run/supervisor

# Configure supervisord
COPY docker/remote-desktop/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Set up directories and permissions
RUN mkdir -p /home/${REMOTE_DESKTOP_USER}/work /home/${REMOTE_DESKTOP_USER}/.local/share/applications /home/${REMOTE_DESKTOP_USER}/.local/bin && \
    chown -R ${REMOTE_DESKTOP_USER}:${REMOTE_DESKTOP_USER} /home/${REMOTE_DESKTOP_USER}

# Switch to the user
USER ${REMOTE_DESKTOP_USER}

# Set up environment variables
ENV PATH="/home/${REMOTE_DESKTOP_USER}/.local/bin:${PATH}"
ENV XDG_DATA_HOME="/home/${REMOTE_DESKTOP_USER}/.local/share"
ENV XDG_CONFIG_HOME="/home/${REMOTE_DESKTOP_USER}/.config"

# Install Windsurf for the user
RUN mkdir -p "${XDG_DATA_HOME}/applications" "${XDG_CONFIG_HOME}" && \
    curl -s https://raw.githubusercontent.com/PumpkinSeed/windsurf-installer/refs/heads/main/install.sh | bash

# Set the default working directory
WORKDIR /home/${REMOTE_DESKTOP_USER}/work

# Start supervisord
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
