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
RUN groupadd -g 1000 user && \
    useradd -m -u 1000 -g 1000 -s /bin/bash user && \
    echo "user:Cwit2016" | chpasswd && \
    adduser user sudo && \
    echo "user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Configure supervisord
COPY docker/remote-desktop/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Set up directories and permissions
RUN mkdir -p /home/user/work /home/user/.local/share/applications /home/user/.local/bin && \
    chown -R user:user /home/user

# Switch to the user
USER user

# Set up environment variables
ENV PATH="/home/user/.local/bin:${PATH}"
ENV XDG_DATA_HOME="/home/user/.local/share"
ENV XDG_CONFIG_HOME="/home/user/.config"

# Install Windsurf for the user
RUN mkdir -p "${XDG_DATA_HOME}/applications" "${XDG_CONFIG_HOME}" && \
    curl -s https://raw.githubusercontent.com/PumpkinSeed/windsurf-installer/refs/heads/main/install.sh | bash

# Set the default working directory
WORKDIR /home/user/work

# Start supervisord
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
