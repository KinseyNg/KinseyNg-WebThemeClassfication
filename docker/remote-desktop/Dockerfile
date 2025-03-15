FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install necessary dependencies including supervisor
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
    && rm -rf /var/lib/apt/lists/*

# Create user and set up home directory
RUN groupadd -g 1000 user && \
    useradd -m -u 1000 -g 1000 -s /bin/bash user && \
    echo "user:Cwit2016" | chpasswd && \
    adduser user sudo && \
    echo "user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Configure supervisord
COPY docker/remote-desktop/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Install Windsurf
RUN curl -s https://raw.githubusercontent.com/PumpkinSeed/windsurf-installer/refs/heads/main/install.sh | sh

# Set up work directory and permissions
RUN mkdir -p /home/user/work && \
    chown -R user:user /home/user

# Switch to the user
USER user

# Set the default working directory
WORKDIR /home/user/work

# Start supervisord
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
