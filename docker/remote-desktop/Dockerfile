FROM scottyhardy/docker-remote-desktop:latest

USER root

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create user and set up home directory
RUN useradd -m -d /home/user -s /bin/bash user && \
    echo "user:Cwit2016" | chpasswd && \
    adduser user sudo

# Install Windsurf (replace with actual installation commands when available)
RUN curl -s https://raw.githubusercontent.com/PumpkinSeed/windsurf-installer/refs/heads/main/install.sh | sh

# Set proper ownership
RUN mkdir -p /home/user/work && \
    chown -R user:user /home/user

# Switch to the user
USER user

# Set the default working directory
WORKDIR /home/user/work

# Keep the container running
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/supervisord.conf"]
