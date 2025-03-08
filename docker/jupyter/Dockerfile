FROM quay.io/jupyter/minimal-notebook:latest

COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN pip install torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/cu118 
RUN pip install anytree matplotlib seaborn
RUN pip install ipywidgets
RUN pip install nltk 

# Install SSH server
USER root
RUN apt-get update && \
    apt-get install -y openssh-server rsyslog && \
    mkdir /var/run/sshd

# Set password for the default `jovyan` user
#RUN echo "jovyan:jupyter" | chpasswd

# Allow password authentication and configure SSH
# Configure SSH to allow remote login
# Configure SSH for auditing
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    echo "AllowUsers root jovyan" >> /etc/ssh/sshd_config && \
    echo "LogLevel VERBOSE" >> /etc/ssh/sshd_config

# Redirect SSH logs directly to stdout using tail
RUN touch /var/log/auth.log

# Configure rsyslog to forward logs to Docker stdout
#RUN echo '*.* /dev/stdout' >> /etc/rsyslog.conf && \
 #   echo 'auth,authpriv.* /dev/stdout' >> /etc/rsyslog.conf
# Enable rsyslog for SSH audit logging
RUN sed -i 's/#module(load="imklog")/module(load="imklog")/' /etc/rsyslog.conf && \
    sed -i 's/#module(load="imuxsock")/module(load="imuxsock")/' /etc/rsyslog.conf

# Enable SSH and logging services to start on boot
RUN systemctl enable ssh && \
    systemctl enable rsyslog

# Expose ports for Jupyter Notebook and SSH
EXPOSE 8888 22


# Set password for the default `jovyan` user
RUN echo "jovyan:jupyter" | chpasswd

# Set password for the root user
RUN echo "root:kinseyng0826" | chpasswd


# Install and configure the FTP server (vsftpd)
RUN apt-get install -y vsftpd && \
    mkdir -p /var/ftp && \
    mkdir -p /root/jupyter-docker-compose && \
    chmod -R 755 /root/jupyter-docker-compose

# Configure vsftpd
RUN echo "listen=YES" >> /etc/vsftpd.conf && \
    echo "anonymous_enable=NO" >> /etc/vsftpd.conf && \
    echo "local_enable=YES" >> /etc/vsftpd.conf && \
    echo "write_enable=YES" >> /etc/vsftpd.conf && \
    echo "local_umask=022" >> /etc/vsftpd.conf && \
    echo "chroot_local_user=YES" >> /etc/vsftpd.conf && \
    echo "allow_writeable_chroot=YES" >> /etc/vsftpd.conf && \
    echo "user_sub_token=\$USER" >> /etc/vsftpd.conf && \
    echo "local_root=/root/jupyter-docker-compose" >> /etc/vsftpd.conf && \
    echo "pasv_enable=YES" >> /etc/vsftpd.conf && \
    echo "pasv_min_port=40000" >> /etc/vsftpd.conf && \
    echo "pasv_max_port=40010" >> /etc/vsftpd.conf

# Create an FTP user and set the password
RUN useradd -m -d /root/jupyter-docker-compose ftpuser && \
    echo "ftpuser:kinseyng0826" | chpasswd

# Expose FTP port (default 21) and passive data ports
EXPOSE 21 40000-40010

# Enable vsftpd to start on boot
RUN systemctl enable vsftpd


# Start both SSH and Jupyter Notebook servers on container boot
CMD service rsyslog start && service ssh start &&  service vsftpd start && tail -f /var/log/auth.log
