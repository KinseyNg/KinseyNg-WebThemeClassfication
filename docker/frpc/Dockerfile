FROM snowdreamtech/frpc:0.52.3

USER root

# Install OpenConnect and supervisor
RUN apk add --no-cache openconnect supervisor bash

# Copy VPN credentials and systemd service file
COPY docker/frpc/vpn_credentials /etc/openconnect/vpn_credentials
COPY docker/frpc/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
