#!/bin/bash

# start ssh server
if [ -n "$PUBLIC_KEY" ]; then
    mkdir -p ~/.ssh
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
    chmod 700 -R ~/.ssh

    dpkg-reconfigure openssh-server  # generate ssh keys
    service ssh start
fi

# start cloudflare tunnel
if [ -n "$CLOUDFLARED_TUNNEL_ARGS" ]; then
    /cloudflared $CLOUDFLARED_TUNNEL_ARGS &
fi

# start openchat server
python3 -m ochat.serving.openai_api_server --model $MODEL --host 127.0.0.1 --port 18888 --engine-use-ray --worker-use-ray --disable-log-requests --disable-log-stats $ARGS &

wait
