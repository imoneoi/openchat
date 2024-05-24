FROM python:3.11-slim-bookworm

######### Setup system

RUN mkdir /workspace && mkdir /workspace/transformers_cache
WORKDIR /workspace

ENV HF_HOME /workspace/transformers_cache

######### Install system dependencies
RUN apt update && apt install -y git bash curl wget libxml2

# Install ssh server, remove all pre-generated ssh host keys, and disable password auth
RUN apt install -y openssh-server && \
    rm -f /etc/ssh/ssh_host_* && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/g' /etc/ssh/sshd_config

# Install CUDA (for FlashAttention 2)
RUN wget -q --show-progress --progress=bar:force:noscroll -O cuda_installer https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run && \
    chmod +x cuda_installer && \
    ./cuda_installer --silent --toolkit --override && \
    rm -f cuda_installer

######### Install OpenChat
# Install OpenChat
RUN pip3 install ninja packaging torch
RUN pip3 install ochat

######### Install Cloudflared
RUN wget -q --show-progress --progress=bar:force:noscroll -O /cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x /cloudflared

######### Startup script

COPY start.sh /start.sh
ENTRYPOINT ["/start.sh"]
