# fpath: ./Dockerfile

FROM python:3.11-slim

# Install system packages and basic tools
RUN apt-get update && apt-get install -y \
  build-essential \
  libgl1-mesa-glx \
  libglib2.0-0 \
  libpng-dev \
  libjpeg-dev \
  curl \
  git \
  tzdata \ 
  nano \
  zsh \
  net-tools \
  openssh-server \
  && rm -rf /var/lib/apt/lists/* \
  && sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended \
  && git clone https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k \
  && git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions \
  && git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# Install uv and create virtual environment
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
  . $HOME/.profile && \
  export PATH="$HOME/.local/bin:$PATH" && \
  uv venv /app/.venv && \
  . /app/.venv/bin/activate && \
  uv pip install --upgrade pip

# Copy zsh configuration files
COPY ./zsh/zshrc /root/.zshrc
COPY ./zsh/p10k.zsh /root/.p10k.zsh
COPY ./.project_root ./app/.project_root

# Set working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/data /app/runs /app/models /app/src /app/figures \
  && chmod 777 /app/data /app/runs /app/models /app/src /app/figures

# Copy and install requirements
COPY requirements.txt .
RUN . /app/.venv/bin/activate && \
  export PATH="$HOME/.local/bin:$PATH" && \
  uv pip install -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Activate virtual environment by default
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV TZ=Asia/Seoul

# Set Timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# SSH 설정 추가 (기존 내용 아래에)
RUN mkdir /var/run/sshd
RUN echo 'root:yourpassword' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
# 추가
RUN sed -i 's/PasswordAuthentication no/PasswordAuthentication yes/' /etc/ssh/sshd_config
RUN sed -i 's/#Port 22/Port 22/' /etc/ssh/sshd_config
RUN echo "export VISIBLE=now" >> /etc/profile


# Port 설정 추가
EXPOSE 22

CMD [ "/usr/sbin/sshd", "-D" ]

# Set zsh as default shell
SHELL ["/bin/zsh", "-c"]