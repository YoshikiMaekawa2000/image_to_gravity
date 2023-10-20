FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
########## basis ##########
RUN apt update
RUN apt install -y --no-install-recommends \
    cmake \
    git \
    vim \
    tmux \
    build-essential \
    curl \
    wget \
    tzdata \
    tree \
    clangd \
    nodejs \
    npm
######### Python ##########
RUN apt-get update &&\
	apt-get install -y \
		python3 \
        python3-dev \
		python3-pip &&\
	pip3 install \
		tqdm \
		matplotlib \
		tensorflow \
		tensorboardX \
		torch \
		torchvision
######### neovim install #########
RUN cd /root/ && wget https://github.com/neovim/neovim/releases/download/stable/nvim-linux64.tar.gz
RUN cd /root/ && tar xzvf nvim-linux64.tar.gz
RUN rm /root/nvim-linux64.tar.gz
RUN mv /root/nvim-linux64 /
RUN ln -sf /nvim-linux64/bin/nvim /usr/bin/nvim
######### vim/neovim setup #########
RUN pip3 install jedi-language-server
RUN npm install -g n
RUN n stable
RUN mkdir -p /root/.vim/undo
RUN mkdir -p /root/.vim/plugged
RUN mkdir -p /root/.config/nvim
######### dotfiles #########
RUN git clone https://github.com/YoshikiMaekawa2000/dotfiles_for_docker.git /root/dotfiles
ENV DOTFILES="/root/dotfiles"
RUN ln -sf $DOTFILES/profile.sh /etc/profile.d/profile.sh
RUN ln -sf $DOTFILES/vim/vimrc /root/.vimrc
RUN ln -sf $DOTFILES/vim/init.vim /root/.config/nvim/init.vim
RUN ln -sf $DOTFILES/vim/coc-settings.json /root/.config/nvim/coc-settings.json
RUN ln -sf $DOTFILES/tmux.conf /root/.tmux.conf
######### NO cache ##########
ARG CACHEBUST=1
######### My package ##########
RUN cd /home &&\
	git clone https://github.com/YoshikiMaekawa2000/image_to_gravity.git
######### initial position ##########
WORKDIR /home/image_to_gravity/pysrc
