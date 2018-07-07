FROM ufoym/deepo

RUN apt-get -y update && apt-get -y install git wget python-dev python3-dev libopenmpi-dev python-pip zlib1g-dev cmake
WORKDIR /srv
# note: openai baseline depends on tensorflow. We force --no-deps to avoid this.
# note: which means that we have to uninstall tensorflow (cpu-only version)
ADD ./requirements.txt /srv/requirements.txt
RUN pip install -r requirements.txt
# remove the cpu-only version of tensorflow
RUN yes | pip uninstall tensorflow tensorflow-gpu
RUN yes | pip install tensorflow-gpu

ENV LANG C.UTF-8

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# add mujoco binary and license file

RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && rm mujoco.zip
COPY ./mjkey.txt /root/.mujoco/
ENV LD_LIBRARY_PATH /root/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
# we mount the project directory under /opt/project.
# This way the image does not have to
#WORKDIR /opt/project
#RUN pip install -e /opt/project

# ENTRYPOINT ["run_server"]
