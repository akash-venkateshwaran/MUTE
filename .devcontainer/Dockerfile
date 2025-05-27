FROM althack/ros2:humble-dev

ENV DISPLAY=:0


ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*
RUN git clone https://github.com/ompl/ompl.git && cd ompl && git reset --hard 2db81e2


# Based on https://github.com/ompl/ompl/blob/2db81e2154cad93f4b823b2afcd3e7c021ea2b2f/scripts/docker/ompl.Dockerfile
# avoid interactive configuration dialog from tzdata, which gets pulled in
# as a dependency
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        castxml \
        cmake \
        libboost-filesystem-dev \
        libboost-numpy-dev \
        libboost-program-options-dev \
        libboost-python-dev \
        libboost-serialization-dev \
        libboost-system-dev \
        libboost-test-dev \
        libeigen3-dev \
        libexpat1 \
        libflann-dev \
        libtriangle-dev \
        ninja-build \
        pkg-config \
        python3-dev \
        python3-numpy \
        python3-pip \
        pypy3 \
        wget \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*
RUN pip3 install pygccxml pyplusplus



# Install ROS Humble Gazebo packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends ros-humble-gazebo-* \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install rqt_graph
RUN apt-get update \
    && apt-get install -y --no-install-recommends  ros-humble-rqt-graph \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    


# ENV DEBIAN_FRONTEND=noninteractive
# RUN dpkg --add-architecture i386 \
#     && apt-get update \
#     && apt-get install -y --no-install-recommends \
#         gnupg2 \
#         software-properties-common \
#         wget \
#     && wget -nc https://dl.winehq.org/wine-builds/winehq.key \
#     && apt-key add winehq.key \
#     && apt-add-repository https://dl.winehq.org/wine-builds/ubuntu/ \
#     && apt-get update \
#     && apt-get install -y --no-install-recommends \
#         wine64 \
#         wine32 \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*
# ENV DEBIAN_FRONTEND=dialog


#Install python3 pkgs
RUN pip3 install pyproj
RUN pip3 install plotly
RUN pip3 install netCDF4
RUN pip3 install scipy
RUN pip3 install matplotlib
RUN pip3 install geopandas
RUN pip3 install -U pymoo
RUN pip3 install -U scikit-learn
RUN pip3 install scikit-datasets
RUN pip3 install PyYAML
RUN pip3 install h5py
RUN pip3 install cartopy
RUN pip3 install numpy-stl
RUN pip3 install git-lfs
# PBML:
RUN pip install mesh-to-sdf
RUN pip install pyglet==v1.5.28

RUN apt update && apt install -y msttcorefonts

WORKDIR /ompl
RUN cmake \
        -G Ninja \
        -B build \
        -DPYTHON_EXEC=/usr/bin/python3 \
        -DOMPL_REGISTRATION=OFF \
        -DCMAKE_INSTALL_PREFIX=/usr \
    && cmake --build build -t update_bindings \
    && NPROC=$(nproc) \
    && HALF_NPROC=$((NPROC / 2)) \
    && cmake --build build -- -j $HALF_NPROC \
    && cmake --install build \
    && cd tests/cmake_export \
    && cmake -B build -DCMAKE_INSTALL_PREFIX=../../install \
    && cmake --build build



# Install RViz2
RUN apt-get update \
    && apt-get install -y --no-install-recommends  ros-humble-rviz2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    

# Install PlotJuggler and the ROS 2 plugin
RUN apt-get update && apt-get install -y \
    ros-humble-plotjuggler-ros \
    && rm -rf /var/lib/apt/lists/*
    
# ** [Optional] Uncomment this section to install additional packages. **
#
# ENV DEBIAN_FRONTEND=noninteractive
# RUN apt-get update \
#    && apt-get -y install --no-install-recommends <your-package-list-here> \
#    #
#    # Clean up
#    && apt-get autoremove -y \
#    && apt-get clean -y \
#    && rm -rf /var/lib/apt/lists/*
# ENV DEBIAN_FRONTEND=dialog

# Set up auto-source of workspace for ros user
ARG WORKSPACE
RUN echo "if [ -f ${WORKSPACE}/install/setup.bash ]; then source ${WORKSPACE}/install/setup.bash; fi" >> /home/ros/.bashrc

# Additional instructions:
# Edit your bashrc >>vi ~/.bashrc
# Add three lines in the end and save:
# export DISPLAY=:0
# source /usr/share/gazebo/setup.sh

# You need to include the folders inside a pkg in the setup.py to include them in install/share/pkg