# Use Ubuntu 22.04 (will be supported until April 2027)
FROM ubuntu:jammy

# Add openMVG binaries to path
ENV PATH $PATH:/opt/openMVG_Build/install/bin

# Get dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata; \
  apt-get install -y \
  cmake \
  build-essential \
  graphviz \
  git \
  coinor-libclp-dev \
  libceres-dev \
  libflann-dev \
  libjpeg-dev \
  liblemon-dev \
  libpng-dev \
  libtiff-dev \
  gdb \
  python3-minimal \
  python3-pip \
  python3-tk \
  qtcreator \
  qtbase5-dev \
  qt5-qmake \
  libqglviewer-dev-qt5 \
  libqt5svg5-dev \
  ffmpeg libsm6 libxext6 &&\
  apt-get autoclean && apt-get clean

    
RUN pip3 install \
    scipy \
    opencv-python \
    matplotlib

# # Clone the openvMVG repo
# ADD . /opt/openMVG
# RUN cd /opt/openMVG && git submodule update --init --recursive

# # Build
# RUN mkdir /opt/openMVG_Build; \
#   cd /opt/openMVG_Build; \
#   cmake -DCMAKE_BUILD_TYPE=RELEASE \
#     -DCMAKE_INSTALL_PREFIX="/opt/openMVG_Build/install" \
#     -DOpenMVG_BUILD_TESTS=ON \
#     -DOpenMVG_BUILD_EXAMPLES=ON \
#     -DCOINUTILS_INCLUDE_DIR_HINTS=/usr/include \
#     -DLEMON_INCLUDE_DIR_HINTS=/usr/include/lemon \
#     -DCLP_INCLUDE_DIR_HINTS=/usr/include \
#     -DOSI_INCLUDE_DIR_HINTS=/usr/include \
#     ../openMVG/src; \
#     make -j 4;

# RUN cd /opt/openMVG_Build && make test && make install;

   