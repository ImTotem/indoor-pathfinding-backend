# RTABMap with CUDA + CudaSift + g2o
# RTX 3060 (Ampere, compute capability 8.6)

FROM nvidia/cuda:12.0.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 1. Build dependencies
RUN apt-get update && apt-get install -y \
    cmake build-essential git wget pkg-config \
    # OpenCV
    libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libatlas-base-dev gfortran \
    # RTABMap
    libeigen3-dev libsqlite3-dev libpcl-dev \
    libboost-all-dev libflann-dev libproj-dev \
    libsuitesparse-dev libgtest-dev \
    libfreenect-dev libdc1394-dev \
    libyaml-cpp-dev liboctomap-dev \
    libqhull-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. OpenCV 4.8 with CUDA
WORKDIR /tmp
RUN git clone --depth 1 --branch 4.8.0 https://github.com/opencv/opencv.git && \
    git clone --depth 1 --branch 4.8.0 https://github.com/opencv/opencv_contrib.git && \
    cd opencv && mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules \
          -D WITH_CUDA=ON \
          -D CUDA_ARCH_BIN="8.6" \
          -D CUDA_ARCH_PTR="" \
          -D WITH_CUDNN=OFF \
          -D OPENCV_DNN_CUDA=OFF \
          -D ENABLE_FAST_MATH=ON \
          -D CUDA_FAST_MATH=ON \
          -D WITH_CUBLAS=ON \
          -D WITH_GTK=ON \
          -D BUILD_EXAMPLES=OFF \
          -D BUILD_DOCS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D BUILD_TESTS=OFF \
          -D OPENCV_GENERATE_PKGCONFIG=ON \
          .. && \
    make -j$(nproc) && \
    make install && ldconfig && \
    cd /tmp && rm -rf opencv opencv_contrib

# 3. g2o (graph optimizer)
RUN git clone --depth 1 https://github.com/RainerKuemmerle/g2o.git /tmp/g2o && \
    cd /tmp/g2o && mkdir -p build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D BUILD_WITH_MARCH_NATIVE=OFF \
          -D G2O_BUILD_EXAMPLES=OFF \
          -D G2O_BUILD_APPS=OFF \
          .. && \
    make -j$(nproc) && \
    make install && ldconfig && \
    rm -rf /tmp/g2o

# 4. CudaSift (GPU-accelerated SIFT)
RUN git clone --depth 1 https://github.com/Celebrandil/CudaSift.git /tmp/CudaSift && \
    cd /tmp/CudaSift && mkdir -p build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D CUDA_ARCH="8.6" \
          .. && \
    make -j$(nproc) && \
    make install && ldconfig && \
    rm -rf /tmp/CudaSift

# 5. RTABMap from source (with CUDA, CudaSift, g2o)
RUN git clone --depth 1 https://github.com/introlab/rtabmap.git /tmp/rtabmap && \
    cd /tmp/rtabmap && mkdir -p build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          .. && \
    make -j$(nproc) && \
    make install && ldconfig && \
    rm -rf /tmp/rtabmap

# --- Runtime stage ---
FROM nvidia/cuda:12.0.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgtk-3-0 libavcodec-dev libavformat-dev libswscale-dev \
    libjpeg8 libpng16-16 libtiff5 \
    libsqlite3-0 libpcl-dev \
    libboost-filesystem1.74.0 libboost-serialization1.74.0 \
    libboost-system1.74.0 libboost-thread1.74.0 \
    libboost-program-options1.74.0 libboost-regex1.74.0 \
    libflann1.9 libproj22 \
    libsuitesparseconfig5 libcholmod3 libcxsparse3 \
    libyaml-cpp0.7 liboctomap1.9 \
    libfreenect0.5 libdc1394-25 \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local /usr/local
RUN ldconfig

# Verify
RUN rtabmap-reprocess --version && \
    which rtabmap-console && \
    which rtabmap-export

RUN mkdir -p /data
WORKDIR /data
CMD ["tail", "-f", "/dev/null"]
