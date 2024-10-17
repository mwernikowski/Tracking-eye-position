sudo apt install -y build-essential cmake pkg-config unzip yasm git checkinstall
sudo apt install -y gcc-8 g++-8 gdb
sudo apt install -y python2
sudo apt install -y libjpeg-dev libpng-dev libtiff-dev
sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev libavresample-dev
sudo apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt install -y libxvidcore-dev x264 libx264-dev libfaac-dev libmp3lame-dev libtheora-dev 
sudo apt install -y libfaac-dev libmp3lame-dev libvorbis-dev
sudo apt install -y libopencore-amrnb-dev libopencore-amrwb-dev
sudo apt install -y libdc1394-22 libdc1394-22-dev libxine2-dev libv4l-dev v4l-utils
sudo apt install -y libomp-dev
sudo ln -s -f /usr/include/libv4l1-videodev.h /usr/include/linux/videodev.h
sudo apt-get install -y libgtk-3-dev
sudo apt-get install -y python3-dev python3-pip
sudo apt install -y python3-testresources
sudo apt-get install -y libtbb-dev
sudo apt-get install -y libatlas-base-dev gfortran
sudo apt-get install -y libprotobuf-dev protobuf-compiler
sudo apt-get install -y libgoogle-glog-dev libgflags-dev
sudo apt-get install -y libgphoto2-dev libeigen3-dev libhdf5-dev doxygen
wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/4.6.0.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/4.6.0.zip
unzip opencv.zip
unzip opencv_contrib.zip
sudo pip install virtualenv virtualenvwrapper
sudo rm -rf .cache/pip
export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
source /usr/local/bin/virtualenvwrapper.sh
mkvirtualenv cv -p python3
pip install numpy
cd opencv-4.6.0
mkdir build
cd build/
cmake -DCMAKE_INSTALL_PREFIX=/usr/local       -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.6.0/modules       -DBUILD_LIST=cudaarithm,cudafilters,cudaimgproc,cudev,imgcodecs,highgui,video,videoio,calib3d       -DENABLE_FAST_MATH=ON -DWITH_CUDA=ON -DCUDA_FAST_MATH=ON -DWITH_FFMPEG=ON -DWITH_LAPACK=ON       -DINSTALL_C_EXAMPLES=OFF -DINSTALL_PYTHON_EXAMPLES=OFF       -DCMAKE_C_COMPILER=/usr/bin/gcc-8 -DCMAKE_CXX_COMPILER=/usr/bin/g++-8 ..
make -j6
sudo make -j6 install
rm -rf opencv.zip opencv_contrib.zip opencv-4.6.0 opencv_contrib-4.6.0
echo 'export LD_LIBRARY_PATH=/usr/local/lib/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
export LD_LIBRARY_PATH=/usr/local/lib/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
