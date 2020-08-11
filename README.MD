# Installation
## On *NIX Systems
First check if Python is installed
`python3 -v`

If it isn't:
```bash
sudo add-apt-repository ppa:jonathonf/python-3.6
sudo apt-get update
sudo apt-get install python3.6
```

Next, install OpenCV
```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install build-essential cmake
sudo apt-get install python-devel numpy
sudo apt-get install gcc gcc-c++
sudo apt-get install libpng-devel
sudo apt-get install libjpeg-turbo-devel
sudo apt-get install jasper-devel
sudo apt-get install openexr-devel
sudo apt-get install libtiff-devel
sudo apt-get install libwebp-devel
sudo apt-get install git
git clone https://github.com/opencv/opencv.git
mkdir build
cd build
cmake ../
```

You should see the following output:
Python 2:
```bash
--     Interpreter:                 /usr/bin/python2.7 (ver 2.7.6)
--     Libraries:                   /usr/lib/x86_64-linux-gnu/libpython2.7.so (ver 2.7.6)
--     numpy:                       /usr/lib/python2.7/dist-packages/numpy/core/include (ver 1.8.2)
--     packages path:               lib/python2.7/dist-packages
```
Python 3:
```bash
--     Interpreter:                 /usr/bin/python3.4 (ver 3.4.3)
--     Libraries:                   /usr/lib/x86_64-linux-gnu/libpython3.4m.o (ver 3.4.3)
--     numpy:                       /usr/lib/python3/dist-packages/numpy/core/nclude (ver 1.8.2)
--     packages path:               lib/python3.4/dist-packages
```
Then, start the build with the following commands:
```bash
make
sudo make install
```
This will take a while (usually ~30 minutes)

Once it is successful, run the following python script to verify the installation in python:
```python
import cv2 as cv
print(cv.__version__)
```

Next, install dlib:
```bash
sudo apt-get install libgtk-3-dev
sudo apt-get install libboost-all-dev
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
sudo pip3 install numpy
sudo pip3 install scipy
sudo pip3 install scikit-image
sudo pip3 install dlib
```

Next install openface:
```bash
git clone https://github.com/TadasBaltrusaitis/OpenFace.git
cd OpenFace 
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE CMAKE_CXX_FLAGS="-std=c++11" -D MAKE_EXE_LINKER_FLAGS="-std=c++11" ..
make
```

Finally, install pandas and Flask
```bash
sudo pip3 install Flask
sudo pip3 install pandas
```

Clone Recogniser
```bash
cd ~
git clone https://github.com/saurabhschheda/Recogniser.git
cd Recogniser
```

To run:
`python3 app.py --host=0.0.0.0`

On a browser, navigate to the public IP address of the host on port 5000.
For example, if the host is on `192.168.0.2`, navigate to `192.168.0.2:5000/upload`