CXX = g++
INCLUDE_DIRS=/usr/local/tara-opencv/include
LIBRARY_DIRS=/usr/local/tara-opencv/lib
#INCLUDE_DIRS=/usr/local/tara-opencv-4.0/include/opencv4/
#LIBRARY_DIRS=/usr/local/tara-opencv-4.0/lib/
.PHONY:
depth: stereo_depth_v2.o
	$(CXX) -march=armv8.1a+simd -I$(INCLUDE_DIRS) -L$(LIBRARY_DIRS) -g -o stereo_depth stereo_depth_v2.o -lrt -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_imgcodecs -lopencv_ximgproc -lopencv_flann 
	#$(CXX) -march=armv8.1a+simd $(pkg-config --cflags --libs /usr/local/tara-opencv-4.0/lib/pkgconfig/opencv.pc) -g -o stereo_depth stereo_depth_v2.o -lrt
