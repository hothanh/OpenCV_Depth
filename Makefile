CXX = g++
.PHONY:
depth: stereo_depth_v2.o
	$(CXX) -I/usr/local/include/opencv -I/usr/local/include/opencv2 -L/usr/local/lib/ -g -o stereo_depth stereo_depth_v2.o -lrt -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_imgcodecs -lopencv_ximgproc 
