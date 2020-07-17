main: mst.cpp
	g++ mst.cpp --std=c++11 -fopenmp -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_video -lopencv_videoio -lopencv_highgui -o main -g
