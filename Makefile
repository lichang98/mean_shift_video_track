main: mean_shift_track.cpp
	g++ mean_shift_track.cpp --std=c++11 -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_video -lopencv_videoio -lopencv_highgui -o main -g

.PHONY clean:
	rm main