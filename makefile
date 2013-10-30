CV_HOME=/home/zhouping/tonyren/cv

all: lshFreak

lshFreak: lshFreak.o
	g++ -I$(CV_HOME)/include -L$(CV_HOME)/include/lib -o lshFreak lshFreak.o -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_flann -lopencv_nonfree

lshFreak.o: lshFreak.cpp
	g++ -I$(CV_HOME)/include -c lshFreak.cpp
	
clean:
	rm *.o
	rm lshFreak
