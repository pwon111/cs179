CC = g++
FLAGS = -g -o

INCLUDE = -I/usr/X11R6/include -I/usr/include -I/home/nailen/Documents/cs179/project/
LIBDIR = -L/usr/X11R6/lib -L/usr/local/lib
SRC = raytrace.cpp
LIBS = -lm

EXENAME = rt

all: rt

rt: $(SRC)
	$(CC) $(FLAGS) $(EXENAME) $(INCLUDE) $(LIBDIR) $(SRC) $(LIBS)

clean:
	rm -f *.o $(EXENAME)

PHONY: all clean
