CXX=g++
SOURCES=xim.c
OBJECTS=$(SOURCES:.c=.o)
PROGRAM=xim

all: $(PROGRAM)

$(PROGRAM): $(OBJECTS)
	$(CXX) -g $(OBJECTS) -o $@ -lm3api -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_features2d -L/usr/local/lib

.c.o: $(patsubst %.c,%.o,$(wildcard *.c))
	$(CXX) -g -Wall -c $< -I /usr/local/include/ -o $@

clean:
	rm -f $(PROGRAM) $(OBJECTS)

