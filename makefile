# Makefile for building and running a simple neural network library in C++
# Compiler: g++

CXX = g++
CXXFLAGS = -std=c++17 -O2
# The name of the executable

TARGET = nn_library
# The source file

SOURCE = nn_library.cpp
SOURCES = $(SOURCE)
# Default target: all

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)
# Run the executable

run: all
	./$(TARGET)
# Clean up build artifacts

clean:
	rm -f $(TARGET)
	rm -f *.o

.PHONY: all run clean