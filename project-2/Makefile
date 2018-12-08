# Script variables
PYTHON          = python
SRC_DIR         = src
INPUT_DIR       = input
OUTPUT_DIR      = output

# Preparation stage
prepare:
	mkdir -p $(INPUT_DIR)
	wget -q -O input/scale.mp4 https://storage.googleapis.com/mo446/scale.mp4
	wget -q -O input/rotate.mp4  https://storage.googleapis.com/mo446/rotate.mp4
	wget -q -O input/translate.mp4  https://storage.googleapis.com/mo446/translate.mp4
	wget -q -O input/target.jpg https://storage.googleapis.com/mo446/target.jpg

# Execution stage
frame:
	mkdir -p $(INPUT_DIR)/frames/scale
	mkdir -p $(INPUT_DIR)/frames/rotate
	mkdir -p $(INPUT_DIR)/frames/translate
	$(PYTHON) $(SRC_DIR)/frame.py

run:
	mkdir -p $(OUTPUT_DIR)/frames/scale
	mkdir -p $(OUTPUT_DIR)/frames/rotate
	mkdir -p $(OUTPUT_DIR)/frames/translate
	$(PYTHON) $(SRC_DIR)/main_opencv.py
	rm $(OUTPUT_DIR)/frames/scale/*
	rm $(OUTPUT_DIR)/frames/rotate/*
	rm $(OUTPUT_DIR)/frames/translate/*
	$(PYTHON) $(SRC_DIR)/main.py

sift-demo:
	mkdir -p $(OUTPUT_DIR)/sift-demo
	$(PYTHON) $(SRC_DIR)/sift-demo.py

all: clean prepare frame run sift-demo

# Cleaning output directory
clean:
	rm -rf $(OUTPUT_DIR)
	rm -rf $(INPUT_DIR)

.PHONY: prepare run
