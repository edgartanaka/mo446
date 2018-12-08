# Script variables
PYTHON          = python
SRC_DIR         = src
INPUT_DIR       = input
OUTPUT_DIR      = output

# Preparation stage
prepare:
	mkdir -p $(INPUT_DIR)
	wget -q -P input/ https://storage.googleapis.com/mo446/hotel.seq.zip
	#wget -q -P input/ https://storage.googleapis.com/mo446/dinoRing.zip
	#wget -q -P input/ https://storage.googleapis.com/mo446/temple.zip
	#wget -q -P input/ https://storage.googleapis.com/mo446/slow.flv
	unzip -d $(INPUT_DIR) $(INPUT_DIR)/hotel.seq.zip -x "__MACOSX/*"

# Execution stage
run: keypoints klt sfm

klt:
	mkdir -p $(OUTPUT_DIR)/klt
	$(PYTHON) $(SRC_DIR)/klt.py

keypoints:
	mkdir -p $(OUTPUT_DIR)/keypoints
	$(PYTHON) $(SRC_DIR)/keypoints.py

sfm:
	mkdir -p $(OUTPUT_DIR)/sfm
	$(PYTHON) $(SRC_DIR)/sfm.py


all: clean prepare run

# Cleaning output directory
clean:
	rm -rf $(OUTPUT_DIR)
	rm -rf $(INPUT_DIR)

.PHONY: prepare run
