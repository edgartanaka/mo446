# Script variables
PYTHON          = python
SRC_DIR         = src
INPUT_DIR       = input
OUTPUT_DIR      = output

# Preparation stage
prepare:
	mkdir -p $(INPUT_DIR)
	mkdir -p $(OUTPUT_DIR)
	wget -q -P input/ https://storage.googleapis.com/mo446/p4/480p/easy.mp4
	wget -q -P input/ https://storage.googleapis.com/mo446/p4/480p/medium.mp4
	wget -q -P input/ https://storage.googleapis.com/mo446/p4/480p/hard.mp4


# Execution stage
run:
	pip install -r requirements.txt
	bash install_chrome.sh
	$(PYTHON) $(SRC_DIR)/detection.py
	$(PYTHON) $(SRC_DIR)/final_video.py


all: clean prepare run

# Cleaning output directory
clean:
	rm -rf $(OUTPUT_DIR)
	rm -rf $(INPUT_DIR)

.PHONY: prepare run
