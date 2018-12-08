# Script variables
PYTHON          = python
SRC_DIR         = src
INPUT_DIR       = input
OUTPUT_DIR      = output
INPUT_DOG  = $(INPUT_DIR)/dog.jpg
INPUT_LAKE  = $(INPUT_DIR)/lake.jpg
INPUT_LENA  = $(INPUT_DIR)/lena.jpg
INPUT_PEPPERS  = $(INPUT_DIR)/peppers.png


# Preparation stage
all: $(INPUT_DOG) $(INPUT_LAKE) $(INPUT_LENA) $(INPUT_PEPPERS) $(OUTPUT_DIR)

$(INPUT_DIR):
	mkdir -p $(INPUT_DIR)

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)/convolution
	mkdir -p $(OUTPUT_DIR)/color
	mkdir -p $(OUTPUT_DIR)/edge
	mkdir -p $(OUTPUT_DIR)/cartoonize

$(INPUT_DOG): $(INPUT_DIR)
	wget -q -O $(INPUT_DOG) https://slackhideo.files.wordpress.com/2018/09/p1-dog.jpg && touch $(INPUT_DOG)

$(INPUT_LAKE): $(INPUT_DIR)
	wget -q -O $(INPUT_LAKE) https://slackhideo.files.wordpress.com/2018/09/p1-leaf-boat.jpg && touch $(INPUT_LAKE)

$(INPUT_LENA): $(INPUT_DIR)
	wget -q -O $(INPUT_LENA) https://slackhideo.files.wordpress.com/2018/09/p1-lena.jpg && touch $(INPUT_LENA)

$(INPUT_PEPPERS): $(INPUT_DIR)
	wget -q -O $(INPUT_PEPPERS) https://storage.googleapis.com/mo446/peppers.png && touch $(INPUT_PEPPERS)

# Execution stage
run: convolution color edge cartoonize

# Convolution test
convolution:
	$(PYTHON) $(SRC_DIR)/convolution.py

# Colour reduction test
color:
	$(PYTHON) $(SRC_DIR)/color.py

# Edge detection
edge:
	$(PYTHON) $(SRC_DIR)/edge.py

# Cartoonize all images in input directory
cartoonize:
	$(PYTHON) $(SRC_DIR)/cartoonize.py

# Cleaning output directory
clean:
	rm -rf $(OUTPUT_DIR)

.PHONY: all run convolution color edge clean
