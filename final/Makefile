# Script variables
PYTHON          = python
MAKE            = make
SRC_DIR         = src
INPUT_DIR       = input
OUTPUT_DIR      = output

# Gets the input data
$(INPUT_DIR)/Data:
	mkdir -p $(INPUT_DIR)
	mkdir -p $(OUTPUT_DIR)
	wget -q -P $(INPUT_DIR)/ https://storage.googleapis.com/mo446/final/Data.zip
	unzip -d $(INPUT_DIR) -o $(INPUT_DIR)/Data.zip -x "__MACOSX/*"
	pip install --upgrade pip
	pip install -r requirements.txt

# Execution stage
run: $(INPUT_DIR)/Data
	$(PYTHON) $(SRC_DIR)/predictions.py

# Execution with features
features: $(OUTPUT_DIR)/target_features.mat
	$(PYTHON) $(SRC_DIR)/global_features_utilisation.py

# Computation of global features
$(OUTPUT_DIR)/target_features.mat: lear-gist-python $(INPUT_DIR)/Data
	$(PYTHON) $(SRC_DIR)/global_features_computation.py

lear-gist-python: fftw-3.3.8
	git clone https://github.com/tuttieee/lear-gist-python \
    && cd lear-gist-python && ./download-lear.sh \
    && $(PYTHON) setup.py build_ext && $(PYTHON) setup.py install && cd ..

fftw-3.3.8:
	wget http://www.fftw.org/fftw-3.3.8.tar.gz
	tar xf fftw-3.3.8.tar.gz
	cd fftw-3.3.8 && ./configure --enable-single --enable-shared &&	$(MAKE) \
    &&	$(MAKE) install && ldconfig && cd ..

all: clean features

# Cleaning output directory
clean:
	rm -rf $(OUTPUT_DIR)
	rm -rf $(INPUT_DIR)

.PHONY: clean run features
