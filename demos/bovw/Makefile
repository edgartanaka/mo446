# COMMANDS
MKDIR=mkdir -p
CP=cp
PYTHON=python
RM=rm -r -f

# FOLDERS
SRC_FOLDER=.
BIN_FOLDER=$(BUILD_FOLDER)


# SRCs and OBJs
SRC = $(wildcard $(SRC_FOLDER)/*.py)
MAIN=$(BIN_FOLDER)/main.py

before_build:
	pip install -r requirements.txt
	pip3 install -r requirements.txt

# BUILD COMMAND
build:
	$(CP) $(SRC) $(BIN_FOLDER)

# RUN COMMAND
test:
	pip install -r requirements.txt
	pip3 install -r requirements.txt
	$(PYTHON) $(MAIN) -a $(ASSETS_FOLDER) -o $(OUTPUT_FOLDER)

# CLEAN COMMAND
clean:
	$(RM) $(BIN_FOLDER)
	$(RM) $(OUTPUT_FOLDER)
