# List of markdown files
MARKDOWNS := quantizing_models.md measuring_inference_times.md

# Convert markdown filenames to notebook filenames
NOTEBOOKS := $(MARKDOWNS:.md=.ipynb)

# Default target to build all notebooks
all: $(NOTEBOOKS)

# Clean target to remove generated notebooks
clean:
	del quantizing_models.ipynb measuring_inference_times.ipynb

# Pattern rule to convert markdown to notebook
%.ipynb: %.md
	pandoc --wrap=none $< -o $@