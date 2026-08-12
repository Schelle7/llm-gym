CONDA_ENV := llm-gym
PYTHON_VERSION := 3.12

.PHONY: install

install:
	@conda run -n $(CONDA_ENV) python --version >/dev/null 2>&1 || conda create -n $(CONDA_ENV) python=$(PYTHON_VERSION) -y
	conda run -n $(CONDA_ENV) python -m pip install --editable .