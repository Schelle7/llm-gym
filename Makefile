CONDA_ENV := llm-gym
PYTHON_VERSION := 3.12

.PHONY: install dev dev-backend dev-frontend

install:
	@conda run -n $(CONDA_ENV) python --version >/dev/null 2>&1 || conda create -n $(CONDA_ENV) python=$(PYTHON_VERSION) -y
	conda run -n $(CONDA_ENV) python -m pip install --editable .

dev:
	$(MAKE) --jobs=2 dev-backend dev-frontend

dev-backend:
	python -m uvicorn llm_gym.server.app:app --reload \
		--reload-dir llm_gym

dev-frontend:
	npm --prefix frontend run dev
