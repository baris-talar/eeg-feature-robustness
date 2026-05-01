PYTHON ?= python
CACHE_ENV = XDG_CACHE_HOME=$(CURDIR)/.cache MPLCONFIGDIR=$(CURDIR)/.cache/matplotlib

.PHONY: install install-editable test compile reproduce-core figures check clean clean-intermediates

install:
	$(PYTHON) -m pip install -r requirements.txt

install-editable:
	$(PYTHON) -m pip install -e .

test:
	$(CACHE_ENV) PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

compile:
	$(PYTHON) -m compileall src

reproduce-core:
	$(CACHE_ENV) PYTHONPATH=src $(PYTHON) -m eeg_feature_robustness.pipeline

figures:
	$(CACHE_ENV) PYTHONPATH=src $(PYTHON) -m eeg_feature_robustness.pipeline --report-only

check: test compile

clean:
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
	find paper -maxdepth 1 -type f \( -name '*.aux' -o -name '*.log' -o -name '*.out' \) -delete
	rm -rf .cache

clean-intermediates:
	find results -maxdepth 1 -type f -name '*.npy' -delete
