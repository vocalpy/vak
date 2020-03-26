SPECTS_SCRIPT=./tests/setup_scripts/rerun_prep.py
RESULTS_SCRIPT=./tests/setup_scripts/rerun_learncurve.py

.PHONY: variables clean results all

variables:
	@echo SPECTS_SCRIPT: $(SPECTS_SCRIPT)
	@echo RESULTS_SCRIPT: $(RESULTS_SCRIPT)

clean :
	rm -rf ./tests/test_data/configs/test_*.toml
	rm -rf ./tests/test_data/prep/*/*
	rm -rf ./tests/test_data/results/train/*/
	rm -rf ./tests/test_data/results/learncurve/*/
	rm -rf ./tests/test_data/results/predict/*/

results : $(RESULTS_SCRIPT) $(SPECTS_SCRIPT)
	python $(SPECTS_SCRIPT)
	python $(RESULTS_SCRIPT)

all : results
