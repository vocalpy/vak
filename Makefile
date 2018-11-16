.PHONY: all clean

clean :
	rm -rf ./tests/test_data/results/*
	rm -rf ./tests/test_data/spects/*
	rm ./src/bin/tmp_Makefile_config.ini

all : results

results : data
	python ./tests/setup_scripts/remake_results.py

data : config
	python ./tests/setup_scripts/remake_spects.py

config :
	cp ./tests/setup_scripts/Makefile_config.ini ./tests/setup_scripts/tmp_Makefile_config.ini
