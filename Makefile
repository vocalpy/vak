.PHONY: all clean

clean :
	rm -rf ./tests/test_data/results/*
	rm -rf ./tests/test_data/spects/*
	rm ./src/bin/tmp_Makefile_config.ini

all : results

results : data
	python ./src/bin/remake_results.py

data : config
	python ./src/bin/remake_spects.py

config :
	cp ./src/bin/Makefile_config.ini ./src/bin/tmp_Makefile_config.ini
