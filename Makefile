.PHONY: all clean

clean :
	rm -rf ./tests/test_data/results/*
	rm -rf ./tests/test_data/spects/*


all : results

results : data
	python ./src/bin/remake_results.py

data :
	python ./src/bin/remake_spects.py
