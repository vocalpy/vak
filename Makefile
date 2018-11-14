.PHONY: all clean

clean :
	rm -rf ./tests/test_data/results
	rm -rf ./tests/test_data/spect_files


all : results

results : data
	python ./src/bin/remake_results.py

data :
	python ./src/bin/remake_spects.py
