.PHONY: all clean download

clean :
	rm -rf ./tests/test_data/results/*
	rm -rf ./tests/test_data/spects/*
	rm ./tests/setup_scripts/tmp_Makefile_config.ini

download :
	wget -O ./tests/test_data/spects/spects.tar.gz "https://ndownloader.figshare.com/files/14523563"
	tar -xvf ./tests/test_data/spects/spects.tar.gz -C ./tests/test_data/spects/
	wget -O ./tests/test_data/results/results.tar.gz "https://ndownloader.figshare.com/files/14523560"
	tar -xvf ./tests/test_data/results/results.tar.gz -C ./tests/test_data/results/

all : results

results : data
	python ./tests/setup_scripts/remake_results.py

data : config
	python ./tests/setup_scripts/remake_spects.py

config :
	cp ./tests/setup_scripts/Makefile_config.ini ./tests/setup_scripts/tmp_Makefile_config.ini
