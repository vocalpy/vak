SOURCE_TEST_DATA_TAR=tests/test_data/source/source_test_data.tar.gz
SOURCE_TEST_DATA_URL=https://osf.io/v4f83/download

TEST_DATA_GENERATE_SCRIPT=./src/scripts/test_data/test_data_generate.py
GENERATED_TEST_DATA_TAR=tests/test_data/generated/generated_test_data.tar.gz
GENERATED_TEST_DATA_URL=https://osf.io/2c39a/download
GENERATED_TEST_DATA_TOP_LEVEL_DIRS=tests/test_data/generated/configs tests/test_data/generated/prep tests/test_data/generated/results

help:
	@echo 'Makefile for vak                                                           			'
	@echo '                                                                           			'
	@echo 'Usage:                                                                     			'
	@echo '     make test-data-generate     generate test data used by tests    					'
	@echo '     make test-data-tar          place generated test data in compressed tar file       	'
	@echo '     make test-data-download     download generated test data .tar and expand        	'
	@echo '     make test-data-clean        remove generated test data          					'
	@echo '     make variables              show variables defined for Makefile 					'

variables:
	@echo '     SOURCE_TEST_DATA_TAR                : $(GENERATED_TEST_DATA_TAR)				'
	@echo '     SOURCE_TEST_DATA_URL                : $(GENERATED_TEST_DATA_URL)				'
	@echo '     TESTS_DATA_GENERATE_SCRIPT    		: $(TEST_DATA_GENERATE_SCRIPT)				'
	@echo '     GENERATED_TEST_DATA_TAR      		: $(GENERATED_TEST_DATA_TAR)				'
	@echo '     GENERATED_TEST_DATA_URL      		: $(GENERATED_TEST_DATA_URL)				'
	@echo '     GENERATED_TEST_DATA_TOP_LEVEL_DIRS	: $(GENERATED_TEST_DATA_TOP_LEVEL_DIRS)		'

test-data-generate : $(TEST_DATA_GENERATE_SCRIPT)
	python $(TEST_DATA_GENERATE_SCRIPT)

test-data-tar:
	tar -czvf $(GENERATED_TEST_DATA_TAR) $(GENERATED_TEST_DATA_TOP_LEVEL_DIRS)

test-data-download:
	wget $(SOURCE_TEST_DATA_URL) -O $(SOURCE_TEST_DATA_TAR)
	tar -xvzf $(SOURCE_TEST_DATA_TAR)
	wget $(GENERATED_TEST_DATA_URL) -O $(GENERATED_TEST_DATA_TAR)
	tar -xvzf $(GENERATED_TEST_DATA_TAR)

test-data-clean :
	rm -rfv ./tests/test_data/source/*
	rm -rfv ./tests/test_data/generated/*


.PHONY: help variables test-data-generate test-data-tar test-data-download test-data-clean
