SOURCE_TEST_DATA_TAR=tests/test_data/source/source_test_data.tar.gz
SOURCE_TEST_DATA_URL=https://osf.io/7ru4s/download

TEST_DATA_GENERATE_SCRIPT=./src/scripts/test_data/test_data_generate.py
GENERATED_TEST_DATA_TAR=tests/test_data/generated/generated_test_data.tar.gz
GENERATED_TEST_DATA_URL=https://osf.io/2c39a/download
GENERATED_TEST_DATA_TOP_LEVEL_DIRS=tests/test_data/generated/configs tests/test_data/generated/prep tests/test_data/generated/results

help:
	@echo 'Makefile for vak                                                           			'
	@echo '                                                                           			'
	@echo 'Usage:                                                                     			'
	@echo '     make test-data-clean-source          remove source test data                        '
	@echo '     make test-data-download-source       download source test data                      '
	@echo '     make test-data-generate              generate vak files used by tests from source data   '
	@echo '     make test-data-clean-generate        remove generated test data          					'
	@echo '     make test-data-tar-generate          place generated test data in compressed tar file       	'
	@echo '     make test-data-download-generate     download generated test data .tar and expand        	'
	@echo '     make variables              show variables defined for Makefile 					'

variables:
	@echo '     SOURCE_TEST_DATA_TAR                : $(GENERATED_TEST_DATA_TAR)				'
	@echo '     SOURCE_TEST_DATA_URL                : $(GENERATED_TEST_DATA_URL)				'
	@echo '     TESTS_DATA_GENERATE_SCRIPT    		: $(TEST_DATA_GENERATE_SCRIPT)				'
	@echo '     GENERATED_TEST_DATA_TAR      		: $(GENERATED_TEST_DATA_TAR)				'
	@echo '     GENERATED_TEST_DATA_URL      		: $(GENERATED_TEST_DATA_URL)				'
	@echo '     GENERATED_TEST_DATA_TOP_LEVEL_DIRS	: $(GENERATED_TEST_DATA_TOP_LEVEL_DIRS)		'

test-data-clean-source:
	rm -rfv ./tests/test_data/source/*

test-data-download-source:
	wget $(SOURCE_TEST_DATA_URL) -O $(SOURCE_TEST_DATA_TAR)
	tar -xvzf $(SOURCE_TEST_DATA_TAR)

test-data-generate : $(TEST_DATA_GENERATE_SCRIPT)
	python $(TEST_DATA_GENERATE_SCRIPT)

test-data-clean-generate :
	rm -rfv ./tests/test_data/generated/*

test-data-tar-generate:
	tar -czvf $(GENERATED_TEST_DATA_TAR) $(GENERATED_TEST_DATA_TOP_LEVEL_DIRS)

test-data-download-generate:
	wget $(GENERATED_TEST_DATA_URL) -O $(GENERATED_TEST_DATA_TAR)
	tar -xvzf $(GENERATED_TEST_DATA_TAR)

.PHONY: help variables test-data-clean-source test-data-download-source test-data-generate test-data-clean-generate test-data-tar-generate test-data-download-generate
