include .make/venv.Makefile

TEST_DATA_GENERATE_SCRIPT=./tests/scripts/generate_data_for_tests.py

DATA_FOR_TESTS_DIR=./tests/data_for_tests/
GENERATED_TEST_DATA_DIR=${DATA_FOR_TESTS_DIR}generated/
CONFIGS_DIR=${GENERATED_TEST_DATA_DIR}configs
PREP_DIR=${GENERATED_TEST_DATA_DIR}prep/
RESULTS_DIR=${GENERATED_TEST_DATA_DIR}results/
RESULTS_CI=$(shell ls -d ${RESULTS_DIR}*/*/teenytweetynet)
GENERATED_TEST_DATA_CI_DIRS=${CONFIGS_DIR} ${PREP_DIR} ${RESULTS_CI}
GENERATED_TEST_DATA_ALL_DIRS=${GENERATED_TEST_DATA_CI_DIRS} $(shell ls -d ${RESULTS_DIR}/*/*/tweetynet)

SOURCE_TEST_DATA_TAR=${DATA_FOR_TESTS_DIR}source/source_test_data.tar.gz
GENERATED_TEST_DATA_CI_TAR=${GENERATED_TEST_DATA_DIR}generated_test_data.ci.tar.gz
GENERATED_TEST_DATA_ALL_TAR=${GENERATED_TEST_DATA_DIR}generated_test_data.tar.gz

SOURCE_TEST_DATA_URL=https://osf.io/mjksu/download
GENERATED_TEST_DATA_ALL_URL=https://osf.io/8swue/download
GENERATED_TEST_DATA_CI_URL=https://osf.io/tcwe8/download

help:
	@echo 'Makefile for vak                                                           			'
	@echo '                                                                           			'
	@echo 'Usage:                                                                     			'
	@echo '     make test-data-clean-source				remove source test data                        '
	@echo '     make test-data-download-source          download source test data                      '
	@echo '     make test-data-generate                 generate vak files used by tests from source data   '
	@echo '     make test-data-clean-generated          remove generated test data          					'
	@echo '     make test-data-tar-generated-all        place all generated test data in compressed tar file       	'
	@echo '     make test-data-tar-generated-ci         place generated test data for CI in compressed tar file       	'
	@echo '     make test-data-download-generated-all   download .tar with all generated test data and expand        	'
	@echo '     make test-data-download-generated-ci    download .tar with generated test data for CI and expand        	'
	@echo '     make variables                          show variables defined for Makefile 					'
	@echo '     make venv	                     	    make virtual environment (as `.venv`) if one does not exist 					'
	@echo '     make clean-venv                         remove any existing virtual environment (`rm -rf .venv`)					'
	@echo '     make show-venv                          show virtual environment and variables used to make it	'

variables:
	@echo '     TESTS_DATA_GENERATE_SCRIPT      : $(TEST_DATA_GENERATE_SCRIPT)				'
	@echo ''
	@echo '     DATA_FOR_TESTS_DIR              : $(DATA_FOR_TESTS_DIR)		'
	@echo '     GENERATED_TEST_DATA_DIR         : $(GENERATED_TEST_DATA_DIR)		'
	@echo '     PREP_DIR                        : $(PREP_DIR)		'
	@echo '     RESULTS_DIR                     : $(RESULTS_DIR)		'
	@echo '     RESULTS_CI                      : $(RESULTS_CI)		'
	@echo '     GENERATED_TEST_DATA_CI_DIRS     : $(GENERATED_TEST_DATA_CI_DIRS)		'
	@echo '     GENERATED_TEST_DATA_ALL_DIRS    : $(GENERATED_TEST_DATA_ALL_DIRS)		'
	@echo ''
	@echo '     SOURCE_TEST_DATA_TAR            : $(SOURCE_TEST_DATA_TAR)				'
	@echo '     GENERATED_TEST_DATA_CI_TAR      : $(GENERATED_TEST_DATA_CI_TAR)				'
	@echo '     GENERATED_TEST_DATA_ALL_TAR     : $(GENERATED_TEST_DATA_ALL_TAR)				'
	@echo ''
	@echo '     SOURCE_TEST_DATA_URL            : $(SOURCE_TEST_DATA_URL)				'
	@echo '     GENERATED_TEST_DATA_ALL_URL 	: $(GENERATED_TEST_DATA_ALL_URL)				'
	@echo '     GENERATED_TEST_DATA_CI_URL      : $(GENERATED_TEST_DATA_CI_URL)				'

test-data-clean-source:
	rm -rfv ./tests/data_for_tests/source/*

test-data-download-source:
	wget -q $(SOURCE_TEST_DATA_URL) -O $(SOURCE_TEST_DATA_TAR)
	tar -xzf $(SOURCE_TEST_DATA_TAR)

test-data-generate : $(TEST_DATA_GENERATE_SCRIPT)
	python $(TEST_DATA_GENERATE_SCRIPT)

test-data-clean-generated :
	rm -rfv ./tests/data_for_tests/generated/*

test-data-tar-generated-all:
	tar -czvf $(GENERATED_TEST_DATA_ALL_TAR) $(GENERATED_TEST_DATA_ALL_DIRS)

test-data-tar-generated-ci:
	tar -czvf $(GENERATED_TEST_DATA_CI_TAR) $(GENERATED_TEST_DATA_CI_DIRS)

test-data-download-generated-all:
	wget -q $(GENERATED_TEST_DATA_ALL_URL) -O $(GENERATED_TEST_DATA_ALL_TAR)
	tar -xzf $(GENERATED_TEST_DATA_ALL_TAR)

test-data-download-generated-ci:
	wget -q $(GENERATED_TEST_DATA_CI_URL) -O $(GENERATED_TEST_DATA_CI_TAR)
	tar -xzf $(GENERATED_TEST_DATA_CI_TAR)

.PHONY: help variables \
        test-data-clean-source test-data-download-source \
        test-data-generate test-data-clean-generated \
        test-data-tar-generated-all test-data-tar-generated-all \
        test-data-download-generated-all test-data-download-generated-ci
