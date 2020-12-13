TEST_DATA_GENERATE_SCRIPT=./src/scripts/test_data/test_data_generate.py

help:
	@echo 'Makefile for vak                                                                 '
	@echo '                                                                                 '
	@echo 'Usage:                                                                           '
	@echo '   make test-data-generate                   generate test data used by tests    '
	@echo '   make test-data-clean                      remove generated test data          '
	@echo '   make variables                            show variables defined for Makefile '

variables:
	@echo '     TESTS_RESULTS_SCRIPT    : $(TESTS_RESULTS_SCRIPT)       '
	@echo '     TESTS_CLEAN_SCRIPT      : $(TESTS_CLEAN_SCRIPT)         '

test-data-generate : $(TEST_DATA_GENERATE_SCRIPT)
	python $(TEST_DATA_GENERATE_SCRIPT)

test-data-clean :
	rm -rfv ./tests/test_data/generated/*


.PHONY: help variables test-data-generate test-data-clean
