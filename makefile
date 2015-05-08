PYTHON = python

UI_FILES = $(shell find -type f -name '*.ui')

.PHONY: test run i18n

run:
	$(PYTHON) nlmeans.py

ui: $(UI_FILES)
	$(PYTHON) -c 'import PyQt4.uic; PyQt4.uic.compileUiDir(".", recurse=True)'
