# adapted from
# https://github.com/sio/Makefile.venv             v2021.12.16
#
#
# Insert `include Makefile.venv` at the bottom of your Makefile to enable these
# rules.
#
# When writing your Makefile use '$(VENV)/python' to refer to the Python
# interpreter within virtual environment and '$(VENV)/executablename' for any
# other executable in venv.
#
# This Makefile provides the following targets:
#   venv
#       Create a virtual environment if one does not exist.
#       After running `make venv`, activate the virtual environment
#       with `. ./bin/activate`
#   show-venv
#       Show versions of Python and pip, and the path to the virtual environment
#   clean-venv
#       Remove virtual environment
# This Makefile can be configured via following variables:
#   PY
#       Command name for system Python interpreter. It is used only initialy to
#       create the virtual environment
#       Default: python3
#   WORKDIR
#       Parent directory for the virtual environment.
#       Default: current working directory.
#   VENVDIR
#       Python virtual environment directory.
#       Default: $(WORKDIR)/.venv
#
# This Makefile was written for GNU Make and may not work with other make
# implementations.
#
#
# Copyright (c) 2019-2020 Vitaly Potyarkin
#
# Licensed under the Apache License, Version 2.0
#    <http://www.apache.org/licenses/LICENSE-2.0>
#


#
# Configuration variables
#

WORKDIR?=.
VENVDIR?=$(WORKDIR)/.venv
MARKER=.initialized-with-Makefile.venv


#
# Python interpreter detection
#

_PY_AUTODETECT_MSG=Detected Python interpreter: $(PY). Use PY environment variable to override

ifeq (ok,$(shell test -e /dev/null 2>&1 && echo ok))
NULL_STDERR=2>/dev/null
else
NULL_STDERR=2>NUL
endif

ifndef PY
_PY_OPTION:=python3
ifeq (ok,$(shell $(_PY_OPTION) -c "print('ok')" $(NULL_STDERR)))
PY=$(_PY_OPTION)
endif
endif

ifndef PY
_PY_OPTION:=$(VENVDIR)/bin/python
ifeq (ok,$(shell $(_PY_OPTION) -c "print('ok')" $(NULL_STDERR)))
PY=$(_PY_OPTION)
$(info $(_PY_AUTODETECT_MSG))
endif
endif

ifndef PY
_PY_OPTION:=$(subst /,\,$(VENVDIR)/Scripts/python)
ifeq (ok,$(shell $(_PY_OPTION) -c "print('ok')" $(NULL_STDERR)))
PY=$(_PY_OPTION)
$(info $(_PY_AUTODETECT_MSG))
endif
endif

ifndef PY
_PY_OPTION:=py -3
ifeq (ok,$(shell $(_PY_OPTION) -c "print('ok')" $(NULL_STDERR)))
PY=$(_PY_OPTION)
$(info $(_PY_AUTODETECT_MSG))
endif
endif

ifndef PY
_PY_OPTION:=python
ifeq (ok,$(shell $(_PY_OPTION) -c "print('ok')" $(NULL_STDERR)))
PY=$(_PY_OPTION)
$(info $(_PY_AUTODETECT_MSG))
endif
endif

ifndef PY
define _PY_AUTODETECT_ERR
Could not detect Python interpreter automatically.
Please specify path to interpreter via PY environment variable.
endef
$(error $(_PY_AUTODETECT_ERR))
endif


#
# Internal variable resolution
#

VENV=$(VENVDIR)/bin
EXE=
# Detect windows
ifeq (win32,$(shell $(PY) -c "import __future__, sys; print(sys.platform)"))
VENV=$(VENVDIR)/Scripts
EXE=.exe
endif

touch=touch $(1)
ifeq (,$(shell command -v touch $(NULL_STDERR)))
# https://ss64.com/nt/touch.html
touch=type nul >> $(subst /,\,$(1)) && copy /y /b $(subst /,\,$(1))+,, $(subst /,\,$(1))
endif

ifeq (,$(shell command -v $(firstword $(RM)) $(NULL_STDERR)))
RMDIR=rd /s /q
else
RMDIR=$(RM) -r
endif


#
# Virtual environment
#

.PHONY: venv
venv: $(VENV)/$(MARKER)

.PHONY: clean-venv
clean-venv:
	-$(RMDIR) "$(VENVDIR)"

.PHONY: show-venv
show-venv: venv
	@$(VENV)/python -c "import sys; print('Python ' + sys.version.replace('\n',''))"
	@$(VENV)/pip --version
	@echo venv: $(VENVDIR)

.PHONY: debug-venv
debug-venv:
	@echo "PATH (Shell)=$$PATH"
	@$(MAKE) --version
	$(info PATH (GNU Make)="$(PATH)")
	$(info SHELL="$(SHELL)")
	$(info PY="$(PY)")
	$(info VENVDIR="$(VENVDIR)")
	$(info VENVDEPENDS="$(VENVDEPENDS)")
	$(info WORKDIR="$(WORKDIR)")


$(VENV):
	$(PY) -m venv $(VENVDIR)
	$(VENV)/python -m pip install --upgrade pip setuptools wheel
	. $(VENV)/activate && pip install -e '.[test,doc,dev]'

$(VENV)/$(MARKER): $(VENVDEPENDS) | $(VENV)
	$(call touch,$(VENV)/$(MARKER))

#
# Commandline tools (wildcard rule, executable name must match package name)
#

ifneq ($(EXE),)
$(VENV)/%: $(VENV)/%$(EXE) ;
.PHONY:    $(VENV)/%
.PRECIOUS: $(VENV)/%$(EXE)
endif

$(VENV)/%$(EXE): $(VENV)/$(MARKER)
	$(VENV)/pip install --upgrade $*
	$(call touch,$@)
