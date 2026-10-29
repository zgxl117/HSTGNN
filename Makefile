.DEFAULT_GOAL := help

# Pass values via exported environment variables, never interpolate user paths
# into shell commands. Spaces and shell metacharacters stay literal arguments.
PYTHON := $(or $(PYTHON),$(HSTGNN_PYTHON))
DATASET := $(or $(DATASET),$(HSTGNN_DATASET))
PROFILE := $(or $(PROFILE),$(HSTGNN_PROFILE))
RUN_NAME := $(or $(RUN_NAME),$(HSTGNN_RUN_NAME))
TRAIN_CONFIG := $(or $(TRAIN_CONFIG),$(HSTGNN_TRAIN_CONFIG))
PREDICTIONS := $(or $(PREDICTIONS),$(HSTGNN_PREDICTIONS))
SMOKE := $(or $(SMOKE),$(HSTGNN_SMOKE),0)
DRY_RUN := $(or $(DRY_RUN),$(HSTGNN_DRY_RUN),0)

export HSTGNN_PYTHON := $(PYTHON)
export HSTGNN_DATASET := $(DATASET)
export HSTGNN_PROFILE := $(PROFILE)
export HSTGNN_RUN_NAME := $(RUN_NAME)
export HSTGNN_TRAIN_CONFIG := $(TRAIN_CONFIG)
export HSTGNN_PREDICTIONS := $(PREDICTIONS)
export HSTGNN_SMOKE := $(SMOKE)
export HSTGNN_DRY_RUN := $(DRY_RUN)

.PHONY: help ui smoke run reuse check datasets test

help:
	@bash ./run.sh help
	@printf '\nMake 用法：make ui | smoke | run | reuse | check | datasets | test\n'
	@printf '变量：DATASET=... PYTHON=... PROFILE=... RUN_NAME=... DRY_RUN=1\n'
	@printf '复用：make reuse TRAIN_CONFIG=... PREDICTIONS=... SMOKE=1\n'

ui smoke run reuse check datasets test:
	@bash ./run.sh $@
