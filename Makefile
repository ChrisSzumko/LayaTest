.DEFAULT_GOAL := help

PYTHON ?= .venv/bin/python
PORT ?= 8501
THRESHOLD ?= 0.80
REPORT_DIR ?= reports
TEST_ARGS ?= -q

.PHONY: help setup test webui evaluate evaluate-ungated smoke

help:
	@printf '%s\n' \
	  'Laya playground commands:' \
	  '  make setup              Install locked dependencies with uv' \
	  '  make test               Run offline software tests' \
	  '  make webui              Start localhost UI (Ctrl+C to stop)' \
	  '  make evaluate           Evaluate all scenarios (threshold 0.80)' \
	  '  make evaluate-ungated   Evaluate with threshold zero; reports/ungated' \
	  '  make smoke              Evaluate three cases; reports/smoke' \
	  '  make help               Show this help (also the default target)' \
	  '' \
	  'Overrides: PYTHON, PORT, THRESHOLD, REPORT_DIR, TEST_ARGS' \
	  '  make webui PORT=8502' \
	  '  make evaluate THRESHOLD=0.5 REPORT_DIR=reports/custom' \
	  '  make test TEST_ARGS="-q -k permission"' \
	  '  HF_HUB_OFFLINE=1 make webui' \
	  'Run make setup first; other targets never install dependencies implicitly.'

setup:
	uv sync --locked

test:
	"$(PYTHON)" -m pytest $(TEST_ARGS)

webui:
	"$(PYTHON)" main.py --server.port "$(PORT)"

evaluate:
	"$(PYTHON)" main.py --evaluate --threshold "$(THRESHOLD)" --report-dir "$(REPORT_DIR)"

evaluate-ungated: THRESHOLD = 0
evaluate-ungated: REPORT_DIR = reports/ungated
evaluate-ungated:
	"$(PYTHON)" main.py --evaluate --threshold "$(THRESHOLD)" --report-dir "$(REPORT_DIR)"

smoke: REPORT_DIR = reports/smoke
smoke:
	"$(PYTHON)" main.py --evaluate --limit 3 --threshold "$(THRESHOLD)" --report-dir "$(REPORT_DIR)"
