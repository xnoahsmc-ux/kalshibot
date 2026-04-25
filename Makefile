PY ?= python3
SRC = src
TESTS = tests
PYTHONPATH ?= $(SRC)
export PYTHONPATH

.PHONY: install test backtest paper run report halt resume lint clean

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

test:
	$(PY) -m pytest $(TESTS) -q

backtest:
	$(PY) -m kalshibot.v2.cli backtest

paper:
	$(PY) -m kalshibot.v2.cli paper

run:
	$(PY) -m kalshibot.v2.cli run

report:
	$(PY) -m kalshibot.v2.cli report

halt:
	mkdir -p data && touch data/HALT
	@echo "HALT engaged. v2 will refuse all new orders until 'make resume'."

resume:
	rm -f data/HALT
	@echo "HALT cleared."

lint:
	$(PY) -m ruff check $(SRC) $(TESTS) || true

clean:
	rm -f data/markets.db data/paper_ledger.jsonl data/journal.jsonl
	rm -rf .pytest_cache __pycache__
