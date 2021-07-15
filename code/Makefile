.PHONY: install
install: Pipfile
	python3 -m pipenv install

.PHONY: run
run: install
	python3 -m pipenv run python3 ./scripts/solutions.py --question $(QUESTION)
