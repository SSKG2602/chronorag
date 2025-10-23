.PHONY: run test

run:
	python -m app.uvicorn_runner

test:
	pytest -q
