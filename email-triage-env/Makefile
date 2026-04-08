.PHONY: install serve test validate demo docker-build docker-run zip clean

## Install Python dependencies
install:
	pip install -r requirements.txt

## Start the OpenEnv server locally (port 7860)
serve:
	python app.py

## Run the full unit-test suite
test:
	pytest tests/ -v --tb=short

## Run the local OpenEnv validator against a running server
validate:
	python validate.py --url http://localhost:7860

## Run the rule-based demo agent (no API key required)
demo:
	python demo_agent.py --env-url http://localhost:7860

## Run the LLM baseline (requires OPENAI_API_KEY)
baseline:
	python baseline.py --model gpt-4o-mini

## Build the Docker image
docker-build:
	docker build -t email-triage-env .

## Run the Docker container
docker-run:
	docker run --rm -p 7860:7860 email-triage-env

## Full CI sequence: test → serve (bg) → validate → demo → stop server
ci:
	pytest tests/ -v --tb=short
	python app.py & \
	  sleep 4 && \
	  python validate.py && \
	  python demo_agent.py && \
	  kill %1

## Package source into a zip (excludes cache/pycache)
zip:
	zip -r email-triage-env.zip . \
	  --exclude "*.pyc" \
	  --exclude "*/__pycache__/*" \
	  --exclude "*/.pytest_cache/*" \
	  --exclude "*.egg-info/*"
	@echo "Created email-triage-env.zip"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache
