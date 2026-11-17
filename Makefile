# Allocator skeleton. Swap the problem with PACK=<folder under allocator/packs>.
# Deploy flags checked against `adk deploy cloud_run --help` / `agent_engine --help` (ADK 2.14.1).
SHELL := /bin/bash
PY ?= python3
ADK ?= adk
PACK ?= dallas_heat
STORE ?= memory
APPROVAL_MODE ?= confirm
REGION ?= us-central1
PROJECT ?= $(GOOGLE_CLOUD_PROJECT)
SERVICE ?= oasis-agent
PYV ?= 3.12
BQ_DATASET ?= a4i_climate
BQ_LOCATION ?= US
ROOT_MODEL ?= gemini-2.5-flash
NUMBER_CHECK ?= flag
GENAI_LOCATION ?= $(REGION)
CLI = PACK=$(PACK) STORE=$(STORE) APPROVAL_MODE=$(APPROVAL_MODE) BQ_PROJECT=$(PROJECT) BQ_DATASET=$(BQ_DATASET) \
  BQ_LOCATION=$(BQ_LOCATION) $(PY) -m oasis_engine.cli
RUNTIME_ENV = PACK=$(PACK) STORE=$(STORE) APPROVAL_MODE=$(APPROVAL_MODE) ROOT_MODEL=$(ROOT_MODEL) \
  NUMBER_CHECK=$(NUMBER_CHECK) BQ_PROJECT=$(PROJECT) BQ_DATASET=$(BQ_DATASET) BQ_LOCATION=$(BQ_LOCATION) \
  GOOGLE_GENAI_USE_VERTEXAI=TRUE


.PHONY: help install test lint validate seed demo dial rank compare state propose approve commit reject \
        shock conditions verify-audit outbox web run guardian conditions-agent eval-agent \
        deploy-run deploy-engine deploy-conditions proxy preflight new-pack test-cloud clean

help:  ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-18s %s\n", $$1, $$2}'

install:  ## Create .venv and install dependencies
	python3.13 -m venv .venv && $(PY) -m pip install -U pip && $(PY) -m pip install -r requirements-dev.txt

test:  ## Offline test suite (engine, gateway, agent wiring, guardian, conditions, lints)
	$(PY) -m pytest -q tests

lint:  ## Static security lints only
	$(PY) -m pytest -q tests/test_static_lints.py

validate:  ## Validate the pack (add STORE=bigquery to check tables/columns too)
	$(CLI) validate --pack $(PACK) $(if $(filter bigquery,$(STORE)),--with-store,)

seed:  ## Load seed data (BigQuery: dataset + DDL + load jobs; memory: reset)
	$(CLI) seed

demo:  ## Full story in one process: refuse, approve, shock->STALE, re-plan, commit, audit
	$(CLI) demo

dial:  ## What does fairness cost? alpha sweep
	$(CLI) dial

rank compare state outbox verify-audit:  ## Read-only reports
	$(CLI) $@

propose:  ## Propose a plan (nothing moves)
	$(CLI) propose

approve:  ## Human approval: make approve PLAN=P-xxxx BY=name
	$(CLI) approve $(PLAN) --by $(BY)

commit:  ## make commit PLAN=P-xxxx CODE=XXXXXX
	$(CLI) commit $(PLAN) $(CODE)

reject:  ## make reject PLAN=P-xxxx REASON="..."
	$(CLI) reject $(PLAN) --reason "$(REASON)"

shock:  ## Field update: make shock LOT=L01 DELTA=-10
	$(CLI) shock $(or $(LOT),L01) $(or $(DELTA),-10) --note "truck arrived short"

conditions:  ## Refresh conditions from the provider (or FILE=rows.json)
	$(CLI) conditions $(if $(FILE),--file $(FILE),)

web:  ## adk web with allocator, guardian, conditions_agent (needs model access)
	PACK=$(PACK) STORE=$(STORE) APPROVAL_MODE=$(APPROVAL_MODE) $(ADK) web --port 8000 .

run:  ## Terminal chat with the root agent
	PACK=$(PACK) STORE=$(STORE) APPROVAL_MODE=$(APPROVAL_MODE) $(ADK) run allocator

guardian:  ## Drive the guardian Workflow offline (add ARGS=shock)
	PACK=$(PACK) STORE=memory PYTHONPATH=. $(PY) scripts/guardian_drive.py $(ARGS)

conditions-agent:  ## Serve the A2A conditions agent locally on :8001
	PORT=8001 $(PY) -m conditions_agent.server

eval-agent:  ## adk eval golden + red-team sets (needs model access)
	PACK=$(PACK) STORE=memory APPROVAL_MODE=confirm $(ADK) eval allocator evals/allocator.evalset.json --config_file_path evals/test_config.json --print_detailed_results

deploy-run:  ## Cloud Run + web UI, private, 1 instance. Open with: make proxy
	@test -n "$(PROJECT)" || (echo "Set PROJECT=<gcp-project-id>"; exit 1)
	$(ADK) deploy cloud_run --project $(PROJECT) --region $(REGION) --service_name $(SERVICE) --with_ui \
	  --python_version $(PYV) $(foreach kv,$(RUNTIME_ENV) GOOGLE_CLOUD_LOCATION=$(GENAI_LOCATION),--env $(kv)) allocator \
	  -- --no-allow-unauthenticated --max-instances=1

proxy:  ## Open the private Cloud Run service at http://localhost:8080 (your gcloud identity)
	gcloud run services proxy $(SERVICE) --project $(PROJECT) --region $(REGION) --port 8080

deploy-engine:  ## Agent Runtime. Requires STORE=bigquery APPROVAL_MODE=out_of_band
	@test -n "$(PROJECT)" || (echo "Set PROJECT=<gcp-project-id>"; exit 1)
	@test "$(STORE)" = bigquery || (echo "Agent Runtime runs many instances: use STORE=bigquery"; exit 1)
	@test "$(APPROVAL_MODE)" = out_of_band || (echo "Use APPROVAL_MODE=out_of_band (confirmation on Agent Runtime is unverified)"; exit 1)
	$(PY) scripts/write_engine_config.py /tmp/allocator_engine_config.json $(RUNTIME_ENV)
	$(ADK) deploy agent_engine --project $(PROJECT) --region $(REGION) --display_name $(SERVICE) \
	  --python_version $(PYV) --agent_engine_config_file /tmp/allocator_engine_config.json allocator

deploy-conditions:  ## STRETCH-3: conditions agent to Cloud Run (public; synthetic data only)
	gcloud run deploy conditions-agent --source conditions_agent --project $(PROJECT) --region $(REGION) \
	  --allow-unauthenticated --max-instances=1

preflight:  ## Check local + cloud readiness (read-only; prints what is missing)
	PROJECT=$(PROJECT) REGION=$(REGION) BQ_DATASET=$(BQ_DATASET) PACK=$(PACK) $(PY) scripts/preflight.py

new-pack:  ## Copy dallas_heat to a new pack: make new-pack NAME=shelter_beds
	@test -n "$(NAME)" || (echo "Set NAME=<pack_name>"; exit 1)
	$(PY) scripts/new_pack.py $(NAME)


test-cloud:  ## Live BigQuery contract test: make test-cloud PROJECT=<id> (dataset allocator_test)
	RUN_BQ=1 STORE=bigquery BQ_PROJECT=$(PROJECT) BQ_DATASET=$(or $(TEST_DATASET),allocator_test) BQ_LOCATION=$(BQ_LOCATION) \
	  $(PY) -m pytest -q tests/test_bigquery_live.py

clean:
	rm -rf .pytest_cache **/__pycache__
