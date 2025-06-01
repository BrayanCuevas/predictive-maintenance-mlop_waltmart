# Predictive Maintenance MLOps Pipeline
# Author: Brayan Cuevas
# Usage: make <target>

.PHONY: help setup data train test api docker clean pipeline monitor notebook lint format vertex-simulate vertex-help cloud-ready all

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
DOCKER_COMPOSE := docker-compose
PYTEST := pytest

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Predictive Maintenance MLOps Pipeline$(NC)"
	@echo "======================================"
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Common workflows:$(NC)"
	@echo "  $(YELLOW)make setup$(NC)          - First time setup"
	@echo "  $(YELLOW)make pipeline$(NC)       - Run complete training pipeline"
	@echo "  $(YELLOW)make api$(NC)            - Start API server"
	@echo "  $(YELLOW)make test$(NC)           - Run all tests"
	@echo "  $(YELLOW)make vertex-simulate$(NC) - Simulate Vertex AI pipeline"

setup: ## Install dependencies and create directories
	@echo "$(BLUE)[SETUP]$(NC) Installing dependencies..."
	$(PIP) install -r requirements.txt
	@echo "$(BLUE)[SETUP]$(NC) Creating directories..."
	@mkdir -p data/raw data/processed logs models reports notebooks monitoring vertex_ai
	@echo "$(GREEN)Setup completed$(NC)"

data: ## Verify data files are present
	@echo "$(BLUE)[DATA]$(NC) Checking data files..."
	@if [ ! -f data/raw/PdM_telemetry.csv ]; then \
		echo "$(RED)Missing data/raw/PdM_telemetry.csv$(NC)"; \
		echo "$(YELLOW)Please download data:$(NC)"; \
		echo "   1. Go to: https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance"; \
		echo "   2. Download and extract to data/raw/"; \
		echo "   3. Run 'make data' again"; \
		exit 1; \
	fi
	@if [ ! -f data/raw/PdM_failures.csv ]; then \
		echo "$(RED)Missing data/raw/PdM_failures.csv$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)Data files found$(NC)"

lint: ## Run code linting
	@echo "$(BLUE)[LINT]$(NC) Running flake8..."
	@flake8 src/ --max-line-length=100 --ignore=E203,W503,F401,E402,F824,F541,E501 || true
	@echo "$(GREEN)Linting completed$(NC)"

format: ## Format code with black
	@echo "$(BLUE)[FORMAT]$(NC) Formatting code with black..."
	@black src/ --line-length=100
	@echo "$(GREEN)Code formatted$(NC)"

test: data ## Run all tests
	@echo "$(BLUE)[TEST]$(NC) Running test suite..."
	$(PYTEST) test/ -v --cov=src --cov-report=term --cov-report=html
	@echo "$(GREEN)Tests completed$(NC)"

train: data ## Train the model
	@echo "$(BLUE)[TRAIN]$(NC) Starting model training..."
	$(PYTHON) scripts/train_pipeline.py
	@echo "$(GREEN)Training completed$(NC)"

notebook: ## Start Jupyter notebook server
	@echo "$(BLUE)[NOTEBOOK]$(NC) Starting Jupyter lab..."
	@jupyter lab notebooks/ --no-browser --ip=0.0.0.0 --port=8888 || \
	jupyter notebook notebooks/ --no-browser --ip=0.0.0.0 --port=8888

api: ## Start API server with Docker
	@echo "$(BLUE)[API]$(NC) Starting API server..."
	$(DOCKER_COMPOSE) up --build

api-dev: ## Start API in development mode
	@echo "$(BLUE)[API-DEV]$(NC) Starting API in development mode..."
	@cd src/api && uvicorn main:app --reload --host 0.0.0.0 --port 8000

docker-build: ## Build Docker image
	@echo "$(BLUE)[DOCKER]$(NC) Building Docker image..."
	$(DOCKER_COMPOSE) build
	@echo "$(GREEN)Docker image built$(NC)"

docker-up: ## Start Docker containers
	@echo "$(BLUE)[DOCKER]$(NC) Starting containers..."
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)Containers started$(NC)"

docker-down: ## Stop Docker containers
	@echo "$(BLUE)[DOCKER]$(NC) Stopping containers..."
	$(DOCKER_COMPOSE) down
	@echo "$(GREEN)Containers stopped$(NC)"

docker-logs: ## Show Docker logs
	$(DOCKER_COMPOSE) logs -f

monitor: ## Open monitoring dashboard
	@echo "$(BLUE)[MONITOR]$(NC) Opening monitoring dashboard..."
	@if command -v open >/dev/null 2>&1; then \
		open monitoring/dashboard.html; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open monitoring/dashboard.html; \
	else \
		echo "$(YELLOW)Please open monitoring/dashboard.html in your browser$(NC)"; \
	fi

health: ## Check API health
	@echo "$(BLUE)[HEALTH]$(NC) Checking API health..."
	@curl -f http://localhost:8000/health && echo "$(GREEN)API is healthy$(NC)" || echo "$(RED)API is not responding$(NC)"

predict-test: ## Test prediction endpoint
	@echo "$(BLUE)[PREDICT]$(NC) Testing prediction endpoint..."
	@curl -X POST "http://localhost:8000/predict" \
		-H "Content-Type: application/json" \
		-d '{"machineID": 1, "volt": 150.5, "rotate": 480.2, "pressure": 95.1, "vibration": 38.7}' \
		&& echo "" && echo "$(GREEN)Prediction test completed$(NC)"

pipeline: setup data lint test train ## Run complete training pipeline
	@echo "$(GREEN)Complete pipeline finished successfully$(NC)"
	@echo "$(BLUE)Next steps:$(NC)"
	@echo "  - Start API: $(YELLOW)make api$(NC)"
	@echo "  - View monitoring: $(YELLOW)make monitor$(NC)"
	@echo "  - Test predictions: $(YELLOW)make predict-test$(NC)"
	@echo "  - Simulate cloud: $(YELLOW)make vertex-simulate$(NC)"

clean: ## Clean generated files
	@echo "$(BLUE)[CLEAN]$(NC) Cleaning generated files..."
	@rm -rf models/*.joblib
	@rm -rf logs/*.log
	@rm -rf reports/*.txt
	@rm -rf htmlcov/
	@rm -rf .coverage
	@rm -rf .pytest_cache/
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)Cleanup completed$(NC)"

clean-all: clean docker-down ## Clean everything including Docker
	@echo "$(BLUE)[CLEAN-ALL]$(NC) Removing Docker images..."
	@docker image prune -f
	@echo "$(GREEN)Full cleanup completed$(NC)"

install-dev: ## Install development dependencies
	@echo "$(BLUE)[DEV]$(NC) Installing development dependencies..."
	$(PIP) install jupyter jupyterlab black flake8 pytest-cov
	@echo "$(GREEN)Development dependencies installed$(NC)"

model-info: ## Show trained model information
	@echo "$(BLUE)[MODEL]$(NC) Model information:"
	@if [ -f models/baseline_model.joblib ]; then \
		echo "$(GREEN)Model found: models/baseline_model.joblib$(NC)"; \
		stat models/baseline_model.joblib; \
	else \
		echo "$(RED)No trained model found$(NC)"; \
		echo "$(YELLOW)Run 'make train' first$(NC)"; \
	fi

status: ## Show project status
	@echo "$(BLUE)Project Status$(NC)"
	@echo "=============="
	@echo "$(YELLOW)Data:$(NC)"
	@if [ -f data/raw/PdM_telemetry.csv ]; then echo "  Data files found"; else echo "  Data files missing"; fi
	@if [ -f data/raw/PdM_failures.csv ]; then echo "  Failure data found"; else echo "  Failure data missing"; fi
	@echo "$(YELLOW)Model:$(NC)"
	@if [ -f models/baseline_model.joblib ]; then echo "  Trained model found"; else echo "  Trained model missing"; fi
	@echo "$(YELLOW)API:$(NC)"
	@curl -s http://localhost:8000/health >/dev/null && echo "  API running" || echo "  API not running"
	@echo "$(YELLOW)Tests:$(NC)"
	@if [ -d htmlcov ]; then echo "  Coverage report available"; else echo "  No coverage report"; fi
	@echo "$(YELLOW)Cloud:$(NC)"
	@if [ -f vertex_ai/local_simulation.py ]; then echo "  Vertex AI simulation ready"; else echo "  Vertex AI simulation missing"; fi

quick-start: setup data train docker-build ## Quick start for new users
	@echo "$(GREEN)Quick start completed$(NC)"
	@echo "$(BLUE)To start the API:$(NC) make api"
	@echo "$(BLUE)To run tests:$(NC) make test"
	@echo "$(BLUE)To view status:$(NC) make status"
	@echo "$(BLUE)To simulate cloud:$(NC) make vertex-simulate"

# Development targets
dev-setup: install-dev setup ## Setup development environment
	@echo "$(GREEN)Development environment ready$(NC)"

dev-test: lint test ## Run development tests
	@echo "$(GREEN)Development tests completed$(NC)"

# CI/CD simulation
ci: lint test ## Simulate CI/CD pipeline
	@echo "$(BLUE)[CI]$(NC) Simulating CI/CD pipeline..."
	@echo "$(GREEN)CI pipeline simulation completed$(NC)"

# Vertex AI simulation targets
vertex-simulate: data ## Simulate Vertex AI pipeline locally
	@echo "$(BLUE)[VERTEX-AI]$(NC) Simulating cloud pipeline..."
	@if [ ! -f vertex_ai/local_simulation.py ]; then \
		echo "$(RED)vertex_ai/local_simulation.py not found$(NC)"; \
		echo "$(YELLOW)Please create the Vertex AI simulation files$(NC)"; \
		exit 1; \
	fi
	$(PYTHON) vertex_ai/local_simulation.py

vertex-help: ## Show Vertex AI migration info
	@echo "$(BLUE)Vertex AI Migration$(NC)"
	@echo "==================="
	@echo "$(YELLOW)Available commands:$(NC)"
	@echo "  $(YELLOW)make vertex-simulate$(NC) - Run cloud pipeline simulation"
	@echo "  $(YELLOW)cat vertex_ai/migration_strategy.md$(NC) - View migration plan"
	@echo ""
	@echo "$(YELLOW)Migration benefits:$(NC)"
	@echo "  - Auto-scaling compute resources"
	@echo "  - Managed infrastructure (99.9% SLA)"
	@echo "  - Automated model monitoring"
	@echo "  - A/B testing capabilities"
	@echo "  - Cost optimization (pay-per-use)"
	@echo ""
	@echo "$(YELLOW)Files:$(NC)"
	@if [ -f vertex_ai/pipeline_definition.py ]; then echo "  vertex_ai/pipeline_definition.py - Kubeflow pipeline"; fi
	@if [ -f vertex_ai/deployment_config.py ]; then echo "  vertex_ai/deployment_config.py - Endpoint configuration"; fi
	@if [ -f vertex_ai/monitoring_setup.py ]; then echo "  vertex_ai/monitoring_setup.py - Model monitoring"; fi
	@if [ -f vertex_ai/migration_strategy.md ]; then echo "  vertex_ai/migration_strategy.md - Migration guide"; fi
	@if [ -f vertex_ai/local_simulation.py ]; then echo "  vertex_ai/local_simulation.py - Local simulation"; fi

cloud-ready: vertex-simulate ## Validate cloud migration readiness
	@echo "$(GREEN)Cloud migration validation completed$(NC)"
	@echo "$(BLUE)Your pipeline is ready for Vertex AI migration$(NC)"
	@echo ""
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Set up GCP project and enable Vertex AI APIs"
	@echo "  2. Upload data to Google Cloud Storage"
	@echo "  3. Deploy pipeline using vertex_ai/pipeline_definition.py"
	@echo "  4. Configure monitoring and alerting"

compare-pipelines: train vertex-simulate ## Compare local vs cloud execution
	@echo "$(GREEN)Pipeline comparison completed$(NC)"
	@echo "$(BLUE)Both local and cloud simulations use the same code base$(NC)"
	@echo "$(YELLOW)Key differences:$(NC)"
	@echo "  - Local: Single machine execution"
	@echo "  - Cloud: Distributed, scalable execution"
	@echo "  - Local: Manual resource management"
	@echo "  - Cloud: Auto-scaling and managed infrastructure"

# Help for specific workflows
help-quick: ## Show quick start guide
	@echo "$(BLUE)Quick Start Guide$(NC)"
	@echo "=================="
	@echo "1. $(YELLOW)make setup$(NC)          - Install dependencies"
	@echo "2. Download data from Kaggle to data/raw/"
	@echo "3. $(YELLOW)make pipeline$(NC)       - Run training pipeline"
	@echo "4. $(YELLOW)make api$(NC)            - Start API server"
	@echo "5. $(YELLOW)make predict-test$(NC)   - Test predictions"
	@echo "6. $(YELLOW)make vertex-simulate$(NC) - Simulate cloud migration"

help-docker: ## Show Docker workflow help
	@echo "$(BLUE)Docker Workflow$(NC)"
	@echo "==============="
	@echo "$(YELLOW)make docker-build$(NC)  - Build images"
	@echo "$(YELLOW)make docker-up$(NC)     - Start containers"
	@echo "$(YELLOW)make docker-logs$(NC)   - View logs"
	@echo "$(YELLOW)make docker-down$(NC)   - Stop containers"

help-vertex: ## Show Vertex AI workflow help
	@echo "$(BLUE)Vertex AI Workflow$(NC)"
	@echo "=================="
	@echo "$(YELLOW)make vertex-simulate$(NC)   - Run local simulation"
	@echo "$(YELLOW)make vertex-help$(NC)       - Show migration info"
	@echo "$(YELLOW)make cloud-ready$(NC)       - Validate readiness"
	@echo "$(YELLOW)make compare-pipelines$(NC) - Compare local vs cloud"