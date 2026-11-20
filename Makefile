.PHONY: up down logs test clean

up:
	@if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then \
		docker compose up --build -d; \
	else \
		echo "Docker daemon not running; launching local services..."; \
		mkdir -p /tmp/app_data /tmp/app_cache; \
		pkill -f "uvicorn.*8000" 2>/dev/null || true; \
		pkill -f "vite.*3000" 2>/dev/null || true; \
		(cd backend && DATABASE_URL="sqlite:////tmp/app_data/app.db" \
			LLM_API_KEYS="mock_key_1,mock_key_2" \
			ALLOWED_ORIGINS="http://localhost:3000,http://localhost:5173" \
			ENABLE_FAILURE_SIMULATION=true \
			LLM_MODEL="gemini/gemini-3.5-flash-lite" \
			DEFAULT_SYSTEM_PROMPT="You are a helpful assistant." \
			CACHE_DIR="/tmp/app_cache" \
			MOCK_LLM=true \
			.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 > /tmp/backend.log 2>&1 & ); \
		(cd frontend && VITE_API_URL="http://localhost:8000" \
			npx vite preview --port 3000 --host 0.0.0.0 > /tmp/frontend.log 2>&1 & ); \
		sleep 3; \
	fi
	@echo ""
	@echo "=========================================="
	@echo "Services started successfully!"
	@echo "Frontend:  http://localhost:3000"
	@echo "Backend:   http://localhost:8000"
	@echo "Health:    http://localhost:8000/health"
	@echo "=========================================="

down:
	@if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then \
		docker compose down; \
	else \
		echo "Stopping services (preserving database volume)..."; \
		pkill -f "uvicorn.*8000" 2>/dev/null || true; \
		pkill -f "vite.*3000" 2>/dev/null || true; \
		sleep 1; \
		echo "Services stopped."; \
	fi

logs:
	@if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then \
		docker compose logs -f; \
	else \
		tail -n 50 -f /tmp/backend.log /tmp/frontend.log; \
	fi

test:
	cd backend && PYTHONPATH=. .venv/bin/pytest tests/test_api.py -v

clean:
	@if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then \
		docker compose down -v; \
	else \
		echo "Cleaning volume and cache..."; \
		rm -rf /tmp/app_data /tmp/app_cache; \
	fi
