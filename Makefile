.PHONY: up down restart logs ps migrate rollback api worker test test-all test-infra check lint clean setup

# ── Docker Compose ──
up:
	docker compose up -d

down:
	docker compose down

restart:
	docker compose down && docker compose up -d

logs:
	docker compose logs -f

ps:
	docker compose ps

# ── 数据库 ──
migrate:
	alembic upgrade head

rollback:
	alembic downgrade -1

# ── 运行服务 ──
api:
	uvicorn apps.modelops_api.main:app --host 0.0.0.0 --port 8000 --reload

worker:
	celery -A workers.app worker --loglevel=info --pool=solo

# ── 测试 ──
test:
	pytest tests/unit tests/integration tests/test_openapi.py -q

test-all:
	pytest tests/ -q

test-infra:
	RUN_INFRA_TESTS=true pytest tests/integration -q

# ── 契约与质量检查（pre-commit 同款） ──
check:
	@echo "[1/3] 编译检查..."
	python -m compileall -q apps workers packages migrations
	@echo "[2/3] OpenAPI 产物一致性..."
	python scripts/generate_openapi.py --check
	@echo "[3/3] 单元测试 + OpenAPI 测试..."
	python -m pytest tests/unit tests/test_openapi.py -q
	@echo ""
	@echo "全部通过"

lint:
	python -m compileall -q apps workers packages migrations

# ── 首次设置 ──
setup:
	git config core.hooksPath .githooks
	@echo "Git hooks 已配置: .githooks/pre-commit"
	@echo "每次 git commit 时自动运行 make check"

# ── 清理 ──
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -type f -name "*.pyc" -delete 2>/dev/null; \
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null
