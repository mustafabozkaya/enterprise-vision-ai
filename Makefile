# =============================================================================
# Enterprise Vision AI - Makefile
# Common development commands
# =============================================================================

# Colors
BOLD := \033[1m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
RED := \033[0;31m
RESET := \033[0m

# Project settings
PROJECT_NAME = enterprise-vision-ai
PYTHON = python
PIP = pip
VENV = venv

# Default target
.PHONY: help
help:
	@echo ""
	@echo "$(BOLD)$(BLUE)Enterprise Vision AI - Geliştirme Komutları$(RESET)"
	@echo "$(BOLD)===============================================$(RESET)"
	@echo ""
	@echo "$(GREEN)Kurulum | Setup:$(RESET)"
	@echo "  make create-venv      - Sanal ortam oluştur"
	@echo "  make install          - Bağımlılıkları yükle"
	@echo "  make install-dev      - Geliştirme bağımlılıklarını yükle"
	@echo ""
	@echo "$(GREEN)Geliştirme | Development:$(RESET)"
	@echo "  make run              - Uygulamayı çalıştır"
	@echo "  make run-debug        - Debug modunda çalıştır"
	@echo ""
	@echo "$(GREEN)Docker:$(RESET)"
	@echo "  make docker-build     - Docker image oluştur"
	@echo "  make docker-up        - Docker Compose ile başlat"
	@echo "  make docker-down      - Docker Compose ile durdur"
	@echo "  make docker-logs      - Docker logs takip et"
	@echo "  make docker-clean     - Docker container ve image temizle"
	@echo ""
	@echo "$(GREEN)Test:$(RESET)"
	@echo "  make test             - Testleri çalıştır"
	@echo "  make test-cov        - Coverage ile testleri çalıştır"
	@echo "  make test-unit        - Birim testleri çalıştır"
	@echo "  make test-integration - Entegrasyon testleri çalıştır"
	@echo ""
	@echo "$(GREEN)Kod Kalitesi | Code Quality:$(RESET)"
	@echo "  make lint             - Linting kontrolü"
	@echo "  make format          - Kod formatla"
	@echo "  make sort-imports   - Import'ları düzenle"
	@echo "  make type-check      - Tip kontrolü"
	@echo "  make pre-commit      - Pre-commit hook'larını çalıştır"
	@echo ""
	@echo "$(GREEN)Temizlik | Cleanup:$(RESET)"
	@echo "  make clean            - Derleme dosyalarını temizle"
	@echo "  make clean-cache     - Cache dosyalarını temizle"
	@echo "  make clean-all       - Her şeyi temizle"
	@echo ""
	@echo "$(GREEN)Yardımcı | Utility:$(RESET)"
	@echo "  make requirements    - requirements.txt oluştur"
	@echo "  make check-deps      - Bağımlılıkları kontrol et"
	@echo ""

# -----------------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------------

.PHONY: create-venv
create-venv:
	@echo "$(YELLOW)Sanal ortam oluşturuluyor...$(RESET)"
	$(PYTHON) -m venv $(VENV)
	@echo "$(GREEN)Sanal ortam oluşturuldu! Aktifleştirmek için:$(RESET)"
	@echo "  source $(VENV)/bin/activate  (Linux/Mac)"
	@echo "  $(VENV)\\Scripts\\activate     (Windows)"

.PHONY: install
install:
	@echo "$(YELLOW)Bağımlılıklar yükleniyor...$(RESET)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)Bağımlılıklar yüklendi!$(RESET)"

.PHONY: install-dev
install-dev: install
	@echo "$(YELLOW)Geliştirme bağımlılıkları yükleniyor...$(RESET)"
	$(PIP) install -e ".[dev]"
	@echo "$(GREEN)Geliştirme bağımlılıkları yüklendi!$(RESET)"

# -----------------------------------------------------------------------------
# Development
# -----------------------------------------------------------------------------

.PHONY: run
run:
	@echo "$(YELLOW)Uygulama başlatılıyor...$(RESET)"
	streamlit run app.py

.PHONY: run-debug
run-debug:
	@echo "$(YELLOW)Uygulama debug modunda başlatılıyor...$(RESET)"
	streamlit run app.py --logger.level=debug

# -----------------------------------------------------------------------------
# Docker
# -----------------------------------------------------------------------------

.PHONY: docker-build
docker-build:
	@echo "$(YELLOW)Docker image oluşturuluyor...$(RESET)"
	docker build -t $(PROJECT_NAME):latest .

.PHONY: docker-up
docker-up:
	@echo "$(YELLOW)Docker Compose başlatılıyor...$(RESET)"
	docker-compose up -d
	@echo "$(GREEN)Uygulama http://localhost:8501 adresinde çalışıyor!$(RESET)"

.PHONY: docker-down
docker-down:
	@echo "$(YELLOW)Docker Compose durduruluyor...$(RESET)"
	docker-compose down

.PHONY: docker-logs
docker-logs:
	docker-compose logs -f

.PHONY: docker-clean
docker-clean:
	@echo "$(YELLOW)Docker container ve image temizleniyor...$(RESET)"
	docker-compose down -v --rmi local
	@echo "$(GREEN)Docker temizlendi!$(RESET)"

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------

.PHONY: test
test:
	@echo "$(YELLOW)Testler çalıştırılıyor...$(RESET)"
	pytest -v

.PHONY: test-cov
test-cov:
	@echo "$(YELLOW)Coverage ile testler çalıştırılıyor...$(RESET)"
	pytest --cov=. --cov-report=html --cov-report=term

.PHONY: test-unit
test-unit:
	@echo "$(YELLOW)Birim testleri çalıştırılıyor...$(RESET)"
	pytest tests/unit -v

.PHONY: test-integration
test-integration:
	@echo "$(YELLOW)Entegrasyon testleri çalıştırılıyor...$(RESET)"
	pytest tests/integration -v

# -----------------------------------------------------------------------------
# Code Quality
# -----------------------------------------------------------------------------

.PHONY: lint
lint:
	@echo "$(YELLOW)Linting kontrolü yapılıyor...$(RESET)"
	flake8 . --max-line-length=100 --ignore=E501,W503

.PHONY: format
format:
	@echo "$(YELLOW)Kod formatlanıyor...$(RESET)"
	black .

.PHONY: sort-imports
sort-imports:
	@echo "$(YELLOW)Import'lar düzenleniyor...$(RESET)"
	isort .

.PHONY: type-check
type-check:
	@echo "$(YELLOW)Tip kontrolü yapılıyor...$(RESET)"
	mypy . --ignore-missing-imports

.PHONY: pre-commit
pre-commit:
	@echo "$(YELLOW)Pre-commit hook'ları çalıştırılıyor...$(RESET)"
	pre-commit run --all-files

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------

.PHONY: clean
clean:
	@echo "$(YELLOW)Derleme dosyaları temizleniyor...$(RESET)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf build/ dist/ *.egg-info/ .eggs/
	@echo "$(GREEN)Temizlik tamamlandı!$(RESET)"

.PHONY: clean-cache
clean-cache:
	@echo "$(YELLOW)Cache temizleniyor...$(RESET)"
	rm -rf .streamlit/cache/ .hypothesis/ htmlcov/
	@echo "$(GREEN)Cache temizlendi!$(RESET)"

.PHONY: clean-all
clean-all: clean clean-cache
	@echo "$(YELLOW)Docker temizleniyor...$(RESET)"
	docker-compose down -v --rmi local 2>/dev/null || true
	@echo "$(GREEN)Her şey temizlendi!$(RESET)"

# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------

.PHONY: requirements
requirements:
	@echo "$(YELLOW)requirements.txt oluşturuluyor...$(RESET)"
	pip freeze > requirements.txt
	@echo "$(GREEN)requirements.txt oluşturuldu!$(RESET)"

.PHONY: check-deps
check-deps:
	@echo "$(YELLOW)Bağımlılıklar kontrol ediliyor...$(RESET)"
	@$(PIP) list --outdated || true

# -----------------------------------------------------------------------------
# Docker Production
# -----------------------------------------------------------------------------

.PHONY: docker-prod-build
docker-prod-build:
	@echo "$(YELLOW)Production Docker image oluşturuluyor...$(RESET)"
	docker build --target production -t $(PROJECT_NAME):prod .
	@echo "$(GREEN)Production image oluşturuldu!$(RESET)"

.PHONY: docker-prod-up
docker-prod-up:
	@echo "$(YELLOW)Production Docker başlatılıyor...$(RESET)"
	docker run -d -p 8501:8501 --name $(PROJECT_NAME)-prod $(PROJECT_NAME):prod
	@echo "$(GREEN)Production çalışıyor!$(RESET)"
