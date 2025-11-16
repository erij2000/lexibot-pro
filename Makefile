# Makefile - Commandes rapides pour Lexibot Pro

.PHONY: help setup install-fastapi install-rasa db-up db-down start-api start-rasa start-ollama test clean

# Couleurs pour les messages
GREEN=\033[0;32m
YELLOW=\033[1;33m
NC=\033[0m # No Color

help: ## Afficher l'aide
	@echo "${GREEN}📋 Commandes disponibles :${NC}"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  ${YELLOW}%-20s${NC} %s\n", $$1, $$2}'

# ==============================================
# SETUP & INSTALLATION
# ==============================================

setup: ## Configuration initiale complète
	@echo "${GREEN}🚀 Configuration de Lexibot Pro...${NC}"
	@make install-fastapi
	@make install-rasa
	@make db-up
	@echo "${GREEN}✅ Configuration terminée !${NC}"

install-fastapi: ## Installer les dépendances FastAPI
	@echo "${GREEN}📦 Installation des dépendances FastAPI...${NC}"
	python3 -m venv venv_ollama_fastapi
	./venv_ollama_fastapi/bin/pip install --upgrade pip
	./venv_ollama_fastapi/bin/pip install -r backend/requirements_fastapi.txt
	@echo "${GREEN}✅ Dépendances FastAPI installées !${NC}"

install-rasa: ## Installer les dépendances Rasa
	@echo "${GREEN}📦 Installation des dépendances Rasa...${NC}"
	cd rasa && python3 -m venv venv
	cd rasa && ./venv/bin/pip install --upgrade pip
	cd rasa && ./venv/bin/pip install rasa==3.6.20
	@echo "${GREEN}✅ Dépendances Rasa installées !${NC}"

# ==============================================
# DATABASE
# ==============================================

db-up: ## Démarrer PostgreSQL
	@echo "${GREEN}🐘 Démarrage de PostgreSQL...${NC}"
	cd backend && docker-compose up -d
	@echo "${GREEN}✅ PostgreSQL démarré !${NC}"

db-down: ## Arrêter PostgreSQL
	@echo "${YELLOW}🛑 Arrêt de PostgreSQL...${NC}"
	cd backend && docker-compose down

db-reset: ## Réinitialiser la base de données
	@echo "${YELLOW}⚠️  Réinitialisation de la base de données...${NC}"
	cd backend && docker-compose down -v
	cd backend && docker-compose up -d
	@echo "${GREEN}✅ Base de données réinitialisée !${NC}"

# ==============================================
# SERVICES
# ==============================================

start-ollama: ## Démarrer Ollama (manuel)
	@echo "${GREEN}🤖 Démarrage d'Ollama...${NC}"
	@echo "${YELLOW}⚠️  Exécutez 'ollama serve' dans un terminal séparé${NC}"

start-api: ## Démarrer FastAPI
	@echo "${GREEN}🚀 Démarrage de FastAPI...${NC}"
	cd backend/api && ../../venv_ollama_fastapi/bin/python main.py

start-rasa: ## Démarrer Rasa
	@echo "${GREEN}💬 Démarrage de Rasa...${NC}"
	cd rasa && ./venv/bin/rasa run --enable-api --cors "*"

train-rasa: ## Entraîner le modèle Rasa
	@echo "${GREEN}🎓 Entraînement du modèle Rasa...${NC}"
	cd rasa && ./venv/bin/rasa train

# ==============================================
# DÉVELOPPEMENT
# ==============================================

dev: ## Démarrer en mode développement (tout en même temps)
	@echo "${GREEN}🔥 Mode développement${NC}"
	@echo "${YELLOW}Démarrez les services dans l'ordre :${NC}"
	@echo "  1️⃣  make db-up"
	@echo "  2️⃣  ollama serve (terminal séparé)"
	@echo "  3️⃣  make start-api (terminal séparé)"
	@echo "  4️⃣  make start-rasa (terminal séparé)"

test: ## Lancer les tests
	@echo "${GREEN}🧪 Tests de l'API...${NC}"
	curl -f http://localhost:8000/ || echo "${YELLOW}⚠️  API non accessible${NC}"
	curl -f http://localhost:5005/ || echo "${YELLOW}⚠️  Rasa non accessible${NC}"
	curl -f http://localhost:11434/api/version || echo "${YELLOW}⚠️  Ollama non accessible${NC}"

logs-db: ## Voir les logs PostgreSQL
	cd backend && docker-compose logs -f postgres

logs-api: ## Voir les logs API
	tail -f backend/ollama_logs.txt

# ==============================================
# NETTOYAGE
# ==============================================

clean: ## Nettoyer les fichiers temporaires
	@echo "${YELLOW}🧹 Nettoyage...${NC}"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "${GREEN}✅ Nettoyage terminé !${NC}"

clean-all: clean db-down ## Nettoyage complet (DB + fichiers)
	@echo "${YELLOW}🧹 Nettoyage complet...${NC}"
	rm -rf venv_ollama_fastapi
	rm -rf rasa/venv
	@echo "${GREEN}✅ Nettoyage complet terminé !${NC}"

# ==============================================
# UTILITIES
# ==============================================

status: ## Vérifier le statut des services
	@echo "${GREEN}📊 Statut des services :${NC}"
	@echo ""
	@echo "${YELLOW}PostgreSQL :${NC}"
	@docker ps | grep lexibot_postgres || echo "  ❌ Non démarré"
	@echo ""
	@echo "${YELLOW}Ollama :${NC}"
	@curl -s http://localhost:11434/api/version > /dev/null && echo "  ✅ En cours d'exécution" || echo "  ❌ Non démarré"
	@echo ""
	@echo "${YELLOW}FastAPI :${NC}"
	@curl -s http://localhost:8000/ > /dev/null && echo "  ✅ En cours d'exécution" || echo "  ❌ Non démarré"
	@echo ""
	@echo "${YELLOW}Rasa :${NC}"
	@curl -s http://localhost:5005/ > /dev/null && echo "  ✅ En cours d'exécution" || echo "  ❌ Non démarré"

venv-info: ## Afficher les informations des venv
	@echo "${GREEN}📦 Informations des environnements virtuels :${NC}"
	@echo ""
	@echo "${YELLOW}venv_ollama_fastapi :${NC}"
	@if [ -d "venv_ollama_fastapi" ]; then \
		./venv_ollama_fastapi/bin/python --version; \
		./venv_ollama_fastapi/bin/pip list | grep -E "fastapi|uvicorn|sqlalchemy"; \
	else \
		echo "  ❌ Non créé"; \
	fi
	@echo ""
	@echo "${YELLOW}rasa/venv :${NC}"
	@if [ -d "rasa/venv" ]; then \
		cd rasa && ./venv/bin/python --version; \
		cd rasa && ./venv/bin/pip list | grep rasa; \
	else \
		echo "  ❌ Non créé"; \
	fi

# ==============================================
# PRODUCTION
# ==============================================

build: ## Build pour production (TODO)
	@echo "${YELLOW}🏗️  Build production (à implémenter)${NC}"

deploy: ## Déployer (TODO)
	@echo "${YELLOW}🚀 Déploiement (à implémenter)${NC}"