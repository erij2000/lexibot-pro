# 🤖 Lexibot Pro - Assistant Juridique Tunisien

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.5-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Ollama](https://img.shields.io/badge/Ollama-phi3:mini-000000?logo=ollama)](https://ollama.ai/)
[![Rasa](https://img.shields.io/badge/Rasa-3.6.20-5A17EE?logo=rasa)](https://rasa.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791?logo=postgresql)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Lexibot Pro** est un assistant juridique intelligent spécialisé en droit tunisien, propulsé par l'IA (Ollama + Rasa) avec un backend FastAPI sécurisé.

---

## ✨ Fonctionnalités

- 🤖 **Chatbot IA** - Réponses juridiques basées sur Ollama (phi3:mini) et Rasa
- 🔐 **Authentification JWT** - Système de connexion sécurisé avec FastAPI-Users
- 👥 **Gestion des Utilisateurs** - Rôles (Client, Premium, Avocat, Admin) et permissions
- 📅 **Système de Rendez-vous** - Demande et gestion des RDV avec l'avocate
- 💬 **Support Multilingue** - Français, Anglais, Arabe
- 📊 **Base de Données PostgreSQL** - Stockage des conversations et RDV
- 🎨 **Interface Streamlit** - Frontend moderne et réactif

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   FRONTEND (Streamlit)              │
└────────────────────┬────────────────────────────────┘
                     │ HTTP/REST
┌────────────────────▼────────────────────────────────┐
│              BACKEND (FastAPI)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │   Routers    │  │   Services   │  │   Models  │ │
│  │  (Endpoints) │  │   (Logique)  │  │   (DB)    │ │
│  └──────────────┘  └──────────────┘  └───────────┘ │
└────────────────────┬──────────────┬──────────────────┘
                     │              │
        ┌────────────▼──┐       ┌───▼──────────┐
        │  Ollama (LLM) │       │  Rasa (NLU)  │
        └───────────────┘       └──────────────┘
                     │
        ┌────────────▼────────────┐
        │  PostgreSQL (Database)  │
        └─────────────────────────┘
```

---

## 🚀 Démarrage Rapide

### Prérequis

- Python 3.10+
- Docker & Docker Compose
- Ollama installé
- Make (optionnel mais recommandé)

### Installation Complète

```bash
# 1. Cloner le projet
git clone https://github.com/votre-repo/lexibot-pro.git
cd lexibot-pro

# 2. Configuration automatique
make setup

# 3. Démarrer PostgreSQL
make db-up

# 4. Démarrer Ollama (terminal séparé)
ollama serve
ollama pull phi3:mini

# 5. Démarrer FastAPI (terminal séparé)
make start-api

# 6. Démarrer Rasa (terminal séparé)
make start-rasa
```

**🎉 C'est tout !** L'API est disponible sur http://localhost:8000

---

## 📚 Documentation

- 📖 **API Documentation** : http://localhost:8000/docs
- 📋 **Setup Guide** : [SETUP_GUIDE.md](SETUP_GUIDE.md)
- 🔄 **Migration Guide** : [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
- 🏗️ **Structure** : [STRUCTURE.txt](STRUCTURE.txt)

---

## 🛠️ Commandes Utiles

```bash
# Voir toutes les commandes disponibles
make help

# Vérifier le statut des services
make status

# Voir les logs
make logs-api
make logs-db

# Tests
make test

# Nettoyage
make clean
```

---

## 🔑 Endpoints Principaux

### Authentification
- `POST /auth/register` - Créer un compte
- `POST /auth/jwt/login` - Se connecter
- `GET /users/me` - Profil utilisateur

### Chatbot
- `POST /chatbot/ask` - Poser une question juridique

### Rendez-vous
- `POST /appointments/request` - Demander un RDV
- `GET /admin/appointments/pending` - Voir les RDV en attente (Admin)

---

## 🗂️ Structure du Projet

```
LEXIBOT_PRO/
├── backend/
│   ├── api/                # Routes et services
│   │   ├── routers/       # Endpoints FastAPI
│   │   └── services/      # Logique métier
│   ├── auth/              # Configuration auth
│   ├── core/              # Configuration globale
│   ├── database/          # Modèles et config DB
│   ├── models/            # Modèles SQLAlchemy
│   ├── schemas/           # Schémas Pydantic
│   └── main.py            # Point d'entrée
│
├── rasa/                  # Configuration Rasa
│   ├── data/             # Données d'entraînement
│   ├── models/           # Modèles entraînés
│   └── venv/             # venv Rasa séparé
│
├── venv_ollama_fastapi/  # venv FastAPI/Ollama
├── Makefile              # Commandes rapides
└── README.md             # Ce fichier
```

---

## 🔐 Sécurité

- ✅ **JWT Authentication** - Tokens sécurisés avec expiration
- ✅ **Password Hashing** - Bcrypt pour les mots de passe
- ✅ **CORS** - Configuration stricte des origines
- ✅ **Permissions** - Système de rôles et permissions granulaires
- ✅ **HTTPS Ready** - Préparé pour le déploiement sécurisé

---

## 🌍 Environnements Virtuels Séparés

### venv_ollama_fastapi
- FastAPI, Uvicorn
- SQLAlchemy, AsyncPG
- FastAPI-Users
- Ollama integration

### rasa/venv
- Rasa 3.6.20
- Dépendances NLU isolées

**Pourquoi 2 venv ?** Éviter les conflits de dépendances entre Rasa et FastAPI !

---

## 👥 Rôles et Permissions

| Rôle | Permissions |
|------|-------------|
| **CLIENT** | Chatbot, Demande RDV |
| **PREMIUM** | Chatbot, RDV, Historique |
| **LAWYER** | Gestion RDV, Analytics, Calendrier |
| **ADMIN** | Toutes les permissions |

---

## 🧪 Tests

```bash
# Test de l'API
curl http://localhost:8000/

# Test de création d'utilisateur
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@test.com",
    "password": "Test123!",
    "first_name": "Test",
    "last_name": "User"
  }'

# Test de connexion
curl -X POST http://localhost:8000/auth/jwt/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=test@test.com&password=Test123!"
```

---

## 🤝 Contribution

Les contributions sont les bienvenues ! Merci de :

1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

---

## 📝 License

Ce projet est sous licence MIT. Voir [LICENSE](LICENSE) pour plus de détails.

---

## 👤 Auteur

**Lexibot Pro Team**

- Cabinet : Maître Hila Ben Arbia
- 📍 Sousse, Tunisie
- 📞 +216 96 762 574

---

## 🙏 Remerciements

- [FastAPI](https://fastapi.tiangolo.com/) - Framework web moderne
- [Ollama](https://ollama.ai/) - LLM local
- [Rasa](https://rasa.com/) - NLU open-source
- [PostgreSQL](https://www.postgresql.org/) - Base de données robuste

---

**Made with ❤️ in Tunisia 🇹🇳**