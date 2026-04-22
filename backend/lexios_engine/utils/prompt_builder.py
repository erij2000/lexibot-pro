"""
utils/prompt_builder.py — Constructeur de Prompts Juridiques Ultimate v4
========================================================================
Système de templates modulaires pour Lexibot.
Supporte : multi-mode (Pénal/Civil/Commercial/Admin), multi-format (détaillé/synthétique/JSON),
et injection dynamique d'entités/contexte.

Cohérence : Utilise config.py pour les paramètres globaux.
"""

from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass, field
from enum import Enum
import json

from config import settings


class LegalMode(Enum):
    PENAL = "penal"
    CIVIL = "civil"
    COMMERCIAL = "commercial"
    ADMINISTRATIF = "administratif"
    CONSTITUTIONNEL = "constitutionnel"
    GENERAL = "general"


class ResponseFormat(Enum):
    DETAILED = "detailed"      # Réponse structurée complète
    CONCISE = "concise"        # Réponse courte (chat rapide)
    JSON_STRUCTURED = "json"   # Format machine-readable
    STEP_BY_STEP = "steps"     # Analyse étape par étape
    COMPARATIVE = "compare"    # Tableau comparatif


class CitationStyle(Enum):
    STRICT = "strict"          # Cite uniquement si texte exact trouvé
    PARAPHRASE = "paraphrase"  # Reformulation autorisée
    NONE = "none"              # Pas de citation requise


@dataclass
class PromptConfig:
    """Configuration flexible pour la construction de prompts."""
    language: Literal["fr", "ar", "en"] = "fr"
    legal_mode: LegalMode = LegalMode.GENERAL
    response_format: ResponseFormat = ResponseFormat.DETAILED
    citation_style: CitationStyle = CitationStyle.STRICT
    max_length_words: int = 500
    include_entities: bool = True
    include_warnings: bool = True
    include_sources: bool = True
    confidence_level: float = 0.8  # Pour ajuster le ton (0.5=prudent, 0.9=affirmatif)
    
    # Spécifique au mode
    focus_areas: List[str] = field(default_factory=list)  # Ex: ["délais", "compétence", "appel"]
    target_audience: str = "professionnel"  # ou "grand_public", "étudiant"


class LegalPromptBuilder:
    """
    Constructeur professionnel de prompts juridiques.
    Pattern: Template Method + Strategy
    """
    
    # ── TEMPLATES PAR MODE JURIDIQUE ─────────────────────────────────────────
    
    MODE_PROFILES = {
        LegalMode.PENAL: {
            "fr": {
                "role": "Expert en droit pénal tunisien (Code pénal tunisien, CPP, jurisprudence Cour de Cassation)",
                "focus": "infractions, peines, procédure pénale, garde à vue, comparution, appel, cassation",
                "key_articles": ["Code pénal", "CPP", "Loi 2005-75 lutte contre terrorisme"],
                "tone": "formel, précis, vigilant sur les droits de la défense"
            },
            "ar": {
                "role": "خبير في القانون الجزائي التونسي (مجلة الجزاء، قانون الإجراءات الجزائية)",
                "focus": "الجرائم، العقوبات، مسطرة جنائية، الحراسة النظرية، الاستئناف، النقض",
                "key_articles": ["مجلة الجزاء", "قانون الإجراءات الجزائية"],
                "tone": "رسمي، دقيق، يحمي حقوق الدفاع"
            }
        },
        LegalMode.CIVIL: {
            "fr": {
                "role": "Expert en droit civil tunisien (Code des obligations et contrats, Code civil, droit de la famille)",
                "focus": "contrats, obligations, responsabilité civile, preuve, prescription, mariage, divorce, succession",
                "key_articles": ["COC", "Code civil", "CSP (statut personnel)"],
                "tone": "clair, pédagogique, précis sur les droits subjectifs"
            },
            "ar": {
                "role": "خبير في القانون المدني التونسي (مجلة الالتزامات والعقود، القانون المدني)",
                "focus": "العقود، الالتزامات، المسؤولية المدنية، الإثبات، التقادم، الأحوال الشخصية",
                "key_articles": ["مجلة الالتزامات والعقود", "القانون المدني", "مجلة الأحوال الشخصية"],
                "tone": "واضح، تربوي، دقيق"
            }
        },
        LegalMode.COMMERCIAL: {
            "fr": {
                "role": "Expert en droit commercial tunisien (Code de commerce, lois relatives aux sociétés commerciales)",
                "focus": "sociétés, fonds de commerce, effets de commerce, banqueroute, procédures collectives",
                "key_articles": ["Code de commerce", "Loi 2001-117 (sociétés commerciales)"],
                "tone": "pragmatique, business-oriented, précis"
            },
            "ar": {
                "role": "خبير في القانون التجاري التونسي (مجلة التجارة، قوانين الشركات)",
                "focus": "الشركات، رأس المال، الكمبيالات، الإفلاس",
                "key_articles": ["مجلة التجارة", "القانون 2001-117"],
                "tone": "عملي، دقيق"
            }
        },
        LegalMode.ADMINISTRATIF: {
            "fr": {
                "role": "Expert en droit administratif tunisien (jurisprudence du Tribunal administratif, contentieux)",
                "focus": "contrats administratifs, responsabilité de l'administration, recours, excès de pouvoir",
                "key_articles": ["Code juridictions administratives", "jurisprudence TA"],
                "tone": "formel, respectueux du principe de légalité"
            },
            "ar": {
                "role": "خبير في القانون الإداري التونسي",
                "focus": "العقود الإدارية، مسؤولية الإدارة، التعويض، الطعون",
                "key_articles": ["مجلة القضاء الإداري"],
                "tone": "رسمي، يحترم مبدأ المشروعية"
            }
        },
        LegalMode.CONSTITUTIONNEL: {
            "fr": {
                "role": "Constitutionnaliste tunisien (Constitution 2022, droits fondamentaux, contrôle constitutionnalité)",
                "focus": "droits et libertés, pouvoirs publics, bloc de constitutionnalité",
                "key_articles": ["Constitution 2022", "décrets lois transition"],
                "tone": "sérieux, fondé sur les principes supra-législatifs"
            },
            "ar": {
                "role": "خبير دستوري تونسي (دستور 2022)",
                "focus": "الحقوق والحريات، السلطات العامة",
                "tone": "جدي، مبني على المبادئ الدستورية"
            }
        },
        LegalMode.GENERAL: {
            "fr": {
                "role": "Lexibot, assistant juridique généraliste spécialisé en droit tunisien",
                "focus": "droit tunisien global, orientation vers spécialiste si nécessaire",
                "key_articles": ["sources générales du droit tunisien"],
                "tone": "professionnel, honnête sur les limites"
            },
            "ar": {
                "role": "Lexibot، المساعد القانوني العام في القانون التونسي",
                "focus": "القانون التونسي العام، التوجيه نحو الاختصاص",
                "key_articles": ["مصادر القانون التونسي"],
                "tone": "مهني، صادق حول الحدود"
            }
        }
    }
    
    # ── TEMPLATES DE FORMAT ─────────────────────────────────────────────────
    
    FORMAT_TEMPLATES = {
        ResponseFormat.DETAILED: {
            "fr": """
RÉPONSE ATTENDUE (Format Détaillé) :
1. **Résumé en une phrase** : Réponse directe à la question
2. **Cadre juridique** : Articles et textes applicables cités précisément
3. **Analyse détaillée** : Application au cas présenté avec nuances
4. **Solution/Conclusion** : Recommandations pratiques et prochaines étapes
5. **Attention/Risques** : Pièges juridiques éventuels à éviter

Longueur : {max_words} mots maximum.
""",
            "ar": """
الرد المتوقع (تفصيلي):
1. **الملخص في جملة واحدة**: إجابة مباشرة
2. **الإطار القانوني**: النصوص القانونية المعمول بها
3. **التحليل التفصيلي**: التطبيق على الحالة المعروضة
4. **الحل/الخلاصة**: التوصيات العملية والخطوات المقبلة
5. **تنبيه/مخاطر**: الأخطاء القانونية الواجب تجنبها

الطول: {max_words} كلمة كحد أقصى.
"""
        },
        ResponseFormat.CONCISE: {
            "fr": "RÉPONSE CONCISE (3-5 phrases maximum) : Réponse directe sans développement théorique. Priorité à l'action pratique.",
            "ar": "رد مختصر (3-5 جمل كحد أقصى): إجابة مباشرة بدون تطوير نظري. الأولوية للعمل."
        },
        ResponseFormat.STEP_BY_STEP: {
            "fr": """
PROCÉDURE ÉTAPE PAR ÉTAPE :
Pour chaque étape, indique : [Action] - [Délai] - [Texte applicable] - [Pièce justificative]

1. Première étape immédiate
2. Étapes suivantes chronologiques
3. Recours possibles si échec
""",
            "ar": """
الإجراء خطوة بخطوة:
لكل خطوة: [الإجراء] - [الأجل] - [النص القانوني] - [الوثيقة المثبتة]

1. الخطوة الأولى الفورية
2. الخطوات التالية زمنياً
3. الطعون الممكنة عند الفشل
"""
        },
        ResponseFormat.JSON_STRUCTURED: {
            "fr": """
RÉPONSE EN JSON STRICT (pas de texte avant ou après) :
```json
{{
  "reponse_principale": "string",
  "fondement_legal": ["article X", "article Y"],
  "conditions": ["condition 1", "condition 2"],
  "delai_eventuel": "string ou null",
  "juridiction_competente": "string",
  "risques": ["risque 1"],
  "recommandations": ["action 1"]
}}
```""",
            "ar": """
رد بصيغة JSON صارمة (لا نص قبل أو بعد):
```json
{{
  "الإجابة_الرئيسية": "نص",
  "الأساس_القانوني": ["المادة س", "المادة ص"],
  "الشروط": ["شرط 1"],
  "الأجل": "نص أو null",
  "الجهة_المختصة": "نص",
  "المخاطر": ["خطر 1"],
  "التوصيات": ["إجراء 1"]
}}
```"""
        },
        ResponseFormat.COMPARATIVE: {
            "fr": """
FORMAT COMPARATIF (Tableau) :
| Critère | Option A | Option B | Recommandation |
|---------|----------|----------|----------------|
| [Nom]   | [Desc]   | [Desc]   | [Choix + Pourquoi] |

Puis conclusion synthétique.
""",
            "ar": """
صيغة مقارنة (جدول):
| المعيار | الخيار أ | الخيار ب | التوصية |
|---------|----------|----------|----------|
| [الاسم] | [الوصف] | [الوصف] | [الاختيار + السبب] |

ثم خلاصة مختصرة.
"""
        }
    }
    
    @classmethod
    def build_synthesis_prompt(
        cls,
        language: str,
        context: str,
        legal_mode: str = "general",
        entities: Dict[str, Any] = None,
        config: Optional[PromptConfig] = None,
        sources_count: int = 0
    ) -> str:
        """
        Construit le prompt de synthèse finale ULTRA-FLEXIBLE.
        
        Args:
            language: 'fr', 'ar', ou 'en'
            context: Contexte RAG récupéré
            legal_mode: 'penal', 'civil', 'commercial', etc.
            entities: Dict d'entités extraites
            config: Configuration avancée (optionnel)
            sources_count: Nombre de sources utilisées (pour la confiance)
        """
        # Détection config par défaut
        if config is None:
            config = PromptConfig(
                language="ar" if language in ["ar", "arabic"] else "fr",
                legal_mode=LegalMode(legal_mode) if legal_mode in [m.value for m in LegalMode] else LegalMode.GENERAL
            )
        
        lang = config.language
        mode = config.legal_mode
        
        # Récupération profil mode
        profile = cls.MODE_PROFILES.get(mode, cls.MODE_PROFILES[LegalMode.GENERAL])[lang]
        
        # Construction des blocs modulaires
        blocks = []
        
        # 1. RÔLE ET PERSONA
        role_block = f"""Tu es {profile['role']}.
DOMAINE DE SPÉCIALISATION : {profile['focus']}.
TON : {profile['tone']}."""
        blocks.append(role_block)
        
        # 2. INSTRUCTIONS SPÉCIFIQUES AU MODE
        mode_instructions = cls._get_mode_specific_instructions(mode, lang)
        blocks.append(mode_instructions)
        
        # 3. GESTION DES ENTITÉS (si activé)
        if config.include_entities and entities:
            entity_block = cls._format_entities(entities, lang)
            blocks.append(entity_block)
        
        # 4. FORMAT DE RÉPONSE
        format_template = cls.FORMAT_TEMPLATES.get(config.response_format, cls.FORMAT_TEMPLATES[ResponseFormat.DETAILED])
        format_block = format_template.get(lang, format_template["fr"]).format(max_words=config.max_length_words)
        blocks.append(format_block)
        
        # 5. RÈGLES ABSOLUES (toujours présentes)
        confidence_instruction = cls._get_confidence_instruction(config.confidence_level, sources_count, lang)
        
        absolute_rules = f"""
RÈGLES ABSOLUES (À RESPECTER STRICTEMENT) :
{confidence_instruction}
1. Base ta réponse UNIQUEMENT sur le contexte documentaire fourni ci-dessous.
2. Cite les articles précisément (ex: "Art. 215 Code pénal" ou "Art. 15 COC").
3. Si le contexte est insuffisant, dis-le CLAIREMENT : "[Insuffisance documentaire] Je ne dispose pas de suffisamment d'informations pour répondre avec certitude."
4. Distribue les niveaux de certitude : [CERTAIN], [VRAISEMBLABLE], [POSSIBLE] selon la qualité des sources.
5. {'Inclus une section RISQUES si applicable.' if config.include_warnings else ''}
6. {'Mentionne les sources utilisées [Source X].' if config.include_sources else ''}
7. Langue obligatoire : {'Français' if lang == 'fr' else 'Arabe' if lang == 'ar' else 'Anglais'} (même si la question est dans une autre langue).
"""
        blocks.append(absolute_rules)
        
        # 6. CITATION STYLE
        if config.citation_style == CitationStyle.STRICT:
            blocks.append("CITATION : Utilise le texte exact des articles entre guillemets si possible.")
        elif config.citation_style == CitationStyle.PARAPHRASE:
            blocks.append("CITATION : Reformulation autorisée mais conserve le sens juridique exact.")
        
        # 7. FOCUS AREAS (si spécifié)
        if config.focus_areas:
            focus_text = "PRIORITÉS D'ANALYSE : " + " | ".join(config.focus_areas)
            blocks.append(focus_text)
        
        # Assemblage final
        system_prompt = "\n\n".join(blocks)
        
        # Ajout contexte à la fin (important pour LLM)
        final_prompt = f"""{system_prompt}

CONTEXTE DOCUMENTAIRE FOURNI ({sources_count} sources) :
{'='*60}
{context[:8000] if len(context) > 8000 else context}
{'='*60}

QUESTION À TRAITER : [USER_QUESTION_PLACEHOLDER]

RÉPONDEZ MAINTENANT SELON LES INSTRUCTIONS CI-DESSUS :"""
        
        return final_prompt
    
    @classmethod
    def build_comparison_prompt(cls, scenario_a: str, scenario_b: str, legal_mode: LegalMode, lang: str = "fr") -> str:
        """Prompt spécial pour comparaison de deux situations."""
        profile = cls.MODE_PROFILES[legal_mode][lang]
        
        return f"""Tu es {profile['role']}. Compare les deux situations suivantes selon le droit tunisien.

SITUATION A :
{scenario_a}

SITUATION B :
{scenario_b}

FORMAT : Tableau comparatif + conclusion sur la meilleure option juridique.
Cite les différences critiques en droit.
"""
    
    @classmethod
    def build_extraction_prompt(cls, text: str, extraction_type: str = "entities", lang: str = "fr") -> str:
        """Prompt pour extraction structurée (usage interne OCR)."""
        templates = {
            "entities": {
                "fr": """Extrais les entités juridiques du texte suivant au format JSON :
{
  "parties": [{"nom": "...", "role": "demandeur/defendeur"}],
  "articles_cites": ["..."],
  "dates": ["..."],
  "montants": ["..."],
  "type_document": "..."
}""",
                "ar": """استخرج الكيانات القانونية من النص بصيغة JSON..."""
            }
        }
        return templates.get(extraction_type, templates["entities"])[lang] + f"\n\nTEXTE : {text[:4000]}"
    
    @staticmethod
    def _get_mode_specific_instructions(mode: LegalMode, lang: str) -> str:
        """Instructions spécifiques très détaillées par mode."""
        instructions = {
            (LegalMode.PENAL, "fr"): """
SPÉCIFICITÉS DROIT PÉNAL :
- Vérifie toujours la qualification juridique (crime/délit/contravention)
- Mentionne la peine maximale ET minimale si applicable
- Précise la juridiction compétente selon la peine encourue (TPI/Cour d'appel)
- Signale les délais de prescription (Art. 10 et suivants CPP)
- ATTENTION aux droits de la défense (garde à vue, avocat)""",
            
            (LegalMode.PENAL, "ar"): """
خصوصيات القانون الجزائي:
- تحقق دائماً من التكييف القانوني (جناية/جنحة/مخالفة)
- اذكر العقوبة القصوى والدنيا إن وجدت
- حدد المحكمة المختصة حسب العقوبة...""",
            
            (LegalMode.CIVIL, "fr"): """
SPÉCIFICITÉS DROIT CIVIL :
- Vérifie la capacité des parties (majeur, non failli)
- Analyse la validité du consentement (vice du consentement ?)
- Précise les délais de prescription (3 ans, 10 ans, 30 selon cas)
- Mentionne les sûretés si applicable (garanties, hypothèques)
- Distribue les charges probatoires""",
            
            (LegalMode.CIVIL, "ar"): """
خصوصيات القانون المدني:
- تحقق من أهلية الأطراف
- حلل صحة الرضا (عيوب الرضا؟)
- حدد آجال التقادم...""",
            
            (LegalMode.COMMERCIAL, "fr"): """
SPÉCIFICITÉS DROIT COMMERCIAL :
- Vérifie la qualité de commerçant
- Distribue les obligations comptables
- Mentionne les effets de commerce si applicable (lettre de change, chèque)
- Précise les procédures collectives (redressement judiciaire, liquidation)""",
        }
        return instructions.get((mode, lang), "")
    
    @staticmethod
    def _format_entities(entities: Dict[str, Any], lang: str) -> str:
        """Formate les entités de manière structurée."""
        if not entities:
            return ""
        
        lines = ["ENTITÉS IDENTIFIÉES DANS LA QUESTION :"]
        
        # Mapping pour affichage
        labels = {
            "fr": {
                "articles_cp": "Articles CP", "articles_cpp": "Articles CPP",
                "articles_coc": "Articles COC", "montants": "Montants",
                "parties": "Parties", "dates": "Dates", "tribunaux": "Juridictions"
            },
            "ar": {
                "articles_cp": "مواد مجلة الجزاء", "articles_coc": "مواد مجلة الالتزامات",
                "parties": "الأطراف", "montants": "المبالغ"
            }
        }
        labels_map = labels.get(lang, labels["fr"])
        
        for key, value in entities.items():
            if value:
                label = labels_map.get(key, key.replace("_", " ").title())
                if isinstance(value, list):
                    val_str = " | ".join(str(v) for v in value[:3])
                else:
                    val_str = str(value)
                lines.append(f"  • {label}: {val_str}")
        
        return "\n".join(lines)
    
    @staticmethod
    def _get_confidence_instruction(confidence: float, sources: int, lang: str) -> str:
        """Adapte les instructions selon le niveau de confiance des sources."""
        if lang == "ar":
            if confidence < 0.6:
                return f"[Sources limitées: {sources}] Soyez prudent et nuancé."
            return "[Sources fiables] Vous pouvez être affirmatif."
        
        if confidence < 0.6:
            return f"[CONFIDENCE: {confidence:.0%}] Les sources sont partielles ({sources} documents). Adoptez un ton prudent, nuancez chaque affirmation."
        elif confidence < 0.8:
            return f"[CONFIDENCE: {confidence:.0%}] Sources modérées. Vous pouvez répondre mais signalez les incertitudes."
        else:
            return f"[CONFIDENCE: {confidence:.0%}] Sources solides. Réponse affirmative autorisée mais restez factuel."


# ── COMPATIBILITÉ ANCIENNE VERSION ─────────────────────────────────────────

class PromptBuilder(LegalPromptBuilder):
    """
    Wrapper pour compatibilité avec code existant.
    Maintient l'API exacte de l'ancien PromptBuilder.
    """
    
    @staticmethod
    def build_synthesis_prompt(
        language: str,
        context: str,
        legal_mode: str = "general",
        entities: Dict[str, Any] = None,
    ) -> str:
        """API legacy - appelle la nouvelle version avec defaults."""
        # Mapping ancien vers nouveau
        mode_map = {
            "penal": LegalMode.PENAL,
            "civil": LegalMode.CIVIL,
            "general": LegalMode.GENERAL
        }
        
        config = PromptConfig(
            language="ar" if "ar" in language.lower() else "fr",
            legal_mode=mode_map.get(legal_mode, LegalMode.GENERAL),
            response_format=ResponseFormat.DETAILED,
            include_entities=bool(entities)
        )
        
        return LegalPromptBuilder.build_synthesis_prompt(
            language=language,
            context=context,
            legal_mode=legal_mode,
            entities=entities or {},
            config=config,
            sources_count=3  # Default legacy
        )
    
    @staticmethod
    def build_query_prompt(
        question: str,
        context: str,
        entities: Dict[str, Any],
        drive_context: Dict[str, str],
    ) -> str:
        """Compatibilité totale avec ancien code."""
        category = drive_context.get("category", "general").lower()
        legal_mode = "penal" if "pénal" in category else ("civil" if "civil" in category else "general")
        
        system = PromptBuilder.build_synthesis_prompt(
            "Français" if "fr" in category else "Arabe",
            context,
            legal_mode,
            entities
        )
        return f"{system}\n\nQuestion spécifique : {question}"


# ── UTILITAIRES RAPIDES ───────────────────────────────────────────────────

def quick_prompt(question: str, mode: str = "general", lang: str = "fr") -> str:
    """Fonction utilitaire rapide sans config complexe."""
    config = PromptConfig(
        language=lang,
        legal_mode=LegalMode(mode) if mode in ["penal", "civil", "commercial"] else LegalMode.GENERAL,
        response_format=ResponseFormat.CONCISE
    )
    return LegalPromptBuilder.build_synthesis_prompt(
        lang, "[CONTEXTE_NON_FOURNI]", mode, {}, config, 0
    )


# ── TESTS ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test 1: Mode pénal détaillé
    config_penal = PromptConfig(
        language="fr",
        legal_mode=LegalMode.PENAL,
        response_format=ResponseFormat.DETAILED,
        focus_areas=["délais", "compétence"]
    )
    
    prompt1 = LegalPromptBuilder.build_synthesis_prompt(
        "fr",
        "Contexte: Art. 215 CP - Le vol est puni de 5 ans...",
        "penal",
        {"articles_cp": ["Art. 215"], "type_incident": "vol"},
        config_penal,
        sources_count=2
    )
    print("=== TEST MODE PÉNAL ===")
    print(prompt1[:1000] + "...")
    print()
    
    # Test 2: Mode civil arabe concis
    prompt2 = LegalPromptBuilder.build_synthesis_prompt(
        "ar",
        "عقد الكراء...",
        "civil",
        {},
        PromptConfig(language="ar", legal_mode=LegalMode.CIVIL, response_format=ResponseFormat.CONCISE),
        1
    )
    print("=== TEST MODE CIVIL ARABE (CONCIS) ===")
    print(prompt2[:800] + "...")