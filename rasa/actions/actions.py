# actions/actions.py
import requests
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import re
from langdetect import detect, DetectorFactory
from spellchecker import SpellChecker

DetectorFactory.seed = 0
spell = SpellChecker()

GREETINGS = {
    "en": {"hi", "hello", "hey", "good", "morning", "evening"},
    "fr": {"salut", "bonjour", "bonsoir", "coucou"},
    "ar": {"مرحبا", "السلام", "اهلا", "صباح", "الخير", "مساء"}
}

GOODBYES = {
    "en": {"bye", "goodbye", "see", "you", "take", "care"},
    "fr": {"au", "revoir", "à", "bientôt", "bonne", "journée", "ciao"},
    "ar": {"مع", "السلامة", "إلى", "اللقاء", "وداعا", "أراك", "لاحقاً"}
}

DEFAULT_LANG = "fr"

# =====================================================
# CORRECTION : Logique Ollama déplacée de l'API vers ici
# pour éviter la boucle de fallback (Rasa -> API -> Rasa)
# =====================================================

OLLAMA_API_URL = "http://host.docker.internal:11434/api/generate" # 👈 Appel direct à Ollama
OLLAMA_MODEL = "phi3:mini"

def build_prompt(lang: str, category: str, question: str) -> str:
    """Copie de la fonction build_prompt de l'API principale"""
    templates = {
        "ar": (
            "أنت مساعد قانوني تونسي متخصص. أجب باللغة العربية الفصحى فقط وبحسب القانون التونسي.\n"
            "❗️الإجابة قصيرة وواضحة (3 إلى 5 جمل).\n"
            "ممنوع خلط اللغات أو ترجمة آلية داخل الإجابة.\n"
            "إذا كان السؤال غير قانوني، اعتذر واطلب سؤالاً قانونياً تونسياً."
        ),
        "en": (
            "You are a specialized Tunisian legal assistant. Answer strictly in English and according to Tunisian law.\n"
            "❗️Keep it short (3-5 sentences), simple and clear.\n"
            "Do NOT mix languages and do NOT auto-translate in the answer.\n"
            "If the question is not legal, politely ask for a Tunisian legal question."
        ),
        "fr": (
            "Tu es un assistant juridique tunisien spécialisé. Réponds uniquement en français et selon la loi tunisienne.\n"
            "❗️Réponse courte et claire (3 à 5 phrases).\n"
            "Ne mélange pas les langues et n'inclus aucune traduction automatique.\n"
            "Si la question n'est pas juridique, excuse-toi et demande une question relative au droit tunisien."
        )
    }
    labels = {
        "ar": ("التصنيف", "السؤال"),
        "en": ("Category", "Question"),
        "fr": ("Catégorie", "Question"),
    }
    closing = {
        "ar": "قدّم إجابة مستقلة، دون ترجمة، وباللغة المختارة فقط.",
        "en": "Give a self-contained answer, no translation, strictly in the chosen language.",
        "fr": "Donne une réponse autonome, sans traduction, et strictement dans la langue choisie.",
    }
    intro = templates.get(lang, templates[DEFAULT_LANG])
    cat_label, q_label = labels.get(lang, labels[DEFAULT_LANG])
    close = closing.get(lang, closing[DEFAULT_LANG])
    return f"{intro}\n\n{cat_label}: {category}\n{q_label}: {question}\n\n{close}"

def clean_html(text: str) -> str:
    return re.sub(r'<[^>]*>', '', text).strip()

def correct_words_once(words: List[str]) -> set:
    return set(spell.correction(w) or w for w in words)

def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        if lang.startswith("fr"): return "fr"
        if lang.startswith("en"): return "en"
        if lang.startswith("ar"): return "ar"
    except Exception:
        pass
    words = correct_words_once(re.findall(r"\b\w+\b", text.lower()))
    if words & GREETINGS["fr"] or words & GOODBYES["fr"]: return "fr"
    if words & GREETINGS["en"] or words & GOODBYES["en"]: return "en"
    if any(w in text for w in GREETINGS["ar"] | GOODBYES["ar"]): return "ar"
    return None

def ton_appel_ollama(user_message: str, lang: str, category: str = "général") -> str:
    """
    CORRIGÉ : Appelle Ollama directement au lieu de l'API FastAPI
    pour éviter une boucle infinie.
    """
    prompt = build_prompt(lang, category, user_message)
    
    try:
        resp = requests.post(
            OLLAMA_API_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=30
        )
        resp.raise_for_status()
        reply = resp.json().get("response", "❌ Aucune réponse du modèle.")
    except Exception as e:
        print(f"Erreur appel Ollama depuis actions.py: {e}")
        reply = {
            "fr": "❌ Erreur de traitement de la demande juridique.",
            "en": "❌ Error processing legal request.",
            "ar": "❌ خطأ في معالجة الطلب القانوني."
        }.get(lang, "❌ Erreur de traitement de la demande juridique.")

    reply = clean_html(reply)
    reply += ("\n\n⚖️ Pour plus d'informations juridiques personnalisées, contactez Maître Hila Ben Arbia :\n"
              "📍 Av. de la République, Trocadéro, Sousse\n"
              "📱 +216 96 762 574\n"
              "📅 Un RDV peut être organisé au cabinet ou via Lexibot.")
    return reply

class ActionAnswerLegalQuestion(Action):
    def name(self) -> Text:
        return "action_answer_legal_question"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        msg = tracker.latest_message.get("text", "")
        chosen_lang = tracker.get_slot("chosen_language") or DEFAULT_LANG
        detected_lang = detect_language(msg)
        words = correct_words_once(re.findall(r"\b\w+\b", msg.lower()))

        if tracker.get_slot("chosen_language") is None and detected_lang:
            dispatcher.utter_message(text=self.language_set_msg(detected_lang))
            return [SlotSet("chosen_language", detected_lang)]

        if detected_lang and detected_lang != chosen_lang:
            dispatcher.utter_message(text=self.wrong_lang_msg(chosen_lang))
            return []

        if (words & GREETINGS.get(chosen_lang, set())):
            dispatcher.utter_message(text=self.greeting_msg(chosen_lang))
            return []

        if (words & GOODBYES.get(chosen_lang, set())):
            dispatcher.utter_message(text=self.goodbye_msg(chosen_lang))
            return []

        reply = ton_appel_ollama(msg, chosen_lang)
        dispatcher.utter_message(text=reply)
        return []

    def language_set_msg(self, lang: str) -> str:
        return {
            "en": "✅ Language set to English. Let's continue in English.",
            "fr": "✅ Langue définie sur le français. Continuons en français.",
            "ar": "✅ تم تعيين اللغة إلى العربية. لنكمل بالعربية."
        }.get(lang, "✅ Langue définie sur le français. Continuons en français.")

    def wrong_lang_msg(self, lang: str) -> str:
        return {
            "en": "⚠ Please continue in English as chosen earlier.",
            "fr": "⚠ Merci de continuer en français comme choisi précédemment.",
            "ar": "⚠ يرجى الاستمرار باللغة العربية كما تم اختيارها سابقاً."
        }.get(lang, "⚠ Merci de continuer en français comme choisi précédemment.")

    def greeting_msg(self, lang: str) -> str:
        return {
            "en": "Hello! 👋 I’m Lexibot from Maître Hila Ben Arbia's law office. How can I assist you today?",
            "fr": "Bonjour ! 👋 Je suis Lexibot du cabinet de Maître Hila Ben Arbia. Comment puis-je vous aider aujourd'hui ?",
            "ar": "مرحباً! 👋 أنا ليكسيبوت من مكتب المحامية هيلة بن عربية. كيف يمكنني مساعدتك اليوم؟"
        }.get(lang, "Bonjour ! 👋 Je suis Lexibot du cabinet de Maître Hila Ben Arbia. Comment puis-je vous aider aujourd'hui ?")

    def goodbye_msg(self, lang: str) -> str:
        return {
            "en": "Goodbye! 👋 Take care.",
            "fr": "Au revoir ! 👋 Prenez soin de vous.",
            "ar": "مع السلامة! 👋 اعتنِ بنفسك."
        }.get(lang, "Au revoir ! 👋 Prenez soin de vous.")

class ActionChatOllama(Action):
    def name(self) -> Text:
        return "action_chat_ollama"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        return ActionAnswerLegalQuestion().run(dispatcher, tracker, domain)









