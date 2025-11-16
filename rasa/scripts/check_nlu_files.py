import os
import yaml

data_dir = "data"

  # adapte si nécessaire

for filename in os.listdir(data_dir):
    path = os.path.join(data_dir, filename)
    if not filename.endswith((".yml", ".yaml", ".json")):
        continue
    try:
        with open(path, encoding="utf-8") as f:
            content = yaml.safe_load(f)
        if isinstance(content, int):
            print(f"❌ Fichier invalide : {filename} contient un entier brut")
        elif not isinstance(content, dict):
            print(f"❌ Fichier invalide : {filename} n'est pas un dictionnaire YAML")
        elif "nlu" not in content and "stories" not in content and "rules" not in content:
            print(f"⚠️ Fichier suspect : {filename} ne contient pas de bloc connu (nlu, stories, rules)")
    except Exception as e:
        print(f"❌ Erreur dans {filename} : {e}")

