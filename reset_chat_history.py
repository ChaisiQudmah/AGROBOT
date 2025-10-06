import json

CHAT_HISTORY_FILE = "chat_history.json"

with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
    json.dump([], f, ensure_ascii=False, indent=4)

print("✅ chat_history.json reset to empty list []")
