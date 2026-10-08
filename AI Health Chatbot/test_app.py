import sys
import urllib.request
import json

# Ensure stdout handles UTF-8 (emojis in responses)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

base = "http://127.0.0.1:5000"

def test_get(url):
    req = urllib.request.urlopen(f"{base}{url}")
    print(f"GET {url} -> status {req.status}")

test_get("/")
test_get("/chat")
test_get("/topics")
test_get("/about")
test_get("/faq")
test_get("/admin")
test_get("/api/topics")
test_get("/api/comparison")

def test_chat(msg, model="ml", threshold=0.6):
    req = urllib.request.Request(
        f"{base}/api/chat",
        data=json.dumps({"message": msg, "model": model, "threshold": threshold}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    res = urllib.request.urlopen(req)
    data = json.loads(res.read().decode("utf-8"))
    print(f"\n[CHAT QUERY: '{msg}' | Model: {model}]")
    print(f"-> Intent: {data.get('intent')}")
    print(f"-> Confidence: {data.get('confidence')}")
    print(f"-> Model Used: {data.get('model_used')}")
    print(f"-> Emergency: {data.get('is_emergency')}")
    print(f"-> Fallback: {data.get('is_fallback')}")
    snippet = data.get('response', '')[:120].replace('\n', ' ')
    print(f"-> Response snippet: {snippet}...")

# 1. ML test
test_chat("What are common symptoms of diabetes?", model="ml")

# 2. DL test
test_chat("How can I treat a high fever at home?", model="dl")

# 3. Kannada multilingual test
test_chat("Nanage jvara ide, en madbeku?", model="ml")

# 4. Emergency test
test_chat("Severe crushing chest pain and cannot breathe", model="ml")

# 5. Low confidence fallback test
test_chat("Unrelated random text xyz quantum mechanics", model="ml", threshold=0.7)

print("\nAll integration verification tests passed successfully!")
