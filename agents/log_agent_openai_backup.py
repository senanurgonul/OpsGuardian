# 🧾 Bu dosya yedek amaçlı saklanmaktadır.
# Local LLM yerine OpenAI API kullanmak isteyenler için örnek ajandır.


import os
import pandas as pd
from dotenv import load_dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import OpenAI

# 1. API anahtarını .env'den al
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # ✅ BU ŞEKİLDE OLMALI!

# 2. CSV verisini oku
df = pd.read_csv("data/logs.csv")

# 3. GPT modelini başlat
llm = OpenAI(temperature=0, api_key=OPENAI_API_KEY)

# 4. Agent oluştur
agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)

# 5. Kullanıcıdan soru al
print("💬 Doğal dil ile log verisini sorgulayabilirsin. Çıkmak için 'q' yaz.")
while True:
    question = input("❓ Soru: ")
    if question.lower() == "q":
        break
    try:
        answer = agent.run(question)
        print(f"✅ Cevap: {answer}")
    except Exception as e:
        print(f"⚠️ Hata: {e}")
        print("🔑 API KEY:", OPENAI_API_KEY[:8] + "..." if OPENAI_API_KEY else "YOK!")
