import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_ollama import OllamaLLM
import sys
import os

# Test senaryoları için yol ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.evaluation_rules import get_test_cases

# CSV'den veri oku
df = pd.read_csv("data/logs.csv")

# LLM modelini yükle
llm = OllamaLLM(model="llama3")

# Ajan oluştur
agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=False,
    agent_type="zero-shot-react-description",  # daha stabil
    allow_dangerous_code=True
)

# Test senaryolarını al
test_cases = get_test_cases(df)

# Terminal arayüzü
print("📊 Data analysis agent is ready. Type 'q' to quit.")

while True:
    try:
        question = input("❓ Question: ")
        if question.lower() == "q":
            break

        # Cevabı al (sade string)
        answer = agent.run(question)
        print(f"\n✅ Answer: {answer}")

        # Test et
        matched = False
        for test in test_cases:
            if test["question"].lower() == question.lower():
                expected = str(test["expected"](df))
                if expected in answer:
                    print(f"🟢 Correct! (Expected: {expected})")
                else:
                    print(f"🔴 Wrong! (Expected: {expected})")
                matched = True
                break

        if not matched:
            print("ℹ️ No test rule found for this question.")

        print()

    except Exception as e:
        print(f"⚠️ Error: {e}")
