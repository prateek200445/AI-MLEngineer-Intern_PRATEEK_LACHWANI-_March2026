import json
import google.generativeai as genai
import os


with open("retrieval_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")  

ragas_data = []

#Limit to 10 questions
for idx, item in enumerate(data[:10], start=1):
    question = item["question"]
    contexts = item.get("retrieved_texts", [])
    ground_truth = item["ground_truth"]

    
    context_text = "\n".join(contexts)
    prompt = f"""
    Answer the question below based only on the given context.

    Context:
    {context_text}

    Question:
    {question}
    """

   
    try:
        response = model.generate_content(prompt)
        generated_answer = response.text.strip()
    except Exception as e:
        print(f" Error generating answer for Q{idx}: {e}")
        generated_answer = ""

    
    ragas_data.append({
        "question": question,
        "contexts": contexts,
        "answer": ground_truth,
        "generated_answer": generated_answer,
        "reference": ground_truth   
    })

    print(f" Processed {idx}: {question[:60]}...")


with open("ragas_eval_data.json", "w", encoding="utf-8") as f:
    json.dump(ragas_data, f, indent=2, ensure_ascii=False)

print("\n Saved 10 Q&A items to ragas_eval_data.json — ready for RAGAS evaluation!")
