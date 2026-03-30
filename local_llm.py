import ollama

class LocalLLM:
    def __init__(self, model_name="llama3.2"):
        self.model_name = model_name

    def __call__(self, prompt: str) -> str:
        response = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
        return response['choices'][0]['message']['content']
