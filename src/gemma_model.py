import ollama
import sys
import base64
import os
import cv2

class GemmaModel:
    def generate(self, prompt, max_tokens=100, temperature=0.7):
        response = ollama.chat(
            model = "gemma3",
            messages = [{"role":"user", "content":prompt}],
            stream=False,
            options={
                "num_predict": max_tokens,
                "temperature": temperature,
            }
        )

        if prompt.lower() == "/end":
            sys.exit(0)

        return response['message']['content']
    
    def generate_stream(self, prompt, max_tokens=500, temperature=0.7):
        stream = ollama.chat(
            model="gemma3",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={
                "num_predict": max_tokens,
                "temperature": temperature,
            }
        )

        if prompt.lower() == "/end":
            sys.exit(0)

        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                yield chunk['message']['content']

# Example usage
if __name__ == "__main__":
    model = GemmaModel()
    os.system("pwd")
    response = model.image_to_text("image.jpeg")
    print(response)
    