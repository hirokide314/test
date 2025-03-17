import os
import google.generativeai as genai

GOOGLE_API_KEY = "AIzaSyCZa9YOjdfjVnonGPdQGKpcZ9M6Mgw4HXY"
print(GOOGLE_API_KEY)

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-1.5-pro-latest")

prompt = "こんにちは"
print(prompt)
response  = model.generate_content(prompt)
print(response.text)