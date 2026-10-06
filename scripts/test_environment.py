import os
from dotenv import load_dotenv


load_dotenv()


assemblyai_key = os.getenv("ASSEMBLYAI_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")


print("AssemblyAI key loaded:", bool(assemblyai_key))
print("Gemini key loaded:", bool(gemini_key))