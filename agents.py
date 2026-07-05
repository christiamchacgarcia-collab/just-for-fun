import os
import json
import time
from anthropic import Anthropic
from openai import OpenAI
from google import genai
from dotenv import load_dotenv

load_dotenv()

anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


class Agent:
    def __init__(self, name, system_prompt, provider="anthropic", use_web_search=False):
        self.name = name
        self.system_prompt = system_prompt
        self.provider = provider  # "anthropic", "openai", or "google"
        self.use_web_search = use_web_search
        self.memory_file = f"{name.lower()}_memory.json"
        self.history = self.load_memory()

    def load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def save_memory(self):
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def clear_memory(self):
        self.history = []
        self.save_memory()

    def memory_summary(self):
        return {
            "name": self.name,
            "message_count": len(self.history),
            "last_messages": self.history[-4:] if self.history else []
        }

    def run(self, user_input):
        self.history.append({"role": "user", "content": user_input, "ts": time.time()})

        if self.provider == "anthropic":
            reply = self._run_anthropic()
        elif self.provider == "openai":
            reply = self._run_openai()
        elif self.provider == "google":
            reply = self._run_google()
        else:
            reply = self._run_anthropic()

        self.history.append({"role": "assistant", "content": reply, "ts": time.time()})
        self.save_memory()
        return reply

    def _run_anthropic(self):
        api_messages = [{"role": m["role"], "content": m["content"]} for m in self.history]
        kwargs = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 1024,
            "system": self.system_prompt,
            "messages": api_messages,
        }
        if self.use_web_search:
            kwargs["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]

        response = anthropic_client.messages.create(**kwargs)
        parts = [block.text for block in response.content if block.type == "text"]
        return "\n".join(parts)

    def _run_openai(self):
        api_messages = [{"role": "system", "content": self.system_prompt}]
        api_messages += [{"role": m["role"], "content": m["content"]} for m in self.history]

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=api_messages,
        )
        return response.choices[0].message.content

    def _run_google(self):
        from google.genai import types

        api_messages = [{"role": m["role"], "content": m["content"]} for m in self.history]
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in api_messages)
        full_prompt = f"{self.system_prompt}\n\n{transcript}"

        config = None
        if self.use_web_search:
            config = types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )

        response = google_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
            config=config,
        )
        return response.text
    