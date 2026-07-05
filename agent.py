import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

class Agent:
    def __init__(self, name, system_prompt):
        self.name = name
        self.system_prompt = system_prompt
        self.history = []  # remembers this agent's own conversation

    def run(self, user_input):
        self.history.append({"role": "user", "content": user_input})

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=self.system_prompt,
            messages=self.history,
        )

        reply = response.content[0].text
        self.history.append({"role": "assistant", "content": reply})
        return reply