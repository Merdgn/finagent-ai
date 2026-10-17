
import os

from groq import Groq


class GroqClient:
    def __init__(
        self,
        model: str = "openai/gpt-oss-20b",
    ):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not configured"
            )

        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        messages = [
            {
                "role": "system",
                "content": system_prompt
                or "You are a helpful AI assistant.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_completion_tokens=1024,
        )

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError("Groq returned an empty response")

        return content.strip()
