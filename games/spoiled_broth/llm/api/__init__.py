from abc import ABC, abstractmethod


class _LLMWrapper(ABC):
    """
    Base class for LLM API
    """

    @abstractmethod
    def prompt(self, prompt: str):
        raise NotImplementedError("Subclasses should implement this!")


class OpenAIWrapper(_LLMWrapper):
    """
    OpenAI API
    """

    def __init__(self, api_key: str = None, model: str = None):
        if api_key is None:
            from spoiled_broth.llm.api._secrets import OPEN_AI_KEY
            api_key = OPEN_AI_KEY
        if model is None:
            from spoiled_broth.llm.api._config import model_openai
            model = model_openai

        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "OpenAI SDK not installed. Install the AI requirements or use the '2-player' image."
            ) from e

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def prompt(self, prompt: str):
        response = self.client.chat.completions.create(model=self.model,
                                                       messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content.strip()


class GeminiWrapper(_LLMWrapper):
    """
    Gemini API
    """

    def __init__(self, api_key: str = None, model: str = None, project: str = None, location: str = None):
        if api_key is None:
            from spoiled_broth.llm.api._secrets import GEMINI_KEY
            api_key = GEMINI_KEY
        if model is None:
            from spoiled_broth.llm.api._config import model_gemini
            model = model_gemini
        if project is None:
            from spoiled_broth.llm.api._config import project_gemini
            project = project_gemini
        if location is None:
            from spoiled_broth.llm.api._config import location_gemini
            location = location_gemini

        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise RuntimeError(
                "Google.genai not installed. Install the AI requirements or use the '2-player' image."
            ) from e

        self._types = types

        self.model = model
        self.client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
        )

        self.generate_content_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            max_output_tokens=8192,
            response_modalities=["TEXT"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="zephyr"
                    )
                ),
            ),
            safety_settings=[types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="OFF"
            ), types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="OFF"
            ), types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="OFF"
            ), types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="OFF"
            )],
        )

    def prompt(self, prompt: str):
        types = self._types
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=f"""{prompt}""")
                ]
            )
        ]
        res = []

        for chunk in self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=self.generate_content_config,
        ):
            if chunk.text:
                res.append(chunk.text)

        return ''.join(res)


if __name__ == "__main__":
    g_m = GeminiWrapper()
    print(g_m.prompt("What is the capital of France?"))
