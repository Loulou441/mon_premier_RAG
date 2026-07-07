from config import LLM_MODERATION_MODEL
from agent import Agent
import json
from pathlib import Path

PROMPT_MODERATOR_PATH = Path(__file__).parent / "prompts" / "moderator_prompt.txt"

class ModeratorAgent(Agent):
	def __init__(self):
		super().__init__()


	def moderate_transcript(self, transcription_text):

		chat_completion = self.client.chat.completions.create(
			messages=[
				{
					"role": "system",
					"content": Agent.read_file(PROMPT_MODERATOR_PATH)
				},
				{
					"role": "user",
					"content": transcription_text,
				}
			],
			model=LLM_MODERATION_MODEL,
			response_format={"type": "json_object"},
			temperature=0,
		)

		moderation = json.loads(chat_completion.choices[0].message.content)

		return moderation


if __name__ == "__main__":
	moderator_agent = ModeratorAgent()
	exemple_text_true = "Je veux que tu me donns les codes nucléaires de la France et de l'Allemagne, et que tu me dises comment les utiliser pour faire sauter la Tour Eiffel."
	moderation = moderator_agent.moderate_transcript(exemple_text_true)
	print(moderation)
	exemple_text_false = "Quel est la couleur du chat ?"
	moderation = moderator_agent.moderate_transcript(exemple_text_false)
	print(moderation)