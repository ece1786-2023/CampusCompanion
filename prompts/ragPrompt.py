from openai import OpenAI
import csv
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_KEY = os.getenv('OPENAI_KEY')
client = OpenAI(
    api_key=OPENAI_KEY
)

def getRAGQuery(content):
    response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                "role": "user",
                "content": content
                },
                {
                "role": "assistant",
                "content":  """Format a RAG search query to look for other similar courses based on the following summarization text, skills, interests, academic goals, and courses taken, exand to include other relevant terms. This will be a search query so only include relevant terms, it doesn't need to be a full sentence. The output MUST include department code
                Input: I am in History and I am interested in Roman history, and other ancient civilizations. I am also working part time as I am in a graduate program
                Output: Hist, Roman History, Ancient Civilizations, Ancient History, Ancient Greece, Roman Empire, Graduate
                Input: 
                """
                }
            ],
            temperature=0.9,
            max_tokens=256,
            top_p=0.75,
            frequency_penalty=0,
            presence_penalty=0
            )
    output = response.choices[0].message.content.strip()
    print("Generated RAG search query:")
    print(output)
    return output


def main():
    # Example of using the language chain
    user_input = "{'interests': 'Logic and Machine Learning', 'goal': 'graduate program in Electrical and Computer Engineering', 'experience': 'completed courses in Trustworthy Machine Learning and an Introduction to Artificial Intelligence', 'summary': 'The student is enrolled in a graduate program in Electrical and Computer Engineering with an interest in Logic and Machine Learning. They have completed courses in Trustworthy Machine Learning and an Introduction to Artificial Intelligence.'}"
    query = getRAGQuery(user_input)
    print(query)

if __name__ == "__main__":
    main()


