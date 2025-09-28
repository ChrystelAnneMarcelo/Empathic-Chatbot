#pip install textblob
import random
from textblob import TextBlob

class EmpathicChatbot:
    def __init__(self):
        # Keep some keywords for stress (since stress can be negative but not always obvious)
        self.stress_keywords = {'stressed', 'overwhelmed', 'anxious', 'pressure', 'worried', 'nervous', 'tense', 'burnt out', 'exhausted'}
        self.happiness_keywords = {'happy', 'excited', 'great', 'amazing', 'wonderful', 'fantastic', 'joy', 'thrilled'}

        # Responses remain the same
        self.sad_responses = [
            "I'm really sorry you're feeling this way. That sounds really tough.",
            "It's completely okay to feel sad sometimes. Your feelings are valid.",
            "I hear you, and I'm here for you. Would you like to talk about what's bothering you?"
        ]
        self.happy_responses = [
            "That's wonderful to hear! I'm so happy for you!",
            "How amazing! Tell me more about what made you feel this way!",
            "Your joy is contagious! What's making your day so great?"
        ]
        self.stress_responses = [
            "I can hear how overwhelmed you feel. That sounds really challenging.",
            "You're carrying a lot right now. Maybe try breaking things down into smaller steps?",
            "Feeling stressed is your body's way of telling you to slow down. Would a short break help?"
        ]
        self.default_responses = [
            "I'm here to listen. How are you feeling today?",
            "I care about how you're doing. Would you like to share more?"
        ]

    def detect_emotion(self, text):
        # Clean text for keyword check
        clean_text = text.lower()
        
        # First, check for stress keywords (often urgent)
        if any(word in clean_text for word in self.stress_keywords):
            return 'stress'
        
        # Use TextBlob for sentiment
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 (negative) to +1 (positive)
        
        if polarity > 0.1:
            return 'happy'
        elif polarity < -0.1:
            return 'sad'
        else:
            return 'neutral'

    def get_response(self, user_input):
        emotion = self.detect_emotion(user_input)
        if emotion == 'sad':
            return random.choice(self.sad_responses)
        elif emotion == 'happy':
            return random.choice(self.happy_responses)
        elif emotion == 'stress':
            return random.choice(self.stress_responses)
        else:
            return random.choice(self.default_responses)

    def run(self):
        print("👋 Hi! I'm your empathic chatbot. I'm here to listen.")
        print("Type 'quit' to exit.\n")
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Bot: Take care! 💙")
                break
            if not user_input:
                continue
            response = self.get_response(user_input)
            print(f"Bot: {response}\n")

if __name__ == "__main__":
    bot = EmpathicChatbot()
    bot.run()