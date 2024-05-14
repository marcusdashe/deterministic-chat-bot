import numpy as np
import nltk
import random
import string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

nltk.download('punkt')  # Download the Punkt tokenizer for sentence splitting

# Example data
data = {
    "greetings": ["Hello", "Hi", "Hey", "Hi there"],
    "responses": ["Hello!", "Hi!", "Hey!", "Hi there! How can I help you today?"],
    "questions": [
        "What is your name?",
        "How are you?",
        "What do you do?",
        "What is machine learning?",
        "Tell me a joke."
    ],
    "answers": [
        "I am a chatbot.",
        "I am fine, thank you.",
        "I am here to help you with your questions.",
        "Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to learn from data.",
        "Why did the scarecrow win an award? Because he was outstanding in his field!"
    ]
}


# Tokenize sentences
def tokenize(text):
    return nltk.word_tokenize(text.lower())

# Create a vocabulary and preprocess input data
vectorizer = CountVectorizer(tokenizer=tokenize, stop_words='english')
X = vectorizer.fit_transform(data['questions'])
y = data['answers']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Function to get response to a user query
def get_response(user_query):
    user_query_vec = vectorizer.transform([user_query])
    similarity_scores = cosine_similarity(user_query_vec, X)
    response_index = similarity_scores.argmax()
    return data['answers'][response_index]

# Test the model with a few examples
for question in data['questions']:
    print(f"User: {question}")
    print(f"Bot: {get_response(question)}")
    print()


def chatbot():
    print("Chatbot: Hi! I'm a chatbot. Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        else:
            response = get_response(user_input)
            print(f"Chatbot: {response}")

# Start the chatbot
chatbot()


