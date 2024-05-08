import re

# Define patterns and responses
patterns = {
    r"(hi|hello|hey|greetings)( there)?": ["Hello!", "Hi!", "Hey there!"],
    r"how are you\??": ["I'm doing well, thank you!", "Great, thanks for asking!"],
    r"what's your name\??": ["I'm a chatbot!", "I'm your friendly chatbot!"],
    r"bye|goodbye": ["Goodbye!", "See you later!", "Bye!"]
    
}

def respond(message):
    for pattern, responses in patterns.items():
        match = re.search(pattern, message.lower())
        if match:return responses  # Return a random response from the matched pattern's responses
    return ["I'm not sure how to respond to that."]

# Example conversation loop
print("Bot: Hello! How can I assist you?")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Bot: Goodbye!")
        break
    response = respond(user_input)
    print("Bot:", response)