
def chatbot():
    print("Chatbot: Hello! Welcome to the chatbot.")
    print("Chatbot: How can I help you?")
    print("Type 'bye' or 'exit' to stop the conversation.\n")

    while True:
        user_input = input("You: ").lower().strip()

        if user_input in ["bye", "exit"]:
            print("Chatbot: Goodbye! Have a nice day!")
            break

        elif "hello" in user_input or "hi" in user_input:
            print("Chatbot: Hello! How can I help you?")

        elif "how are you" in user_input:
            print("Chatbot: I am fine! Thank you for asking.")

        elif "your name" in user_input:
            print("Chatbot: I am a simple rule-based chatbot.")

        elif "help" in user_input:
            print("Chatbot: Sure! Tell me what you need help with.")

        elif "thank" in user_input:
            print("Chatbot: You're welcome!")

        elif "time" in user_input:
            print("Chatbot: I cannot provide the current time.")

        else:
            print("Chatbot: Sorry, I don't understand that.")

chatbot()
