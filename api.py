from groq import Groq
import os

# Step 1: Initialize the API client
client = Groq(api_key=os.getenv("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"))  # Replace with your actual API key

# Step 2: Define messages for the chat
searchTerm = input("Enter your question: ")  # Takes user input dynamically
  # Replace with your dynamic input if needed
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": searchTerm}
]

# Step 3: Create a function to fetch a chat completion
def get_chat_response(client, messages, stream=True):
    try:
        # Request chat completion from the API
        completion = client.chat.completions.create(
            model="llama3-70b-8192",  # Replace with the correct model name
            messages=messages,
            temperature=0.7,         # Adjust creativity level
            max_tokens=256,          # Set token limit
            top_p=0.9,               # Sampling diversity
            stream=stream,           # Enable or disable streaming
            stop=None                # Specify stop conditions if needed
        )
        
        # Step 4: Handle the response
        if stream:
            print("Streaming response:")
            for chunk in completion:
                # Print content as it streams
                print(chunk.choices[0].delta.content or "", end="")
        else:
            # Print full response at once
            print("Full response:")
            print(completion.choices[0].text)
    
    except Exception as e:
        print(f"Error occurred: {e}")

# Step 5: Call the function to get a response
get_chat_response(client, messages)
