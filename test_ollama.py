import ollama

# Call your local Llama model
response = ollama.generate(
    model='llama3.1:8b', 
    prompt='Write a one-sentence tagline for a coffee shop.'
)

# Print only the text answer
print(response['response'])
