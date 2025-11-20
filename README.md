## Development and Deployment of a 'Chat with LLM' Application Using the Gradio Blocks Framework

### AIM
To develop a simple web-based chat application that allows users to interact with a large language model (LLM) in real-time using Gradio Blocks.

### PROBLEM STATEMENT
Accessing and using LLMs via code or APIs can be complex. This experiment creates a user-friendly web interface to chat with an AI model easily.

### PROCEDURE / STEPS

1. **Setup Environment**
   - Install required Python packages (`gradio`, `text_generation`, `python-dotenv`).
   - Load your Hugging Face API key and endpoint from environment variables.

2. **Initialize LLM Client**
   - Connect to the LLM using `text_generation.Client`.
   - Set a timeout to handle large responses.

3. **Define Chat Functions**
   - `format_chat_prompt()` to structure conversation context.
   - `respond()` to send user input to the model and stream responses.
   - Maintain chat history for context-aware responses.

4. **Build Gradio Interface**
   - Add a `Chatbot` component to display conversation.
   - Add a `Textbox` for user input.
   - Include `Advanced options` (system message, temperature).
   - Add `Submit` and `Clear` buttons.

5. **Launch Application**
   - Launch the app with `demo.queue().launch(share=True)`.
   - Users can now access it via a shareable link.

### PROGRAM:
```python
import os
from text_generation import Client
import gradio as gr

# Load API key and endpoint
hf_api_key = os.environ['HF_API_KEY']
hf_endpoint = os.environ['HF_API_FALCOM_BASE']

# Initialize the LLM client
client = Client(hf_endpoint, headers={"Authorization": f"Basic {hf_api_key}"}, timeout=120)

# Format the chat prompt with system instruction
def format_chat_prompt(message, chat_history, instruction):
    prompt = f"System: {instruction}"
    for user_message, bot_message in chat_history:
        prompt += f"\nUser: {user_message}\nAssistant: {bot_message}"
    prompt += f"\nUser: {message}\nAssistant:"
    return prompt

# Respond function using streaming from the model
def respond(message, chat_history, instruction, temperature=0.7):
    prompt = format_chat_prompt(message, chat_history, instruction)
    chat_history = chat_history + [[message, ""]]
    
    stream = client.generate_stream(
        prompt,
        max_new_tokens=1024,
        stop_sequences=["\nUser:", "<|endoftext|>"],
        temperature=temperature
    )
    
    acc_text = ""
    for idx, response in enumerate(stream):
        text_token = response.token.text

        if response.details:
            return  # stop if error

        if idx == 0 and text_token.startswith(" "):
            text_token = text_token[1:]

        acc_text += text_token
        last_turn = list(chat_history.pop(-1))
        last_turn[-1] += acc_text
        chat_history = chat_history + [last_turn]

        yield "", chat_history
        acc_text = ""

# Close previous Gradio sessions
gr.close_all()

# Gradio interface
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(label="Prompt")
    
    with gr.Accordion(label="Advanced options", open=False):
        system = gr.Textbox(
            label="System message",
            lines=2,
            value="A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers."
        )
        temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=1.0, value=0.7, step=0.1)

    btn = gr.Button("Submit")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear chat")

    btn.click(respond, inputs=[msg, chatbot, system, temperature], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot, system, temperature], outputs=[msg, chatbot])

# Launch without specifying port to auto-select a free one
demo.queue().launch(share=True)

```

### OUTPUT:

<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/2bcdf8b7-4dcd-4237-8163-305e7d5c0318" />


### RESULT:

The experiment successfully demonstrates a user-friendly web chat interface, enabling real-time, context-aware interaction with a large language model.
