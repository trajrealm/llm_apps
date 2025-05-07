import airline_csa as csa
import base64
from io import BytesIO
from PIL import Image

def artist(city):
    image_response = csa.openai.images.generate(
           model="dall-e-3",
            prompt=f"An image representing a vacation in {city}, showing tourist spots and everything unique about {city}, in a vibrant pop-art style",
            size="1024x1024",
            n=1,
            response_format="b64_json",
    )
    image_base64 = image_response.data[0].b64_json
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data))

image = artist("Kathmandu, Nepal")
image.show()

def chat(history):
    messages = [{"role": "system", "content": csa.system_message}] + history
    response = csa.openai.chat.completions.create(model=csa.MODEL, messages=messages, tools=csa.tools)
    image = None
    
    if response.choices[0].finish_reason=="tool_calls":
        message = response.choices[0].message
        response, city = csa.handle_tool_call(message)
        messages.append(message)
        messages.append(response)
        image = artist(city)
        response = csa.openai.chat.completions.create(model=csa.MODEL, messages=messages)
        
    reply = response.choices[0].message.content
    history += [{"role":"assistant", "content":reply}]

    # Comment out or delete the next line if you'd rather skip Audio for now..
    # talker(reply)
    
    return history, image

# More involved Gradio code as we're not using the preset Chat interface!
# Passing in inbrowser=True in the last line will cause a Gradio window to pop up immediately.

with csa.gr.Blocks() as ui:
    with csa.gr.Row():
        chatbot = csa.gr.Chatbot(height=500, type="messages")
        image_output = csa.gr.Image(height=500)
    with csa.gr.Row():
        entry = csa.gr.Textbox(label="Chat with our AI Assistant:")
    with csa.gr.Row():
        clear = csa.gr.Button("Clear")

    def do_entry(message, history):
        history += [{"role":"user", "content":message}]
        return "", history

    entry.submit(do_entry, inputs=[entry, chatbot], outputs=[entry, chatbot]).then(
        chat, inputs=chatbot, outputs=[chatbot, image_output]
    )
    clear.click(lambda: None, inputs=None, outputs=chatbot, queue=False)

ui.launch(inbrowser=True)