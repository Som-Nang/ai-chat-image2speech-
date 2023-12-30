import os
import requests
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI

load_dotenv(find_dotenv())

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(url)[0]["generated_text"]
    print("Image to Text:", text)
    return text

def generate_story(scenario):
    template = """ You are a story teller;
    You can generate a simple story based on a simple narrative, the story should be not more than 20 words;
    
    CONTEXT: {scenario}
    STORY:
    """

    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    story_llm = LLMChain(llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1), prompt=prompt, verbose=True)
    story = story_llm.predict(scenario=scenario)
    print("Generated Story:", story)
    return story

def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/microsoft/speecht5_tts"
    headers = {"Authorization": "Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payloads = {"inputs": message}
    
    response = requests.post(API_URL, headers=headers, json=payloads)
    
    with open('audio.flac', 'wb') as file:
        file.write(response.content)
    
#     print("Text to Speech: Audio file generated")

# # Example usage:
# scenario = img2text("12.jpg")
# story = generate_story(scenario)
# text2speech(story)
def main():
    st.set_page_config(page_title= "img 2 audio story", page_icon="😂")

    st.header("Turn img into audio story")
    uploaded_file = st.file_uploader("Choose an img...", type="jpg")

    if uploaded_file is not None: 
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption='Uploaded Img. ', 
                 use_column_width=True)
        scenario = img2text(uploaded_file.name)
        story = generate_story(scenario)
        text2speech(story)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)

            st.audio("audio.flac")

if __name__ == '__main__':
    main()
