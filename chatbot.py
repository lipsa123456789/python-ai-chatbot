import streamlit as st
import os
from groq import Groq
import speech_recognition as sr  # Speech-to-text
from gtts import gTTS  # Text-to-speech
import tempfile  # Temporary file handling
import os  # File handling
import time  # Manage audio file deletion timing
from serpapi import GoogleSearch  # Fetch Google search results
from deep_translator import GoogleTranslator  # Language translation
import requests  # For Google Books API
from PIL import Image
import io
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# ============================== API KEYS ==============================
GROQ_API_KEY ="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"   # Replace with your Groq key
SERPAPI_KEY ="fc0f4a8c9838ebb9e5b95aa2f159f3679212655fef2b2ec64c3e14457a3f1f8c"
api_key="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"
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Read API keys from environment variables
#GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#SERPAPI_KEY = os.getenv("SERPAPI_KEY")

## Check if keys are available
#if not GROQ_API_KEY:
#    st.error("GROQ_API_KEY not found. Please add it to your .env file.")
#    st.stop()

#if not SERPAPI_KEY:
#    st.error("SERPAPI_KEY not found. Please add it to your .env file.")
#    st.stop()

    # Replace with your SerpAPI key

# ============================== INIT ==============================
st.set_page_config(page_title="Chatbot", page_icon="🌍", layout="wide")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]

recognizer = sr.Recognizer()
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

# ============================== MODELS (cached) ==============================
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

try:
    blip_processor, blip_model = load_blip_model()
except Exception as e:
    blip_processor, blip_model = None, None
    st.warning(f"Image captioning model failed to load. Error: {e}")

# ============================== UTILS ==============================
def translate_text(text, target_lang="en"):
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception as e:
        return f"Translation error: {e}"

def get_chat_response(client, messages):
    try:
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            temperature=0.7,
            max_tokens=256,
            top_p=0.9,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error occurred: {e}"

def fetch_search_results(query):
    params = {"q": query, "api_key": SERPAPI_KEY}
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        return [{"title": r.get("title"), "link": r.get("link")} for r in results.get("organic_results", [])]
    except Exception as e:
        return f"Error fetching search results: {e}"

def fetch_images_serpapi(query, num_images=6):
    params = {"q": query, "tbm": "isch", "api_key": SERPAPI_KEY}
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        image_results = results.get("images_results", [])
        return [img.get("original", "") for img in image_results[:num_images] if img.get("original")]
    except Exception as e:
        return [f"Error fetching images: {e}"]

def fetch_place_location(place_name):
    params = {"engine": "google_maps", "q": place_name, "api_key": SERPAPI_KEY}
    try:
        search = GoogleSearch(params)
        results = search.get_dict()

        if "place_results" in results and "gps_coordinates" in results["place_results"]:
            gps = results["place_results"]["gps_coordinates"]
            address = results["place_results"].get("address", "Address not available")
            return {"lat": gps.get("latitude"), "lng": gps.get("longitude"), "address": address}

        if "local_results" in results and results["local_results"] and isinstance(results["local_results"], list):
            first_place = results["local_results"][0]
            gps = first_place.get("gps_coordinates", {})
            return {"lat": gps.get("latitude"), "lng": gps.get("longitude"), "address": first_place.get("address", "Address not available")}

        return None
    except Exception as e:
        return f"Error fetching location: {e}"

def fetch_books(query):
    try:
        url = f"https://www.googleapis.com/books/v1/volumes?q={query}"
        response = requests.get(url)
        data = response.json()
        books = []
        if "items" in data:
            for item in data["items"][:5]:
                volume_info = item.get("volumeInfo", {})
                title = volume_info.get("title", "No Title")
                authors = ", ".join(volume_info.get("authors", ["Unknown Author"]))
                link = volume_info.get("infoLink", "#")
                books.append({"title": title, "authors": authors, "link": link})
        return books
    except Exception as e:
        return [{"title": f"Error fetching books: {e}", "authors": "", "link": "#"}]

def play_audio(response, lang="en"):
    try:
        with st.spinner("Converting response to speech..."):
            tts = gTTS(response, lang=lang)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
                temp_audio_path = temp_audio_file.name
                tts.save(temp_audio_path)
                st.audio(temp_audio_path, format="audio/mp3")
            time.sleep(3)
            os.remove(temp_audio_path)
    except Exception as e:
        st.error(f"Audio playback error: {e}")

def caption_image_with_blip(image_bytes: bytes) -> str:
    if blip_processor is None or blip_model is None:
        return "Image captioning model not available."
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = blip_processor(pil_image, return_tensors="pt")
        with torch.no_grad():
            out = blip_model.generate(**inputs, max_new_tokens=50)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"Captioning error: {e}"

def fetch_directions(origin, destination):
    params = {
        "engine": "google_maps",
        "q": f"from {origin} to {destination}",
        "api_key": SERPAPI_KEY
    }
    try:
        search = GoogleSearch(params)
        results = search.get_dict()

        if "directions" in results:
            steps = []
            for step in results["directions"]["routes"][0]["legs"][0]["steps"]:
                txt = step.get("html_instructions", "")
                txt = txt.replace("<b>", "").replace("</b>", "")
                txt = txt.replace("<div style=\"font-size:0.9em\">", " ").replace("</div>", "")
                steps.append(txt)
            return steps
        else:
            return ["No directions found."]
    except Exception as e:
        return [f"Error fetching directions: {e}"]

# ============================== UI ==============================
st.title("🌍 Chatty - AI Chatbot with Context Awareness")

# Language selection
target_lang = st.selectbox(
    "Select translation language:",
    ["en", "hi", "fr", "es", "de", "ta", "te", "bn", "zh-cn", "ja"],
    index=1
)
st.caption("Example: 'hi' = Hindi, 'fr' = French, 'es' = Spanish, 'de' = German, 'ta' = Tamil, 'bn' = Bengali")

# ---------- Voice Chat ----------
st.header("🎙️ Voice Chat")
if st.button("Record Voice"):
    with st.spinner("Listening..."):
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=2)
                st.info("Speak now...")
                audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                user_prompt = recognizer.recognize_google(audio_data)
                st.success(f"Recognized Text: {user_prompt}")

                st.session_state.messages.append({"role": "user", "content": user_prompt})
                response = get_chat_response(client, st.session_state.messages)
                st.session_state.messages.append({"role": "assistant", "content": response})

                translated_response = translate_text(response, target_lang)
                st.write("### Chatbot Response (English):", response)
                st.write(f"### Translated Response ({target_lang}):", translated_response)
                play_audio(translated_response, lang=target_lang)

        except Exception as e:
            st.error(f"An error occurred: {e}")

# ---------- Text Chat ----------
st.header("⌨️ Text Chat")
user_prompt = st.text_area("Type your prompt:", placeholder="Type your question here...")
if st.button("Send"):
    if user_prompt.strip():
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        response = get_chat_response(client, st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": response})

        translated_response = translate_text(response, target_lang)
        st.write("### Chatbot Response (English):", response)
        st.write(f"### Translated Response ({target_lang}):", translated_response)
        play_audio(translated_response, lang=target_lang)

# ---------- Books / Resources ----------
st.header("📚 Books / Resources")
book_query = st.text_input("Enter a topic/course to search for books:")
if st.button("Search Books"):
    books = fetch_books(book_query)
    if books:
        for book in books:
            st.markdown(f"- **{book['title']}** by *{book['authors']}* [Read More]({book['link']})")

# ---------- Location Search ----------
st.header("📍 Location Search")
place_query = st.text_input("Enter a place to search location:")
if st.button("Search Location"):
    location = fetch_place_location(place_query)
    if isinstance(location, dict) and "lat" in location and "lng" in location:
        st.success(f"📍 Location of {place_query}: {location['address']}")
        st.map({"lat": [location["lat"]], "lon": [location["lng"]]})
    elif isinstance(location, str):  # error string
        st.error(location)
    else:
        st.error("Location not found.")

# ---------- Link Search ----------
st.header("🔗 Link Search")
link_query = st.text_input("Enter a query to search for links:")
if st.button("Search Links"):
    search_results = fetch_search_results(link_query)
    if isinstance(search_results, list) and len(search_results) > 0:
        first_result = search_results[0]["link"]
        st.success(f"Redirecting to: {first_result}")
        st.markdown(f'[Click here to open]({first_result})', unsafe_allow_html=True)
    else:
        st.error("No links found.")

# ---------- Image Search ----------
st.header("🖼️ Image Search")
image_query = st.text_input("Enter a keyword to search for images:")
num_images = st.slider("Number of images to fetch:", 1, 12, 6)
if st.button("Search Images"):
    if image_query.strip():
        image_urls = fetch_images_serpapi(image_query, num_images=num_images)
        if image_urls and not image_urls[0].startswith("Error"):
            cols = st.columns(3)
            for i, url in enumerate(image_urls):
                with cols[i % 3]:
                    st.image(url, caption=f"{image_query} {i+1}", use_container_width=True)
        else:
            st.error("Sorry, no images found.")
    else:
        st.warning("Please enter a keyword.")

# ---------- Image Chat (Upload or Camera) ----------
st.header("📷 Image Chat (Upload or Take Photo)")
c1, c2 = st.columns(2)
with c1:
    uploaded_image = st.file_uploader("Upload an image from gallery:", type=["jpg", "jpeg", "png"])
with c2:
    camera_image = st.camera_input("Or take a photo using your camera:")

image_source = uploaded_image or camera_image

if image_source is not None:
    st.image(image_source, caption="Selected Image", use_container_width=True)
    image_bytes = image_source.getvalue()
    with st.spinner("Analyzing image..."):
        caption = caption_image_with_blip(image_bytes)
    st.success(f"📝 Image Caption: {caption}")

    user_question_about_image = st.text_input("Ask a question about this image:")
    if st.button("Analyze Image"):
        if user_question_about_image.strip():
            st.session_state.messages.append(
                {"role": "user", "content": f"Image shows: {caption}. User asks: {user_question_about_image}"}
            )
            response = get_chat_response(client, st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": response})

            translated_response = translate_text(response, target_lang)
            st.write("### Chatbot Response (English):", response)
            st.write(f"### Translated Response ({target_lang}):", translated_response)
            play_audio(translated_response, lang=target_lang)
        else:
            st.warning("Please type a question about the image.")

# ---------- Way / Route Finder ----------
st.header("🛣️ Route Finder")
origin = st.text_input("Enter your current location:")
destination = st.text_input("Enter your destination:")

if st.button("Get Directions"):
    if origin.strip() and destination.strip():
        with st.spinner("Fetching directions..."):
            directions = fetch_directions(origin, destination)
        if directions:
            st.success(f"Directions from **{origin}** to **{destination}**:")
            for i, step in enumerate(directions, 1):
                st.markdown(f"{i}. {step}")
    else:
        st.warning("Please enter both origin and destination.")

# ---------- Reset ----------
if st.button("Reset Chat"):
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]
    st.success("Chat history cleared.")

st.markdown("---")
st.caption("Built with Streamlit, Groq API, SpeechRecognition, gTTS, SerpAPI, Google Books API, Deep Translator, and BLIP (HuggingFace) for image captioning.")
