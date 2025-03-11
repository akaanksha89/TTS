import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import logging
from fastapi import FastAPI
from pydantic import BaseModel

# Load the model and tokenizers once at startup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = ParlerTTSForConditionalGeneration.from_pretrained("models/indic-parler-tts").to(device)
tokenizer = AutoTokenizer.from_pretrained("models/indic-parler-tts")
description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

# Preprocess the description once at startup
description_input_ids = description_tokenizer(
    "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up.",
    return_tensors="pt"
).to(device)

def generate_audio(prompt):
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        generation = model.generate(
            input_ids=description_input_ids.input_ids,
            attention_mask=description_input_ids.attention_mask,
            prompt_input_ids=prompt_input_ids.input_ids,
            prompt_attention_mask=prompt_input_ids.attention_mask,
            num_beams=1,  # Use fewer beams for faster generation
            early_stopping=True,
        )
    audio_arr = generation.cpu().numpy().squeeze()

    # Save the audio to a file
    file_path = "output.wav"
    sf.write(file_path, audio_arr, model.config.sampling_rate)

    return file_path

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/generate-audio/")
async def generate_audio_endpoint(text_input: TextInput):
    logger.debug("Received request with text: %s", text_input.text)
    prompt = text_input.text
    file_path = generate_audio(prompt)

    response = {"audio_file_path": file_path}
    
    logger.debug("Response: %s", response)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
