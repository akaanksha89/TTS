import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Device and model/tokenizer loading
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = ParlerTTSForConditionalGeneration.from_pretrained("models/indic-parler-tts").to(device)
tokenizer = AutoTokenizer.from_pretrained("models/indic-parler-tts")
description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

# Language and speaker mapping based on provided table
language_speakers = {
    "assamese": {
         "available": ["Amit", "Sita", "Poonam", "Rakesh"],
         "recommended": ["Amit", "Sita"]
    },
    "bengali": {
         "available": ["Arjun", "Aditi", "Tapan", "Rashmi", "Arnav", "Riya"],
         "recommended": ["Arjun", "Aditi"]
    },
    "bodo": {
         "available": ["Bikram", "Maya", "Kalpana"],
         "recommended": ["Bikram", "Maya"]
    },
    "chhattisgarhi": {
         "available": ["Bhanu", "Champa"],
         "recommended": ["Bhanu", "Champa"]
    },
    "dogri": {
         "available": ["Karan"],
         "recommended": ["Karan"]
    },
    "english": {
         "available": ["Thoma", "Mary", "Swapna", "Dinesh", "Meera", "Jatin", "Aakash", "Sneha", 
                       "Kabir", "Tisha", "Chingkhei", "Thoiba", "Priya", "Tarun", "Gauri", "Nisha", 
                       "Raghav", "Kavya", "Ravi", "Vikas", "Riya"],
         "recommended": ["Thoma", "Mary"]
    },
    "gujarati": {
         "available": ["Yash", "Neha"],
         "recommended": ["Yash", "Neha"]
    },
    "hindi": {
         "available": ["Rohit", "Divya", "Aman", "Rani"],
         "recommended": ["Rohit", "Divya"]
    },
    "kannada": {
         "available": ["Suresh", "Anu", "Chetan", "Vidya"],
         "recommended": ["Suresh", "Anu"]
    },
    "malayalam": {
         "available": ["Anjali", "Anju", "Harish"],
         "recommended": ["Anjali", "Harish"]
    },
    "manipuri": {
         "available": ["Laishram", "Ranjit"],
         "recommended": ["Laishram", "Ranjit"]
    },
    "marathi": {
         "available": ["Sanjay", "Sunita", "Nikhil", "Radha", "Varun", "Isha"],
         "recommended": ["Sanjay", "Sunita"]
    },
    "nepali": {
         "available": ["Amrita"],
         "recommended": ["Amrita"]
    },
    "odia": {
         "available": ["Manas", "Debjani"],
         "recommended": ["Manas", "Debjani"]
    },
    "punjabi": {
         "available": ["Divjot", "Gurpreet"],
         "recommended": ["Divjot", "Gurpreet"]
    },
    "sanskrit": {
         "available": ["Aryan"],
         "recommended": ["Aryan"]
    },
    "tamil": {
         "available": ["Kavitha", "Jaya"],
         "recommended": ["Jaya"]
    },
    "telugu": {
         "available": ["Prakash", "Lalitha", "Kiran"],
         "recommended": ["Prakash", "Lalitha"]
    }
}

# Speaker description examples based on provided data.
speaker_descriptions = {
    "Aditi": (
        "Aditi speaks with a slightly higher pitch in a close-sounding environment. "
        "Her voice is clear, with subtle emotional depth and a normal pace, all captured in high-quality recording."
    ),
    "Sita": (
        "Sita speaks at a fast pace with a slightly low-pitched voice, captured clearly in a close-sounding environment "
        "with excellent recording quality."
    ),
    "Tapan": (
        "Tapan speaks at a moderate pace with a slightly monotone tone. The recording is clear, with a close sound and only minimal ambient noise."
    ),
    "Sunita": (
        "Sunita speaks with a high pitch in a close environment. Her voice is clear, with slight dynamic changes, "
        "and the recording is of excellent quality."
    ),
    "Karan": (
        "Karan’s high-pitched, engaging voice is captured in a clear, close-sounding recording. "
        "His slightly slower delivery conveys a positive tone."
    ),
    "Amrita": (
        "Amrita speaks with a high pitch at a slow pace. Her voice is clear, with excellent recording quality "
        "and only moderate background noise."
    ),
    "Bikram": (
        "Bikram speaks with a higher pitch and fast pace, conveying urgency. The recording is clear and intimate, "
        "with great emotional depth."
    ),
    "Anjali": (
        "Anjali speaks with a high pitch at a normal pace in a clear, close-sounding environment. "
        "Her neutral tone is captured with excellent audio quality."
    )
}

def generate_audio_with_description(description_text: str, prompt: str) -> str:
    """
    Generate TTS audio based on a speaker description and the prompt text.
    """
    # Tokenize the provided description and prompt.
    description_input_ids = description_tokenizer(description_text, return_tensors="pt").to(device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate audio using the TTS model.
    with torch.no_grad():
        generation = model.generate(
            input_ids=description_input_ids.input_ids,
            attention_mask=description_input_ids.attention_mask,
            prompt_input_ids=prompt_input_ids.input_ids,
            prompt_attention_mask=prompt_input_ids.attention_mask,
            num_beams=1,         # Fewer beams for faster generation
            early_stopping=True,
        )
    
    # Convert tensor output to numpy array and save as a WAV file.
    audio_arr = generation.cpu().numpy().squeeze()
    file_path = "output.wav"
    sf.write(file_path, audio_arr, model.config.sampling_rate)
    
    return file_path

# Define the input model for the TTS generation API endpoint.
class TTSInput(BaseModel):
    language: str
    speaker: str
    text: str

app = FastAPI()

@app.post("/generate-tts/")
async def generate_tts_endpoint(tts_input: TTSInput):
    language_lower = tts_input.language.lower()
    speaker = tts_input.speaker.strip()
    
    # Validate language support.
    if language_lower not in language_speakers:
        logger.error("Language %s is not supported.", tts_input.language)
        raise HTTPException(status_code=400, detail=f"Language '{tts_input.language}' is not supported.")
    
    # Validate if the speaker is available for the language.
    available_speakers = language_speakers[language_lower]["available"]
    if speaker not in available_speakers:
        logger.error("Speaker %s is not available for language %s.", speaker, tts_input.language)
        raise HTTPException(
            status_code=400,
            detail=f"Speaker '{speaker}' is not available for language '{tts_input.language}'. "
                   f"Available speakers: {', '.join(available_speakers)}"
        )
    
    # Get the speaker's description or use a generic default.
    description_text = speaker_descriptions.get(
        speaker,
        f"{speaker} speaks with a natural tone in a clear, high-quality recording."
    )
    
    logger.debug("Generating audio with language: %s, speaker: %s, description: %s, text: %s",
                 tts_input.language, speaker, description_text, tts_input.text)
    
    file_path = generate_audio_with_description(description_text, tts_input.text)
    
    # Instead of returning the file path, return the file as binary.
    return FileResponse(file_path, media_type="audio/wav", filename="output.wav")

@app.get("/supported-voices/")
async def get_supported_voices():
    """
    Returns a list of supported languages along with their available and recommended speakers.
    """
    supported = []
    for language, data in language_speakers.items():
        supported.append({
            "language": language.capitalize(),
            "available_speakers": data["available"],
            "recommended_speakers": data["recommended"]
        })
    return {"supported_voices": supported}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)