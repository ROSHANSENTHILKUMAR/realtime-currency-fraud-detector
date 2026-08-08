from cartesia import Cartesia
from dotenv import load_dotenv
import os

load_dotenv()
client = Cartesia(api_key=os.getenv("CARTESIA_API_KEY"))

def announce_result(denomination, status, confidence):
    text = f"This is a {denomination} rupee note, {status}, {confidence} percent confidence."
    
    audio_generator = client.tts.bytes(
        model_id="sonic-2",
        transcript=text,
        voice={"mode": "id", "id": "694f9389-aac1-45b6-b726-9d9369183238"},
        output_format={
            "container": "wav",
            "encoding": "pcm_f32le",
            "sample_rate": 44100,
        },
    )
    
    audio_data = b"".join(audio_generator)
    
    with open("output.wav", "wb") as f:
        f.write(audio_data)
    
    print(f"Audio generated: {denomination} rupees, {status}, {confidence}% confidence")

# Test different values
announce_result(500, "genuine", 94)