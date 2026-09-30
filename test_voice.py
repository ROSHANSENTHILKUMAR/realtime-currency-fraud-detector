from cartesia import Cartesia
from groq import Groq
from dotenv import load_dotenv
import os

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

cartesia_client = Cartesia(api_key=os.getenv("CARTESIA_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_announcement(status, confidence):
    """Use Groq to turn raw result into a natural spoken sentence."""
    prompt = f"Write one short, natural spoken sentence announcing that a currency note was detected as {status} with {confidence} percent confidence. Keep it under 15 words."

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def announce_result(status, confidence):
    text = generate_announcement(status, confidence)
    print(f"Generated text: {text}")

    audio_generator = cartesia_client.tts.bytes(
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

    output_path = os.path.join(os.path.dirname(__file__), "output.wav")
    with open(output_path, "wb") as f:
        f.write(audio_data)

    print(f"Audio generated: {status}, {confidence}% confidence")

# Quick test
if __name__ == "__main__":
    announce_result("genuine", 94)