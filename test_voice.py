from cartesia import Cartesia

client = Cartesia(api_key="sk_car_jCM5vxtkJezWBh5cj5eArT")

text = "This is a 500 rupee note, genuine, 94 percent confidence."

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

# Generator-லேருந்து வர்ற chunks-ஐ join பண்ணி, ஒரே bytes object ஆக்குறோம்
audio_data = b"".join(audio_generator)

with open("output.wav", "wb") as f:
    f.write(audio_data)

print("Audio generated successfully! Check output.wav file.")
# Save audio to a file
with open("output.wav", "wb") as f:
    f.write(audio_data)

print("Audio generated successfully! Check output.wav file.")