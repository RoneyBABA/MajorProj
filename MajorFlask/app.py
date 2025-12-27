# -----------------------------------------
# Flask Server for AI Doctor (Voice + Vision)
# -----------------------------------------

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
load_dotenv()

from model import encode_image, analyze_image_with_query, analyze_query
from patient import transcription

# Load API Key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if GROQ_API_KEY is None:
    raise ValueError("GROQ_API_KEY is not set! Add it to .env or environment variables.")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# -----------------------------------------
# System Prompt for Doctor
# -----------------------------------------
system_prompt = """
You are a professional doctor. Given input is the query of patient.
What's in this image (if provided)?. Do you find anything wrong with it medically?
Suggest some quick response actions, which can be implemented immediately. Do not add any numbers or special characters in
your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
Donot say 'In the image I see' but say 'With what I see, I think you have ....'
Do end the response with the specialist (ex:urologist, cardiologist) the user should consult and it strictly should be the very last word of the response.
Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot.
Keep your answer concise (max 2 sentences). No preamble, start your answer right away please.
"""


# -----------------------------------------
# Helper Function
# -----------------------------------------
def process_inputs(audio_filepath, image_filepath=None):

    speech_to_text_output = transcription(
        GROQ_API_KEY=GROQ_API_KEY,
        audio_filepath=audio_filepath,
        stt_model="whisper-large-v3"
    )

    full_query = system_prompt + speech_to_text_output

    if image_filepath is None:
        doctor_response = analyze_query(
            query=full_query,
            model="meta-llama/llama-4-scout-17b-16e-instruct"
        )
    else:
        encoded_img = encode_image(image_filepath)
        doctor_response = analyze_image_with_query(
            query=full_query,
            encoded_image=encoded_img,
            model="meta-llama/llama-4-scout-17b-16e-instruct"
        )

    return speech_to_text_output, doctor_response


@app.route("/", methods=["GET"])
def endpoint():
    return "Server runnig"



@app.route("/process", methods=["POST"])
def process_api():
    """
    Required:
        audio: Audio file (.wav, .mp3 etc.)

    Optional:
        image: Image file (.jpg, .png etc.)
    """

    audio = request.files.get("audio")
    image = request.files.get("image")

    if audio is None:
        return jsonify({"error": "Audio file is required!"}), 400

    audio_path = "temp_audio.wav"
    audio.save(audio_path)

    image_path = None
    if image:
        image_path = "temp_image.jpg"
        image.save(image_path)

    # Process through your pipeline
    stt_output, doctor_response = process_inputs(audio_path, image_path)

    # Remove saved temporary files just before sending the response
    try:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
    except Exception as e:
        print(f"Warning: failed to remove audio file {audio_path}: {e}")

    try:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
    except Exception as e:
        print(f"Warning: failed to remove image file {image_path}: {e}")

    return jsonify({
        "speech_to_text": stt_output,
        "doctor_response": doctor_response
    })



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)