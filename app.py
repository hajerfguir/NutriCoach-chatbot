import gradio as gr
import openai
import yaml
import json
from gtts import gTTS
import tempfile
from datetime import datetime

# Load API key from configuration file
with open("config.yaml") as f:
    config_yaml = yaml.load(f, Loader=yaml.FullLoader)
openai.api_key = config_yaml['token']

# Load translation mappings from JSON
with open("translations.json", "r", encoding="utf-8") as f:
    translations = json.load(f)

# Optional: Test call to GPT‑3.5‑turbo (prints answer to console)
messages = [
    {"role": "system", "content": "You are a nutritionist."},
    {"role": "user", "content": "Help out someone reach their nutrition goals as they talk to you. You SHOULD NOT answer any questions that are not related to nutrition."},
]
ans = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    max_tokens=2048,
    messages=messages,
)
print(ans["choices"][0]["message"]["content"])

# ---------------------------------------------------------------------------------------
# 1) TEXTS & LOGIC FOR NUTRITION ADVICE
# ---------------------------------------------------------------------------------------
def conseiller_nutrition(age, genre, poids, taille, activite, objectif, lang="fr"):
    header = translations[lang]["nutrition_recommendation_title"] + "\n\n"
    header += translations[lang]["nutrition_recommendation_age"].format(age=age) + "\n"
    header += translations[lang]["nutrition_recommendation_gender"].format(genre=genre) + "\n"
    header += translations[lang]["nutrition_recommendation_weight"].format(poids=poids) + "\n"
    header += translations[lang]["nutrition_recommendation_height"].format(taille=taille) + "\n"
    header += translations[lang]["nutrition_recommendation_activity"].format(activite=activite) + "\n"
    header += translations[lang]["nutrition_recommendation_objective"].format(objectif=objectif) + "\n\n"
    
    if lang == "fr":
        if objectif == "Perte de poids":
            advice = translations[lang]["advice_weight_loss"]
        elif objectif == "Gains musculaires":
            advice = translations[lang]["advice_muscle_gain"]
        else:
            advice = translations[lang]["advice_other"]
    else:
        if objectif == "Weight Loss":
            advice = translations[lang]["advice_weight_loss"]
        elif objectif == "Muscle Gain":
            advice = translations[lang]["advice_muscle_gain"]
        else:
            advice = translations[lang]["advice_other"]
    
    return header + advice

def finish_form(step_data, lang="fr"):
    age = step_data.get("age", 25)
    genre = step_data.get("genre", translations[lang]["genre_options"][0])
    poids = step_data.get("poids", 70)
    taille = step_data.get("taille", 170)
    activite = step_data.get("activite", translations[lang]["activite_options"][1])
    objectif = step_data.get("objectif", translations[lang]["objectif_options"][0])
    return conseiller_nutrition(age, genre, poids, taille, activite, objectif, lang)

# ---------------------------------------------------------------------------------------
# 2) CHATBOT FUNCTION USING THE OPENAI CHATGPT API (TEXT)
# ---------------------------------------------------------------------------------------
def chat_with_chatgpt(message, history, lang):
    if history is None:
        history = []
    
    system_prompt = translations[lang]["system_prompt"] + "\n\nRemember to encourage regular check-ins and follow-ups. Ask about progress and suggest weekly updates."
    
    conversation = [
        {"role": "system", "content": system_prompt}
    ]
    
    for user_msg, assistant_msg in history:
        conversation.append({"role": "user", "content": user_msg})
        conversation.append({"role": "assistant", "content": assistant_msg})
    
    conversation.append({"role": "user", "content": message})
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation,
            temperature=0.7,
            max_tokens=150,
        )
        reply = response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        reply = f"Erreur lors de l'appel à l'API: {str(e)}"
    
    history.append((message, reply))
    return "", history, history

# ---------------------------------------------------------------------------------------
# NEW: VOICE CHAT FUNCTION USING THE OPENAI WHISPER API AND gTTS FOR TTS
# ---------------------------------------------------------------------------------------
def voice_chat_with_chatgpt(audio_file, history, lang):
    if audio_file is None:
        # No audio input was provided.
        return None, history, history
    try:
        # Transcribe the audio file using OpenAI's Whisper (model "whisper-1")
        with open(audio_file, "rb") as af:
            transcript = openai.Audio.transcribe("whisper-1", af)
        message = transcript["text"].strip()
    except Exception as e:
        message = f"Erreur lors de la transcription audio: {str(e)}"
    
    if history is None:
        history = []
    
    conversation = [
        {"role": "system", "content": translations[lang]["system_prompt"]}
    ]
    for user_msg, assistant_msg in history:
        conversation.append({"role": "user", "content": user_msg})
        conversation.append({"role": "assistant", "content": assistant_msg})
    conversation.append({"role": "user", "content": message})
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation,
            temperature=0.7,
            max_tokens=150,
        )
        reply = response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        reply = f"Erreur lors de l'appel à l'API: {str(e)}"
    
    history.append((message, reply))
    
    # Convert the reply text to speech using gTTS
    try:
        tts = gTTS(reply, lang="fr" if lang=="fr" else "en")
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        voice_response_file = temp_file.name
    except Exception as e:
        voice_response_file = None
    
    return voice_response_file, history, history

# ---------------------------------------------------------------------------------------
# 3) MULTI-STEP FORM LOGIC (INCL. VALIDATION)
# ---------------------------------------------------------------------------------------
def go_to_next_step(user_input, step_data, step):
    if step == 1:
        step_data["prenom"] = user_input
    elif step == 2:
        step_data["age"] = user_input
    elif step == 3:
        step_data["genre"] = user_input
    elif step == 4:
        step_data["poids"] = user_input
    elif step == 5:
        step_data["taille"] = user_input
    elif step == 6:
        step_data["activite"] = user_input
    elif step == 7:
        step_data["objectif"] = user_input
    step += 1
    return step_data, step

def next_1(prenom_val, step_data_val, step_val):
    step_data_val, step_val = go_to_next_step(prenom_val, step_data_val, step_val)
    return step_data_val, step_val, gr.update(visible=False), gr.update(visible=True)

def next_2(age_val, step_data_val, step_val, lang):
    if age_val < 0:
        return step_data_val, step_val, gr.update(visible=True), gr.update(visible=False), translations[lang]["error_age"]
    else:
        step_data_val, step_val = go_to_next_step(age_val, step_data_val, step_val)
        return step_data_val, step_val, gr.update(visible=False), gr.update(visible=True), ""
    
def next_3(genre_val, step_data_val, step_val):
    step_data_val, step_val = go_to_next_step(genre_val, step_data_val, step_val)
    return step_data_val, step_val, gr.update(visible=False), gr.update(visible=True)

def next_4(poids_val, step_data_val, step_val, lang):
    if poids_val < 0:
        return step_data_val, step_val, gr.update(visible=True), gr.update(visible=False), translations[lang]["error_weight"]
    else:
        step_data_val, step_val = go_to_next_step(poids_val, step_data_val, step_val)
        return step_data_val, step_val, gr.update(visible=False), gr.update(visible=True), ""
    
def next_5(taille_val, step_data_val, step_val, lang):
    if taille_val < 0:
        return step_data_val, step_val, gr.update(visible=True), gr.update(visible=False), translations[lang]["error_height"]
    else:
        step_data_val, step_val = go_to_next_step(taille_val, step_data_val, step_val)
        return step_data_val, step_val, gr.update(visible=False), gr.update(visible=True), ""
    
def next_6(activite_val, step_data_val, step_val):
    step_data_val, step_val = go_to_next_step(activite_val, step_data_val, step_val)
    return step_data_val, step_val, gr.update(visible=False), gr.update(visible=True)

def next_7(objectif_val, step_data_val, step_val, chat_history, lang):
    step_data_val, step_val = go_to_next_step(objectif_val, step_data_val, step_val)
    final_text = finish_form(step_data_val, lang)
    name = step_data_val["prenom"]
    age = step_data_val["age"]
    genre = step_data_val["genre"]
    poids = step_data_val["poids"]
    taille = step_data_val["taille"]
    activite = step_data_val["activite"]
    objectif = step_data_val["objectif"]
    greeting = translations[lang]["greeting"].format(
        name=name, age=age, genre=genre, poids=poids, taille=taille, activite=activite, objectif=objectif
    )
    chat_history.append((" ", greeting))
    return step_data_val, step_val, final_text, gr.update(visible=False), gr.update(visible=True), chat_history

# ---------------------------------------------------------------------------------------
# 4) FUNCTION TO UPDATE UI TEXTS BASED ON LANGUAGE SELECTION
# ---------------------------------------------------------------------------------------
def update_ui_texts(lang):
    return (
        translations[lang]["welcome_form"],         # form_intro (Markdown)
        translations[lang]["step1"],                  # step1 text
        translations[lang]["step2"],                  # step2 text
        translations[lang]["step3"],                  # step3 text
        translations[lang]["step4"],                  # step4 text
        translations[lang]["step5"],                  # step5 text
        translations[lang]["step6"],                  # step6 text
        translations[lang]["step7"],                  # step7 text
        translations[lang]["final"],                  # final text
        translations[lang]["chatbot_title"],          # chatbot title
        gr.update(placeholder=translations[lang]["msg_placeholder"], label=translations[lang]["msg_label"]),  # update msg Textbox
        translations[lang]["send"],                   # send button
        translations[lang]["reset"],                  # reset button
        gr.update(choices=translations[lang]["genre_options"]),       # update genre radio
        gr.update(choices=translations[lang]["activite_options"]),      # update activite radio
        gr.update(choices=translations[lang]["objectif_options"]),      # update objectif radio
        "",  # reset error_age text
        "",  # reset error_weight text
        "",  # reset error_height text
        translations[lang]["voice_input_label"],      # voice input label
        translations[lang]["speak_button"],           # speak button text
        translations[lang]["voice_output_label"],     # voice output label
        translations[lang]["voice_section_header"]    # voice section header text
    )

# ---------------------------------------------------------------------------------------
# 5) APPLY GREEN THEME, ADD LOGO, AND BUILD INTERFACE WITH IMPROVED UI LAYOUT
# ---------------------------------------------------------------------------------------
def create_interface():
    # Create a simple green theme
    theme = gr.themes.Soft(
        primary_hue="green",
        secondary_hue="green",
    ).set(
        button_primary_background_fill="*primary_600",
        button_primary_background_fill_hover="*primary_700",
        block_title_text_size="*text_lg",
        block_label_text_size="*text_md",
    )

    with gr.Blocks(theme=theme) as demo:
        # Header with logo
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("# 🥗 Nutrition Coach")
            with gr.Column(scale=2):
                gr.Image("logo.png", show_label=False, width=100)
            with gr.Column(scale=1):
                lang_dropdown = gr.Dropdown(
                    choices=["fr", "en"],
                    value="fr",
                    label="Language / Langue"
                )
                lang_state = gr.State("fr")

        # States
        step_data = gr.State({
            "prenom": "", "age": 25, "genre": "",
            "poids": 70, "taille": 170,
            "activite": "", "objectif": ""
        })
        history = gr.State([])

        with gr.Tabs():
            # Personal Information Tab
            with gr.Tab("📋 Personal Information"):
                with gr.Column(scale=1, min_width=600):  # Replace Box with Column
                    form_intro = gr.Markdown(translations["fr"]["welcome_form"])
                    
                    prenom = gr.Textbox(
                        label="Name",
                        placeholder="Enter your name"
                    )
                    
                    with gr.Row():
                        age = gr.Number(
                            label="Age",
                            value=25,
                            minimum=0,
                            info="Must be between 16 and 100 years"
                        )
                        taille = gr.Number(
                            label="Height (cm)",
                            value=170,
                            minimum=0,
                            info="Height in centimeters"
                        )
                        poids = gr.Number(
                            label="Weight (kg)",
                            value=70,
                            minimum=0,
                            info="Weight in kilograms"
                        )
                    
                    genre = gr.Radio(
                        choices=translations["fr"]["genre_options"],
                        label="Gender",
                        value=translations["fr"]["genre_options"][0]
                    )
                    
                    activite = gr.Radio(
                        choices=translations["fr"]["activite_options"],
                        label="Activity Level",
                        value=translations["fr"]["activite_options"][1]
                    )
                    
                    objectif = gr.Radio(
                        choices=translations["fr"]["objectif_options"],
                        label="Objective",
                        value=translations["fr"]["objectif_options"][0]
                    )
                    
                    validation_message = gr.Markdown("")
                    submit_btn = gr.Button("💾 Save Information", variant="primary")
                    output_text = gr.Markdown("")

            # Chatbot Tab
            with gr.Tab("💬 Nutrition ChatBot"):
                with gr.Column(scale=1, min_width=600):  # Replace Box with Column
                    chatbot = gr.Chatbot(
                        height=400,
                        bubble_full_width=False,
                        show_label=False
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Ask me anything about nutrition...",
                            label="Your message",
                            scale=4
                        )
                        send = gr.Button("Send 📤", scale=1, variant="primary")
                    
                    reset = gr.Button("Clear Chat 🗑️", variant="secondary")
                    
                    # Voice Section
                    gr.Markdown("## 🎤 Voice Interaction")
                    with gr.Row():
                        voice_input = gr.Audio(
                            type="filepath",
                            label="Record your message"
                        )
                        speak_button = gr.Button("Convert to Text 🗣️", variant="primary")
                    voice_output = gr.Audio(label="AI Response")

        # Event handlers
        lang_dropdown.change(
            fn=update_ui_texts,
            inputs=lang_dropdown,
            outputs=[
                form_intro, output_text, msg, send, reset,
                genre, activite, objectif, validation_message,
                voice_input, speak_button, voice_output
            ]
        ).then(
            lambda lang: lang,
            inputs=lang_dropdown,
            outputs=lang_state
        )
        
        submit_btn.click(
            fn=update_user_info,
            inputs=[step_data, history, lang_state],
            outputs=[output_text, history]
        )

        # Chat functionality
        msg.submit(
            fn=chat_with_chatgpt,
            inputs=[msg, history, lang_state],
            outputs=[msg, chatbot, history]
        )
        
        send.click(
            fn=chat_with_chatgpt,
            inputs=[msg, history, lang_state],
            outputs=[msg, chatbot, history]
        )
        
        reset.click(
            lambda: ([], []),
            outputs=[chatbot, history]
        )
        
        speak_button.click(
            fn=voice_chat_with_chatgpt,
            inputs=[voice_input, history, lang_state],
            outputs=[voice_output, history, chatbot]
        )

        # Update form data when inputs change
        for input_component in [prenom, age, genre, poids, taille, activite, objectif]:
            input_component.change(
                fn=lambda *args: {
                    "prenom": args[0],
                    "age": args[1],
                    "genre": args[2],
                    "poids": args[3],
                    "taille": args[4],
                    "activite": args[5],
                    "objectif": args[6]
                },
                inputs=[prenom, age, genre, poids, taille, activite, objectif],
                outputs=[step_data]
            )
        
    return demo

def validate_inputs(weight, target_weight, height, age):
    messages = []
    
    # Basic health checks
    if weight < 30 or weight > 300:
        messages.append("⚠️ The entered current weight seems unrealistic. Please verify.")
    
    if target_weight < 40 or target_weight > 200:
        messages.append("⚠️ The target weight seems unrealistic. Please consult a healthcare professional.")
    
    if abs(target_weight - weight) > weight * 0.3:
        messages.append("⚠️ Attempting to change weight by more than 30% may be unsafe. Please consult a healthcare professional.")
    
    if height < 120 or height > 250:
        messages.append("⚠️ The entered height seems unrealistic. Please verify.")
    
    if age < 16 or age > 100:
        messages.append("⚠️ This application is designed for users between 16 and 100 years old.")
    
    return messages

def get_chat_response(user_info, chat_history):
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    system_prompt = """You are a friendly and knowledgeable nutrition coach. Provide personalized advice based on the user's information.
    Always encourage healthy, sustainable habits. Include specific meal suggestions and exercise tips.
    End your response with a reminder to check back in a week for progress tracking.
    Keep responses concise but informative."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"""User Info (Date: {current_date}):
        - Current Weight: {user_info['weight']}kg
        - Target Weight: {user_info['target_weight']}kg
        - Height: {user_info['height']}cm
        - Age: {user_info['age']}
        - Gender: {user_info['gender']}
        - Activity Level: {user_info['activity_level']}
        - Diet Preference: {user_info['diet_preference']}
        
        Previous chat history: {chat_history}
        
        Provide personalized nutrition and exercise advice."""}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=500,
        temperature=0.7
    )
    
    return response.choices[0].message['content']

def update_chat(weight, target_weight, height, age, gender, activity_level, diet_preference, chat_history):
    # Validate inputs first
    validation_messages = validate_inputs(weight, target_weight, height, age)
    
    if validation_messages:
        return chat_history + "\n\n" + "\n".join(validation_messages)
    
    # Prepare user info dictionary
    user_info = {
        "weight": weight,
        "target_weight": target_weight,
        "height": height,
        "age": age,
        "gender": gender,
        "activity_level": activity_level,
        "diet_preference": diet_preference
    }
    
    # Get response from ChatGPT
    response = get_chat_response(user_info, chat_history)
    
    # Update chat history
    new_chat_history = f"{chat_history}\n\nNutrition Coach: {response}" if chat_history else f"Nutrition Coach: {response}"
    
    return new_chat_history

def validate_form_data(data):
    messages = []
    
    # Weight validation
    if data["poids"] < 30 or data["poids"] > 300:
        messages.append("⚠️ The entered weight seems unrealistic. Please verify.")
    
    # Height validation
    if data["taille"] < 120 or data["taille"] > 250:
        messages.append("⚠️ The entered height seems unrealistic. Please verify.")
    
    # Age validation
    if data["age"] < 16 or data["age"] > 100:
        messages.append("⚠️ This application is designed for users between 16 and 100 years old.")
    
    return messages

def update_user_info(form_data, history, lang):
    validation_messages = validate_form_data(form_data)
    
    if validation_messages:
        return "\n".join(validation_messages), history
    
    final_text = finish_form(form_data, lang)
    name = form_data["prenom"]
    age = form_data["age"]
    genre = form_data["genre"]
    poids = form_data["poids"]
    taille = form_data["taille"]
    activite = form_data["activite"]
    objectif = form_data["objectif"]
    
    greeting = translations[lang]["greeting"].format(
        name=name, age=age, genre=genre, poids=poids, 
        taille=taille, activite=activite, objectif=objectif
    )
    
    # Clear previous history and start fresh with updated info
    history.clear()
    history.append((" ", greeting))
    
    return final_text, history

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True)
