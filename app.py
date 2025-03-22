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
    
# TOGGER THEME JS
JS_THEME = """
function toggleTheme() {
    var theme = localStorage.getItem("theme") || "dark";
    var isDarkMode = (theme === "dark");
    var gradioContainer = document.querySelector(".gradio-container");
    var targetedSpans = Array.from(document.querySelectorAll("span")).filter(function(span) {
        return span.innerText === "Enable Text-to-Speech";
    });
    if (isDarkMode) {
        gradioContainer.classList.remove("dark-mode");
        gradioContainer.classList.add("light-mode");
        targetedSpans.forEach(function(span) {
            span.classList.remove("dark-mode");
            span.classList.add("light-mode");
        });
        localStorage.setItem("theme", "light");
        document.getElementById("theme-toggle-btn").innerHTML = "☼";
    } else {
        gradioContainer.classList.remove("light-mode");
        gradioContainer.classList.add("dark-mode");
        targetedSpans.forEach(function(span) {
            span.classList.remove("light-mode");
            span.classList.add("dark-mode");
        });
        localStorage.setItem("theme", "dark");
        document.getElementById("theme-toggle-btn").innerHTML = "☾";
    }
}
"""
css_styles = """
.gradio-container.light-mode {
    background-color: #ffffff;
    color: #000000;
}
.gradio-container.dark-mode {
    background-color: #1e1e1e;
    color: #ffffff;
}

/* Inputs always dark */
input, textarea {
    background-color: #2c2f36 !important;
    color: #ffffff !important;
    border: 1px solid #444 !important;
}

/* Markdown and label text */
.light-mode .gr-markdown,
.light-mode .gradio-container .gr-markdown,
.light-mode h1, 
.light-mode h2,
.light-mode h3,
.light-mode h4,
.light-mode h5,
.light-mode p,
.light-mode label {
  color: #000000 !important;
}
.dark-mode h1, 
.dark-mode h2,
.dark-mode h3,
.dark-mode p,
.dark-mode label {
  color: #ffffff !important;
}

/* Tabs (Fix "Nutrition ChatBot" visibility) */
.gradio-container .gr-tab-label {
    opacity: 1 !important;
    color: #000 !important;
}
.dark-mode .gr-tab-label {
    color: #fff !important;
}
.gradio-container .gr-tab-label.selected {
    font-weight: bold;
    border-bottom: 2px solid #22c55e;
}
.typing-text::after {
    content: '|';
    animation: blink 1s step-start infinite;
}

@keyframes blink {
    50% { opacity: 0; }
}
#header-section {
    text-align: center;
    margin-top: 20px;
    margin-bottom: 10px;
}

#logo {
    margin: 0 auto;
    display: block;
}

#intro-banner {
    font-size: 1.5em;
    font-weight: bold;
    color: #22c55e;
}

#toggle-row {
    justify-content: center;
    margin-top: 10px;
}
#header-section {
    text-align: center;
    margin-top: 10px;
    margin-bottom: 0;
}

#logo {
    display: block;
    margin-left: auto;
    margin-right: auto;
    border-radius: 20px;
    box-shadow: 0 0 10px rgba(34, 197, 94, 0.6);
}

#intro-banner {
    font-size: 1.4em;
    font-weight: 800;
    color: #22c55e;
    margin-top: 10px;
    margin-bottom: 10px;
}

#toggle-row {
    justify-content: center;
    align-items: center;
    margin-top: -10px;
    margin-bottom: 20px;
}
.gradio-container {
    background-color: #1e1e1e;
    color: #ffffff;
}
#toggle-row {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 8px;
    margin-top: 15px;
}
"""

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
def chat_with_chatgpt(message, history, lang, file=None):
    if history is None:
        history = []

    file_note = ""
    if file:
        file_name = file.split("/")[-1]
        file_note = f"\n[📎 Attached file: {file_name}]"

    system_prompt = translations[lang]["system_prompt"]

    conversation = [
        {"role": "system", "content": system_prompt}
    ]

    for user_msg, assistant_msg in history:
        conversation.append({"role": "user", "content": user_msg})
        conversation.append({"role": "assistant", "content": assistant_msg})

    full_message = message + file_note
    conversation.append({"role": "user", "content": full_message})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation,
            temperature=0.7,
            max_tokens=150,
        )
        reply = response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        reply = f"Error calling the API: {str(e)}"

    history.append((full_message, reply))
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
        translations[lang]["welcome_form"],  # form_intro
        translations[lang]["step1"],  # step1
        translations[lang]["step2"],  # step2
        translations[lang]["step3"],  # step3
        translations[lang]["step4"],  # step4
        translations[lang]["step5"],  # step5
        translations[lang]["step6"],  # step6
        translations[lang]["step7"],  # step7
        translations[lang]["final"],  # final
        translations[lang]["chatbot_title"],  # chatbot title
        gr.update(placeholder=translations[lang]["msg_placeholder"], label=translations[lang]["msg_label"]),  # msg box
        translations[lang]["send"],  # send btn
        translations[lang]["reset"],  # reset btn
        gr.update(choices=translations[lang]["genre_options"]),  # genre
        gr.update(choices=translations[lang]["activite_options"]),  # activite
        gr.update(choices=translations[lang]["objectif_options"]),  # objectif
        gr.update(label=translations[lang]["name_label"], placeholder=translations[lang]["name_placeholder"]),  # prenom
        gr.update(label=translations[lang]["age_label"], info=translations[lang]["age_info"]),  # age
        gr.update(label=translations[lang]["height_label"], info=translations[lang]["height_info"]),  # taille
        gr.update(label=translations[lang]["weight_label"], info=translations[lang]["weight_info"]),  # poids
        "",  # error age
        "",  # error weight
        "",  # error height
        translations[lang]["voice_input_label"],
        translations[lang]["speak_button"],
        translations[lang]["voice_output_label"],
        translations[lang]["voice_section_header"]
    )
def update_ui_texts(lang):
    return (
        translations[lang]["welcome_form"],  # form_intro
        translations[lang]["final"],         # output_text
        gr.update(placeholder=translations[lang]["msg_placeholder"], label=translations[lang]["msg_label"]),  # msg
        translations[lang]["send"],          # send
        translations[lang]["reset"],         # reset
        gr.update(choices=translations[lang]["genre_options"], label=translations[lang]["gender_label"]),  # genre
        gr.update(choices=translations[lang]["activite_options"], label=translations[lang]["activity_label"]),  # activite
        gr.update(choices=translations[lang]["objectif_options"], label=translations[lang]["objective_label"]),  # objectif
        "",                                  # validation_message
        gr.update(label=translations[lang]["voice_input_label"]),  # voice_input
        gr.update(value=translations[lang]["speak_button"]),       # speak_button
        gr.update(value=translations[lang]["voice_output_label"]), # voice_output
        gr.update(label=translations[lang]["name_label"], placeholder=translations[lang]["name_placeholder"]),  # prenom
        gr.update(label=translations[lang]["age_label"], info=translations[lang]["age_info"]),  # age
        gr.update(label=translations[lang]["height_label"], info=translations[lang]["height_info"]),  # taille
        gr.update(label=translations[lang]["weight_label"], info=translations[lang]["weight_info"]),  # poids
        gr.update(value=translations[lang]["voice_section_header"]),  # voice_header_md
        gr.update(value=translations[lang]["voice_output_label"]),    # ai_response_md
        gr.update(label=translations[lang]["file_upload_label"])      # file_input 
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
    lang = "fr"  # default
    tr = translations[lang]  # shorter alias
    with gr.Blocks(js=JS_THEME, css=css_styles, theme=theme) as demo:

        with gr.Row():
            with gr.Column(scale=1):  # Left spacer
                pass

            with gr.Column(scale=2, elem_id="header-section"):  # Centered logo + intro
                gr.Image("logo.png", elem_id="logo", width=130, interactive=False, show_label=False)
                gr.Markdown(
                    "### Welcome to NutriCoach, your trusted nutrition companion!\n"
                    "Looking for personalized health guidance? You're in the right place.",
                    elem_id="intro-banner"
                )

            with gr.Column(scale=1, elem_id="toggle-row"):  # Language + theme toggle
                lang_dropdown = gr.Dropdown(
                    choices=["fr", "en"],
                    value="fr",
                    interactive=True,
                    show_label=False,
                    elem_id="lang-select"
                )
                theme_toggle = gr.Button("☾", elem_id="theme-toggle-btn")
                lang_state = gr.State("fr")

                theme_toggle.click(
                    fn=lambda: None,
                    inputs=[],
                    outputs=[],
                    js=JS_THEME
                )

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
                    form_intro = gr.Markdown(tr["welcome_form"], elem_id="form-intro")
                    prenom = gr.Textbox(
                        label=tr["name_label"],
                        placeholder=tr["name_placeholder"]
                    )
                    
                    with gr.Row():
                        age = gr.Number(
                            label=tr["age_label"],
                            value=25,
                            minimum=0,
                            info=tr["age_info"]
                        )
                        taille = gr.Number(
                            label=tr["height_label"],
                            value=170,
                            minimum=0,
                            info=tr["height_info"]
                        )
                        poids = gr.Number(
                            label=tr["weight_label"],
                            value=70,
                            minimum=0,
                            info=tr["weight_info"]
                        )

                    genre = gr.Radio(
                        choices=tr["genre_options"],
                        label=tr["gender_label"],
                        value=tr["genre_options"][0]
                    )

                    activite = gr.Radio(
                        choices=tr["activite_options"],
                        label=tr["activity_label"],
                        value=tr["activite_options"][1]
                    )

                    objectif = gr.Radio(
                        choices=tr["objectif_options"],
                        label=tr["objective_label"],
                        value=tr["objectif_options"][0]
                    )

                    
                    validation_message = gr.Markdown("")
                    submit_btn = gr.Button("Save Information", variant="primary")
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
                        send = gr.Button("Send", scale=1, variant="primary")

                    # Add file upload input (below the message box)
                    file_input = gr.File(
                        label=tr["file_upload_label"],
                        type="filepath",
                        file_types=[".pdf", ".txt", ".docx"]
                    )

                    reset = gr.Button("Clear Chat", variant="secondary")
                             
                    # Voice Section
                    voice_header_md = gr.Markdown(tr["voice_section_header"])
                    with gr.Row():
                        voice_input = gr.Audio(
                            type="filepath",
                        )
                        speak_button = gr.Button("Convert to Text", variant="primary")
                    voice_output = gr.Audio()
                    ai_response_md = gr.Markdown(value=tr["voice_output_label"])

        lang_dropdown.change(
            fn=update_ui_texts,
            inputs=lang_dropdown,
            outputs=[
                form_intro, output_text, msg, send, reset,
                genre, activite, objectif, validation_message,
                voice_input, speak_button, voice_output,
                prenom, age, taille, poids,
                voice_header_md, ai_response_md,
                file_input  # <-- this is the 19th item
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
        send.click(
            fn=chat_with_chatgpt,
            inputs=[msg, history, lang_state, file_input],
            outputs=[msg, chatbot, history]
        )

        msg.submit(
            fn=chat_with_chatgpt,
            inputs=[msg, history, lang_state, file_input],
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