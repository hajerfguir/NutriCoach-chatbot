import gradio as gr
import openai
import yaml
import json
from gtts import gTTS
import tempfile

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
    {"role": "user", "content": "Help out someone reach their nutrition goals as they talk to you."},
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
        translations[lang]["voice_output_label"]      # voice output label
    )

# ---------------------------------------------------------------------------------------
# 5) APPLY GREEN THEME, ADD LOGO, AND BUILD INTERFACE
# ---------------------------------------------------------------------------------------
def create_interface():
    custom_theme = gr.themes.Soft(
        primary_hue="green",
        font=["Helvetica Neue", "Arial", "sans-serif"]
    )

    with gr.Blocks(theme=custom_theme) as demo:
        # Add logo at the top (ensure "logo.png" is in the same folder)
        gr.Image("logo.png", show_label=False, elem_id="logo", width=20)
        
        # Language selection row and state
        with gr.Row():
            lang_dropdown = gr.Dropdown(choices=["fr", "en"], value="fr", label="Select Language / Choisir la langue")
            lang_state = gr.State("fr")
        
        # Global states for multi‑step form and ChatBot conversation
        step = gr.State(1)
        step_data = gr.State({
            "prenom": "",
            "age": 0,
            "genre": "",
            "poids": 0,
            "taille": 0,
            "activite": "",
            "objectif": ""
        })
        history = gr.State([])

        with gr.Tabs():
            # ---------------------------
            # TAB 1: PERSONALIZED FORM
            # ---------------------------
            with gr.Tab("Formulaire personnalisé"):
                # Place the introduction text inside the form tab
                form_intro = gr.Markdown(translations["fr"]["welcome_form"])
                with gr.Row():
                    with gr.Column():
                        # STEP 1
                        with gr.Group(visible=True) as step1_box:
                            step1_md = gr.Markdown(translations["fr"]["step1"])
                            prenom = gr.Textbox(show_label=False)
                            next_btn_1 = gr.Button("Valider")
                        # STEP 2
                        with gr.Group(visible=False) as step2_box:
                            step2_md = gr.Markdown(translations["fr"]["step2"])
                            age = gr.Number(show_label=False, value=25, minimum=0)
                            error_age = gr.Markdown("", visible=False)
                            next_btn_2 = gr.Button("Valider")
                        # STEP 3
                        with gr.Group(visible=False) as step3_box:
                            step3_md = gr.Markdown(translations["fr"]["step3"])
                            genre = gr.Radio(choices=translations["fr"]["genre_options"], value=translations["fr"]["genre_options"][0], show_label=False)
                            next_btn_3 = gr.Button("Valider")
                        # STEP 4
                        with gr.Group(visible=False) as step4_box:
                            step4_md = gr.Markdown(translations["fr"]["step4"])
                            poids = gr.Number(show_label=False, value=70, minimum=0)
                            error_poids = gr.Markdown("", visible=False)
                            next_btn_4 = gr.Button("Valider")
                        # STEP 5
                        with gr.Group(visible=False) as step5_box:
                            step5_md = gr.Markdown(translations["fr"]["step5"])
                            taille = gr.Number(show_label=False, value=170, minimum=0)
                            error_taille = gr.Markdown("", visible=False)
                            next_btn_5 = gr.Button("Valider")
                        # STEP 6
                        with gr.Group(visible=False) as step6_box:
                            step6_md = gr.Markdown(translations["fr"]["step6"])
                            activite = gr.Radio(choices=translations["fr"]["activite_options"], value=translations["fr"]["activite_options"][1], show_label=False)
                            next_btn_6 = gr.Button("Valider")
                        # STEP 7
                        with gr.Group(visible=False) as step7_box:
                            step7_md = gr.Markdown(translations["fr"]["step7"])
                            objectif = gr.Radio(choices=translations["fr"]["objectif_options"], value=translations["fr"]["objectif_options"][0], show_label=False)
                            next_btn_7 = gr.Button("Valider")
                        # STEP 8 (final result)
                        with gr.Group(visible=False) as step8_box:
                            final_md = gr.Markdown(translations["fr"]["final"])
                            done_msg = gr.Markdown("", show_label=False)
                        
                        # STEP FUNCTIONS (linking steps)
                        next_btn_1.click(
                            fn=next_1, 
                            inputs=[prenom, step_data, step],
                            outputs=[step_data, step, step1_box, step2_box]
                        )
                        next_btn_2.click(
                            fn=next_2, 
                            inputs=[age, step_data, step, lang_state],
                            outputs=[step_data, step, step2_box, step3_box, error_age]
                        )
                        next_btn_3.click(
                            fn=next_3, 
                            inputs=[genre, step_data, step],
                            outputs=[step_data, step, step3_box, step4_box]
                        )
                        next_btn_4.click(
                            fn=next_4, 
                            inputs=[poids, step_data, step, lang_state],
                            outputs=[step_data, step, step4_box, step5_box, error_poids]
                        )
                        next_btn_5.click(
                            fn=next_5, 
                            inputs=[taille, step_data, step, lang_state],
                            outputs=[step_data, step, step5_box, step6_box, error_taille]
                        )
                        next_btn_6.click(
                            fn=next_6, 
                            inputs=[activite, step_data, step],
                            outputs=[step_data, step, step6_box, step7_box]
                        )
                        next_btn_7.click(
                            fn=next_7, 
                            inputs=[objectif, step_data, step, history, lang_state],
                            outputs=[step_data, step, done_msg, step7_box, step8_box, history]
                        )
            # ---------------------------
            # TAB 2: NUTRITION CHATBOT
            # ---------------------------
            with gr.Tab("Nutrition ChatBot"):
                chatbot_title_md = gr.Markdown(translations["fr"]["chatbot_title"])
                chatbot = gr.Chatbot(height=200)
                with gr.Row():
                    msg = gr.Textbox(placeholder=translations["fr"]["msg_placeholder"], label=translations["fr"]["msg_label"])
                with gr.Row():
                    send = gr.Button(translations["fr"]["send"])
                    reset = gr.Button(translations["fr"]["reset"])
                send.click(
                    fn=chat_with_chatgpt,
                    inputs=[msg, history, lang_state],
                    outputs=[msg, chatbot, history]
                )
                reset.click(lambda: ([], []), outputs=[chatbot, history])
                
                # ---------------------------
                # NEW: VOICE CHAT SECTION
                # ---------------------------
                with gr.Row():
                    voice_input = gr.Audio(type="filepath", label=translations["fr"]["voice_input_label"])
                with gr.Row():
                    speak_button = gr.Button(translations["fr"]["speak_button"])
                with gr.Row():
                    voice_output = gr.Audio(label=translations["fr"]["voice_output_label"])
                speak_button.click(
                    fn=voice_chat_with_chatgpt,
                    inputs=[voice_input, history, lang_state],
                    outputs=[voice_output, history, history]
                )
        
        # When the user changes the language, update all texts and radio choices.
        lang_dropdown.change(
            fn=update_ui_texts,
            inputs=lang_dropdown,
            outputs=[
                form_intro,         # update form intro (Markdown)
                step1_md,           # step1 markdown
                step2_md,           # step2 markdown
                step3_md,           # step3 markdown
                step4_md,           # step4 markdown
                step5_md,           # step5 markdown
                step6_md,           # step6 markdown
                step7_md,           # step7 markdown
                final_md,           # final markdown
                chatbot_title_md,   # chatbot tab title
                msg,                # update msg Textbox (placeholder and label)
                send,               # send button text
                reset,              # reset button text
                genre,              # update genre radio choices
                activite,           # update activite radio choices
                objectif,           # update objectif radio choices
                error_age,          # reset error_age
                error_poids,        # reset error_weight
                error_taille,       # reset error_height
                voice_input,        # update voice input label
                speak_button,       # update speak button text
                voice_output        # update voice output label
            ]
        ).then(
            lambda lang: lang,
            inputs=lang_dropdown,
            outputs=lang_state
        )
        
    return demo

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True)
