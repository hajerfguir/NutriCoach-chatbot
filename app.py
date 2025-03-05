import gradio as gr 
import openai
import yaml

# Load API key from configuration file
with open("config.yaml") as f: 
    config_yaml = yaml.load(f, Loader=yaml.FullLoader)
openai.api_key = config_yaml['token']

# Optional: Test call to GPT‑3.5-turbo (this prints the answer in the console)
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
def conseiller_nutrition(age, genre, poids, taille, activite, objectif):
    """
    Simple function to generate nutrition advice in French.
    Feel free to customize or expand the logic.
    """
    base_fr = "**Voici vos recommandations nutritionnelles personnalisées :**\n\n"
    base_fr += (
        f"- Âge : {age} ans\n"
        f"- Genre : {genre}\n"
        f"- Poids : {poids} kg\n"
        f"- Taille : {taille} cm\n"
        f"- Niveau d'activité : {activite}\n"
        f"- Objectif : {objectif}\n\n"
    )
    
    if objectif == "Perte de poids":
        base_fr += "➡️ Conseils : Maintenez un déficit calorique, privilégiez les protéines maigres et les fibres."
    elif objectif == "Gains musculaires":
        base_fr += "➡️ Conseils : Augmentez votre apport calorique, consommez des protéines de qualité et des glucides complexes."
    else:
        base_fr += "➡️ Conseils : Adoptez une alimentation équilibrée, variée, et restez bien hydraté."
    
    return base_fr

# ---------------------------------------------------------------------------------------
# 2) CHATBOT FUNCTION USING THE OPENAI CHATGPT API
# ---------------------------------------------------------------------------------------
def chat_with_chatgpt(message, history):
    """
    This function builds a conversation context and calls the ChatGPT API.
    It prints the answer to the console and returns the updated conversation history.
    """
    if history is None:
        history = []
    
    # Define the system prompt for the assistant
    conversation = [
        {
            "role": "system",
            "content": "Vous êtes NutriCoach, un nutritionniste expert fournissant des conseils nutritionnels détaillés en français."
        }
    ]
    
    # Include previous conversation history in the context
    for user_msg, assistant_msg in history:
        conversation.append({"role": "user", "content": user_msg})
        conversation.append({"role": "assistant", "content": assistant_msg})
    
    # Append the new user message
    conversation.append({"role": "user", "content": message})
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation,
            temperature=0.7,
            max_tokens=150,
        )
        print(response["choices"][0]["message"]["content"])
        reply = response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        reply = f"Erreur lors de l'appel à l'API: {str(e)}"
    
    # Update conversation history and return outputs for the Gradio Chatbot tab
    history.append((message, reply))
    return "", history, history

# ---------------------------------------------------------------------------------------
# 3) DEFINE THE CUSTOM THEME
# ---------------------------------------------------------------------------------------
custom_theme = gr.themes.Soft(
    primary_hue="green",
    font=["Helvetica Neue", "Arial", "sans-serif"]
)

# ---------------------------------------------------------------------------------------
# 4) MULTI-STEP FORM LOGIC
# ---------------------------------------------------------------------------------------
def go_to_next_step(user_input, step_data, step):
    # Les validations numériques se font dans les fonctions "next_*"
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

def finish_form(step_data):
    age = step_data.get("age", 25)
    genre = step_data.get("genre", "Homme")
    poids = step_data.get("poids", 70)
    taille = step_data.get("taille", 170)
    activite = step_data.get("activite", "Modéré")
    objectif = step_data.get("objectif", "Perte de poids")
    return conseiller_nutrition(age, genre, poids, taille, activite, objectif)

# ---------------------------------------------------------------------------------------
# 5) BUILDING THE GRADIO INTERFACE
# ---------------------------------------------------------------------------------------
def create_interface():
    with gr.Blocks(theme=custom_theme) as demo:
        gr.Image("logo.png", elem_id="logo", show_label=False)
        
        with gr.Tabs():
            # ------------------------------------------------------------------
            # MULTI-STEP FORM TAB
            # ------------------------------------------------------------------
            with gr.Tab("Formulaire personnalisé"):
                gr.Markdown("### Analyse des données personnalisée", elem_id="chatbot-header")
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

                # -----------------------
                # Étape 1: Prénom
                with gr.Group(elem_id="step1", visible=True) as step1_box:
                    gr.Markdown("<p>Quel est votre prénom ?</p>")
                    prenom = gr.Textbox(show_label=False)
                    next_btn_1 = gr.Button("Valider")
                # -----------------------
                # Étape 2: Âge (vérification de la validité)
                with gr.Group(elem_id="step2", visible=False) as step2_box:
                    gr.Markdown("<p>Quel est votre âge ?</p>")
                    age = gr.Number(show_label=False, value=25, minimum=0)
                    error_age = gr.Markdown("", visible=False)
                    next_btn_2 = gr.Button("Valider")
                # -----------------------
                # Étape 3: Genre
                with gr.Group(elem_id="step3", visible=False) as step3_box:
                    gr.Markdown("<p>Quel est votre genre ?</p>")
                    genre = gr.Radio(["Homme", "Femme"], value="Homme", show_label=False)
                    next_btn_3 = gr.Button("Valider")
                # -----------------------
                # Étape 4: Poids (vérification de la validité)
                with gr.Group(elem_id="step4", visible=False) as step4_box:
                    gr.Markdown("<p>Quel est votre poids (kg) ?</p>")
                    poids = gr.Number(show_label=False, value=70, minimum=0)
                    error_poids = gr.Markdown("", visible=False)
                    next_btn_4 = gr.Button("Valider")
                # -----------------------
                # Étape 5: Taille (vérification de la validité)
                with gr.Group(elem_id="step5", visible=False) as step5_box:
                    gr.Markdown("<p>Quelle est votre taille (cm) ?</p>")
                    taille = gr.Number(show_label=False, value=170, minimum=0)
                    error_taille = gr.Markdown("", visible=False)
                    next_btn_5 = gr.Button("Valider")
                # -----------------------
                # Étape 6: Niveau d'activité
                with gr.Group(elem_id="step6", visible=False) as step6_box:
                    gr.Markdown("<p>Quel est votre niveau d'activité ?</p>")
                    activite = gr.Radio(["Sédentaire", "Modéré", "Athlétique"], value="Modéré", show_label=False)
                    next_btn_6 = gr.Button("Valider")
                # -----------------------
                # Étape 7: Objectif
                with gr.Group(elem_id="step7", visible=False) as step7_box:
                    gr.Markdown("<p>Quel est votre objectif principal ?</p>")
                    objectif = gr.Radio(["Perte de poids", "Gains musculaires", "Autre"], value="Perte de poids", show_label=False)
                    next_btn_7 = gr.Button("Valider")
                # -----------------------
                # Étape 8: Affichage des recommandations
                with gr.Group(elem_id="step8", visible=False) as step8_box:
                    gr.Markdown("<p>Merci ! Voici vos recommandations :</p>")
                    done_msg = gr.Markdown("", show_label=False)

                # -----------------------------
                # Fonctions de passage d'étape avec validation
                # -----------------------------
                def next_1(prenom_val, step_data_val, step_val):
                    # Pas de validation spécifique pour le prénom
                    step_data_val, step_val = go_to_next_step(prenom_val, step_data_val, step_val)
                    return step_data_val, step_val, gr.update(visible=False), gr.update(visible=True)
                next_btn_1.click(fn=next_1, inputs=[prenom, step_data, step],
                                 outputs=[step_data, step, step1_box, step2_box])
                
                def next_2(age_val, step_data_val, step_val):
                    if age_val < 0:
                        return step_data_val, step_val, gr.update(visible=True), gr.update(visible=False), "Erreur: L'âge ne peut être négatif."
                    else:
                        step_data_val, step_val = go_to_next_step(age_val, step_data_val, step_val)
                        return step_data_val, step_val, gr.update(visible=False), gr.update(visible=True), ""
                next_btn_2.click(fn=next_2, inputs=[age, step_data, step],
                                 outputs=[step_data, step, step2_box, step3_box, error_age])
                
                def next_3(genre_val, step_data_val, step_val):
                    step_data_val, step_val = go_to_next_step(genre_val, step_data_val, step_val)
                    return step_data_val, step_val, gr.update(visible=False), gr.update(visible=True)
                next_btn_3.click(fn=next_3, inputs=[genre, step_data, step],
                                 outputs=[step_data, step, step3_box, step4_box])
                
                def next_4(poids_val, step_data_val, step_val):
                    if poids_val < 0:
                        return step_data_val, step_val, gr.update(visible=True), gr.update(visible=False), "Erreur: Le poids ne peut être négatif."
                    else:
                        step_data_val, step_val = go_to_next_step(poids_val, step_data_val, step_val)
                        return step_data_val, step_val, gr.update(visible=False), gr.update(visible=True), ""
                next_btn_4.click(fn=next_4, inputs=[poids, step_data, step],
                                 outputs=[step_data, step, step4_box, step5_box, error_poids])
                
                def next_5(taille_val, step_data_val, step_val):
                    if taille_val < 0:
                        return step_data_val, step_val, gr.update(visible=True), gr.update(visible=False), "Erreur: La taille ne peut être négative."
                    else:
                        step_data_val, step_val = go_to_next_step(taille_val, step_data_val, step_val)
                        return step_data_val, step_val, gr.update(visible=False), gr.update(visible=True), ""
                next_btn_5.click(fn=next_5, inputs=[taille, step_data, step],
                                 outputs=[step_data, step, step5_box, step6_box, error_taille])
                
                def next_6(activite_val, step_data_val, step_val):
                    step_data_val, step_val = go_to_next_step(activite_val, step_data_val, step_val)
                    return step_data_val, step_val, gr.update(visible=False), gr.update(visible=True)
                next_btn_6.click(fn=next_6, inputs=[activite, step_data, step],
                                 outputs=[step_data, step, step6_box, step7_box])
                
                def next_7(objectif_val, step_data_val, step_val):
                    step_data_val, step_val = go_to_next_step(objectif_val, step_data_val, step_val)
                    final_text = finish_form(step_data_val)
                    return step_data_val, step_val, final_text, gr.update(visible=False), gr.update(visible=True)
                next_btn_7.click(fn=next_7, inputs=[objectif, step_data, step],
                                 outputs=[step_data, step, done_msg, step7_box, step8_box])
            
            # ------------------------------------------------------------------
            # NUTRITION CHATBOT TAB
            # ------------------------------------------------------------------
            with gr.Tab("Nutrition ChatBot"):
                gr.Markdown("### Discutez avec NutriCoach", elem_id="chatbot-header")
                with gr.Column():
                    chatbot = gr.Chatbot(height=200, elem_id="chatbot-window")
                    state = gr.State([])
                    with gr.Row():
                        msg = gr.Textbox(placeholder="Tapez votre message ici...", label="Vous", elem_id="chat-input")
                    with gr.Row():
                        send = gr.Button("Envoyer", elem_id="send-button")
                        reset = gr.Button("Réinitialiser la conversation", elem_id="reset-button")
                send.click(chat_with_chatgpt, inputs=[msg, state], outputs=[msg, chatbot, state])
                reset.click(lambda: ([], []), outputs=[chatbot, state])
        
        gr.Markdown("<div class='footer' style='text-align: center;'>&copy; 2025 NutriCoach. Tous droits réservés.</div>")
    return demo

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True)
