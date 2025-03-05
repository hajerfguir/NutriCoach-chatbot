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
        base_fr += "➡️ Conseils : Maintenez un déficit calorique, privilégiez les protéines maigres et les fibres. Pour plus de détails, essayez de parler au ChatBot!"
    elif objectif == "Gains musculaires":
        base_fr += "➡️ Conseils : Augmentez votre apport calorique, consommez des protéines de qualité et des glucides complexes. Pour plus de détails, essayez de parler au ChatBot!"
    else:
        base_fr += "➡️ Conseils : Adoptez une alimentation équilibrée, variée, et restez bien hydraté. Pour plus de détails, essayez de parler au ChatBot!"
    
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
            "content": "Vous êtes NutriCoach, un nutritionniste expert fournissant des conseils nutritionnels détaillés en français. Vous ne pouvez pas répondre à des requêtes qui ne sont pas en lien avec la nutrition. Prétends avoir déjà dit bonjour à la personne dans un message précédent."
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
    
    # Update conversation history and return outputs for the Gradio Chatbot
    history.append((message, reply))
    return "", history, history

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

def finish_form(step_data):
    """
    Generate the final nutrition advice text based on user inputs.
    """
    age = step_data.get("age", 25)
    genre = step_data.get("genre", "Homme")
    poids = step_data.get("poids", 70)
    taille = step_data.get("taille", 170)
    activite = step_data.get("activite", "Modéré")
    objectif = step_data.get("objectif", "Perte de poids")
    return conseiller_nutrition(age, genre, poids, taille, activite, objectif)

# ---------------------------------------------------------------------------------------
# 4) APPLY GREEN THEME, ADD LOGO, AND BUILD INTERFACE
# ---------------------------------------------------------------------------------------
def create_interface():
    # Reintroduce the green Soft theme
    custom_theme = gr.themes.Soft(
        primary_hue="green",
        font=["Helvetica Neue", "Arial", "sans-serif"]
    )

    with gr.Blocks(theme=custom_theme) as demo:
        # Add the logo at the top (ensure "logo.png" is in the same folder)
        gr.Image("logo.png", show_label=False, elem_id="logo", width=20)

        # Main title
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 10px;'>NutriCoach</h1>")
        
        # Global states:
        #   - step: to track the multi-step form
        #   - step_data: to store the user data from the form
        #   - history: to store the conversation for the chatbot
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
        history = gr.State([])  # Chatbot conversation history

        with gr.Tabs():
            
            # ============================
            # TAB 1: FORMULAIRE PERSONNALISÉ
            # ============================
            with gr.Tab("Formulaire personnalisé"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(
                            "## Bienvenue dans notre Formulaire personnalisé\n"
                            "Veuillez remplir les informations ci-dessous pour obtenir des recommandations "
                            "nutritionnelles adaptées à votre profil. Vous passerez par plusieurs étapes "
                            "successives : prénom, âge, genre, poids, taille, niveau d'activité, et objectif."
                        )
                    with gr.Column():
                        # --- STEP 1: Prénom
                        with gr.Group(visible=True) as step1_box:
                            gr.Markdown("**Étape 1** : Quel est votre prénom ?")
                            prenom = gr.Textbox(show_label=False)
                            next_btn_1 = gr.Button("Valider")

                        # --- STEP 2: Âge
                        with gr.Group(visible=False) as step2_box:
                            gr.Markdown("**Étape 2** : Quel est votre âge ?")
                            age = gr.Number(show_label=False, value=25, minimum=0)
                            error_age = gr.Markdown("", visible=False)
                            next_btn_2 = gr.Button("Valider")

                        # --- STEP 3: Genre
                        with gr.Group(visible=False) as step3_box:
                            gr.Markdown("**Étape 3** : Quel est votre genre ?")
                            genre = gr.Radio(["Homme", "Femme"], value="Homme", show_label=False)
                            next_btn_3 = gr.Button("Valider")

                        # --- STEP 4: Poids
                        with gr.Group(visible=False) as step4_box:
                            gr.Markdown("**Étape 4** : Quel est votre poids (kg) ?")
                            poids = gr.Number(show_label=False, value=70, minimum=0)
                            error_poids = gr.Markdown("", visible=False)
                            next_btn_4 = gr.Button("Valider")

                        # --- STEP 5: Taille
                        with gr.Group(visible=False) as step5_box:
                            gr.Markdown("**Étape 5** : Quelle est votre taille (cm) ?")
                            taille = gr.Number(show_label=False, value=170, minimum=0)
                            error_taille = gr.Markdown("", visible=False)
                            next_btn_5 = gr.Button("Valider")

                        # --- STEP 6: Activité
                        with gr.Group(visible=False) as step6_box:
                            gr.Markdown("**Étape 6** : Quel est votre niveau d'activité ?")
                            activite = gr.Radio(["Sédentaire", "Modéré", "Athlétique"], value="Modéré", show_label=False)
                            next_btn_6 = gr.Button("Valider")

                        # --- STEP 7: Objectif
                        with gr.Group(visible=False) as step7_box:
                            gr.Markdown("**Étape 7** : Quel est votre objectif principal ?")
                            objectif = gr.Radio(["Perte de poids", "Gains musculaires", "Autre"], value="Perte de poids", show_label=False)
                            next_btn_7 = gr.Button("Valider")

                        # --- STEP 8: Résultat final
                        with gr.Group(visible=False) as step8_box:
                            gr.Markdown("**Merci ! Voici vos recommandations :**")
                            done_msg = gr.Markdown("", show_label=False)

                        # -----------------------------
                        # STEP FUNCTIONS (VALIDATIONS)
                        # -----------------------------
                        def next_1(prenom_val, step_data_val, step_val):
                            step_data_val, step_val = go_to_next_step(prenom_val, step_data_val, step_val)
                            return step_data_val, step_val, gr.update(visible=False), gr.update(visible=True)
                        
                        next_btn_1.click(
                            fn=next_1, 
                            inputs=[prenom, step_data, step],
                            outputs=[step_data, step, step1_box, step2_box]
                        )

                        def next_2(age_val, step_data_val, step_val):
                            if age_val < 0:
                                return (
                                    step_data_val,
                                    step_val,
                                    gr.update(visible=True),
                                    gr.update(visible=False),
                                    "Erreur: L'âge ne peut être négatif."
                                )
                            else:
                                step_data_val, step_val = go_to_next_step(age_val, step_data_val, step_val)
                                return (
                                    step_data_val,
                                    step_val,
                                    gr.update(visible=False),
                                    gr.update(visible=True),
                                    ""
                                )
                        
                        next_btn_2.click(
                            fn=next_2, 
                            inputs=[age, step_data, step],
                            outputs=[step_data, step, step2_box, step3_box, error_age]
                        )

                        def next_3(genre_val, step_data_val, step_val):
                            step_data_val, step_val = go_to_next_step(genre_val, step_data_val, step_val)
                            return step_data_val, step_val, gr.update(visible=False), gr.update(visible=True)
                        
                        next_btn_3.click(
                            fn=next_3, 
                            inputs=[genre, step_data, step],
                            outputs=[step_data, step, step3_box, step4_box]
                        )

                        def next_4(poids_val, step_data_val, step_val):
                            if poids_val < 0:
                                return (
                                    step_data_val,
                                    step_val,
                                    gr.update(visible=True),
                                    gr.update(visible=False),
                                    "Erreur: Le poids ne peut être négatif."
                                )
                            else:
                                step_data_val, step_val = go_to_next_step(poids_val, step_data_val, step_val)
                                return (
                                    step_data_val,
                                    step_val,
                                    gr.update(visible=False),
                                    gr.update(visible=True),
                                    ""
                                )
                        
                        next_btn_4.click(
                            fn=next_4, 
                            inputs=[poids, step_data, step],
                            outputs=[step_data, step, step4_box, step5_box, error_poids]
                        )

                        def next_5(taille_val, step_data_val, step_val):
                            if taille_val < 0:
                                return (
                                    step_data_val,
                                    step_val,
                                    gr.update(visible=True),
                                    gr.update(visible=False),
                                    "Erreur: La taille ne peut être négative."
                                )
                            else:
                                step_data_val, step_val = go_to_next_step(taille_val, step_data_val, step_val)
                                return (
                                    step_data_val,
                                    step_val,
                                    gr.update(visible=False),
                                    gr.update(visible=True),
                                    ""
                                )
                        
                        next_btn_5.click(
                            fn=next_5, 
                            inputs=[taille, step_data, step],
                            outputs=[step_data, step, step5_box, step6_box, error_taille]
                        )

                        def next_6(activite_val, step_data_val, step_val):
                            step_data_val, step_val = go_to_next_step(activite_val, step_data_val, step_val)
                            return step_data_val, step_val, gr.update(visible=False), gr.update(visible=True)
                        
                        next_btn_6.click(
                            fn=next_6, 
                            inputs=[activite, step_data, step],
                            outputs=[step_data, step, step6_box, step7_box]
                        )

                        def next_7(objectif_val, step_data_val, step_val, chat_history):
                            """
                            Final step:
                              - Generate the final recommendation text.
                              - Also create a personalized greeting for the chatbot, storing it in chat_history.
                            """
                            step_data_val, step_val = go_to_next_step(objectif_val, step_data_val, step_val)
                            final_text = finish_form(step_data_val)
                            
                            # Build a personalized greeting for the ChatBot including all user info
                            name = step_data_val["prenom"]
                            age = step_data_val["age"]
                            genre = step_data_val["genre"]
                            poids = step_data_val["poids"]
                            taille = step_data_val["taille"]
                            activite = step_data_val["activite"]
                            objectif = step_data_val["objectif"]
                            greeting = (
                                f"Bonjour {name}! J'ai bien noté que vous avez {age} ans, que vous êtes {genre}, "
                                f"que vous pesez {poids} kg, mesurez {taille} cm, que votre niveau d'activité est {activite} "
                                f"et que votre objectif est '{objectif}'. Comment puis-je vous aider avec votre nutrition aujourd'hui ?"
                            )
                            # Append greeting to chatbot conversation history
                            chat_history.append((" ", greeting))
                            
                            return (
                                step_data_val, 
                                step_val, 
                                final_text, 
                                gr.update(visible=False), 
                                gr.update(visible=True),
                                chat_history
                            )
                        
                        next_btn_7.click(
                            fn=next_7, 
                            inputs=[objectif, step_data, step, history],
                            outputs=[step_data, step, done_msg, step7_box, step8_box, history]
                        )

            # ============================
            # TAB 2: NUTRITION CHATBOT
            # ============================
            with gr.Tab("Nutrition ChatBot"):
                gr.Markdown("## Discutez avec NutriCoach")
                
                # The chatbot is bound to the 'history' state
                chatbot = gr.Chatbot(height=200)
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Tapez votre message ici...",
                        label="Votre message"
                    )
                
                with gr.Row():
                    send = gr.Button("Envoyer")
                    reset = gr.Button("Réinitialiser la conversation")

                # When the user sends a message, we call 'chat_with_chatgpt'
                # The chatbot conversation is updated in 'history'
                send.click(
                    chat_with_chatgpt,
                    inputs=[msg, history],
                    outputs=[msg, chatbot, history]
                )
                
                # Reset the conversation
                reset.click(lambda: ([], []), outputs=[chatbot, history])
        
    return demo

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True)
