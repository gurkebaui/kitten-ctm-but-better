import discord
from discord.ext import commands
from dotenv import load_dotenv # <--- Neu
import os                     # <--- Neu
from inference import load_model, predict, format_messages

# Lade Umgebungsvariablen aus .env Datei
load_dotenv() 

# Konfiguration
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN") # <--- Liest aus .env
if not DISCORD_TOKEN:
    raise ValueError("Kein Discord Token gefunden! Bitte .env Datei prüfen.")
MODEL_PATH = "checkpoints/best_model.pt"
DECISION_THRESHOLD = 0.5  # Ab wann antworten?
MAX_HISTORY = 50           # Wie viele Nachrichten zurück?

# Discord Permissions
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

# Speicher für den Chat-Verlauf
context_history = []

@bot.event
async def on_ready():
    print(f'✅ Bot ist eingeloggt als {bot.user}')
    print(f'🧠 Warte auf Nachrichten... (CTM ist bereit)')

@bot.event
async def on_message(message):
    # 1. Sich selbst ignorieren (Wichtig!)
    if message.author == bot.user:
        return

    # 2. Kontext aktualisieren (In Liste schreiben)
    new_entry = {
        "username": message.author.name,
        "content": message.content
    }
    context_history.append(new_entry)

    # Liste kurz halten (älteste entfernen)
    if len(context_history) > MAX_HISTORY:
        context_history.pop(0)

    # 3. CTM fragen (Darf ich antworten?)
    # Wir nutzen die predict-Funktion aus inference.py
    try:
        prob, ticks = predict(context_history, model, tokenizer, device="cpu")
        
        print(f"--- Nachricht von {message.author.name} ---")
        print(f"CTM Score: {prob:.4f} | Ticks: {ticks}")

        # 4. Entscheidung treffen
        if prob > DECISION_THRESHOLD:
            print(">> CTM sagt: JA.")
            
            # Hier senden wir einen TEST-TEXT
            # (Hier würdest du später das andere AI aufrufen)
            await message.channel.send(f"CTM Check: {prob:.2f} -> JA, ich antworte!")
        
        else:
            print(">> CTM sagt: NEIN. (Ignoriere)")

    except Exception as e:
        print(f"Fehler bei Inference: {e}")

# --- START ---
# Wir laden das Modell im MAIN-Bereich, bevor der Bot startet
print("Lade CTM Modell...")
model, tokenizer = load_model(MODEL_PATH, device="cpu")

# Starte den Bot
bot.run(DISCORD_TOKEN)