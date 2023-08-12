from typing import Final
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

Token: Final = '6652130219:AAHEZ35CRqZxNOdFm2wzElI7kQXBUbhuDUQ'
Bot_USERNAME:Final = '@zmuse_bot'

async def start_command(update:Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Hello! Thanks for using ZMuse. What can I fetch You?')
    
    
    
    def search_music(update: Update, context: CallbackContext):
    query = update.message.text
    search_url = f"{DEEZER_API_BASE_URL}/search"
    params = {'q': query}
    
    response = requests.get(search_url, params=params)
    data = response.json()
    
    if 'data' in data:
        tracks = data['data']
        if tracks:
            message = "Here are some search results:\n"
            for idx, track in enumerate(tracks[:5], start=1):
                message += f"{idx}. {track['title']} by {track['artist']['name']}\n"
            update.message.reply_text(message)
        else:
            update.message.reply_text("No results found.")
    else:
        update.message.reply_text("Error while searching for music.")

def main():
    updater = Updater(token=TOKEN, use_context=True)
    dispatcher = updater.dispatcher
    
    start_handler = CommandHandler('start', start)
    search_handler = MessageHandler(Filters.text & ~Filters.command, search_music)
    
    dispatcher.add_handler(start_handler)
    dispatcher.add_handler(search_handler)
    
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()