from typing import Final
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

Token: Final = '6652130219:AAHEZ35CRqZxNOdFm2wzElI7kQXBUbhuDUQ'
Bot_USERNAME:Final = '@zmuse_bot'

async def start_command(update:Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Hello! Thanks for using ZMuse. What can I fetch You?')
    