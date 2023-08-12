import os
import requests
import asyncio
from typing import Final
from telegram import Update,  InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Updater, CommandHandler, CallbackContext, MessageHandler, filters

TOKEN = '5629735646:AAFGcCoVpEBvFih7Pd-VyAMYyzqPY-Z709Y'
Bot_USERNAME:Final = '@CKODON_bot'
GOOGLE_DRIVE_API_KEY = 'YOUR_GOOGLE_DRIVE_API_KEY'

# For Pdf listing and naming
pdf_list = [
    {"title": "PDF 1", "url": "URL_1"},
    {"title": "PDF 2", "url": "URL_2"},
    {"title": "PDF 3", "url": "URL_3"}
    ]

# Response after starting the bot
async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("Hello! Welcome to the CKODON Library Archive. You can request videos, PDFs, and information. What can I do for you today?")

# Connects users to the Google drive via URL. Still working on the button
async def get_video(update: Update, context: CallbackContext):
    buttons = [
        [InlineKeyboardButton("Watch Video", callback_data="watch_video")]
    ]
    reply_markup = InlineKeyboardMarkup(buttons)
    update.message.reply_text("Click the button below to watch the video:", reply_markup=reply_markup)
    pass




# Workings on PDF
async def get_pdf(update: Update, context: CallbackContext):
    # Creating buttons for the various PDFs. Total number of PDFs = total number of buttons + button for all PDFS
    buttons = []
    for pdf in pdf_list:
        buttons.append([InlineKeyboardButton(pdf["title"], callback_data=pdf["url"])])
    
    buttons.append([InlineKeyboardButton("Get All PDFs", callback_data="all_pdfs")])
    
    reply_markup = InlineKeyboardMarkup(buttons)
    update.message.reply_text("Please choose a PDF:", reply_markup=reply_markup)

# Response to the button clicked by the user
async def button_click(update: Update, context: CallbackContext):
    query = update.callback_query
    pdf_url = query.data
    
    # For video button
    if query.data == "watch_video":
        video_link = "YOUR_VIDEO_LINK"  
        video_message = "Click the link to watch the video:\n" + video_link
        query.message.reply_text(video_message)
        
        
    if pdf_url == "all_pdfs":
        # Send all PDFs as documents
        for pdf in pdf_list:
            query.message.reply_document(document=pdf["url"], caption=pdf["title"])
    else:
        # Send the selected PDF
        query.message.reply_document(document=pdf_url, caption="Selected PDF")
        
    




async def get_info(update: Update, context: CallbackContext):
    query = update.message.text
    # Process the query and provide relevant information
    response = "Here's the information you requested: ..."
    update.message.reply_text(response)

async def main():
    updater = Updater(token=TOKEN, use_context=True)
    bot = updater.bot
    dispatcher = updater.dispatcher

    start_handler = CommandHandler('start', start)
    get_video_handler = CommandHandler('getvideo', get_video)
    get_pdf_handler = CommandHandler('getpdf', get_pdf)
    get_info_handler = MessageHandler(Filters.text & ~Filters.command, get_info)

    dispatcher.add_handler(start_handler)
    dispatcher.add_handler(get_video_handler)
    dispatcher.add_handler(get_pdf_handler)
    dispatcher.add_handler(get_info_handler)

    updater.start_polling()
    await updater.idle()

if __name__ == '__main__':
    main()
    asyncio.run(main())
    