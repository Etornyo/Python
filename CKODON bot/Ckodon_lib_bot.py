import os
import requests
import asyncio

from typing import Final
from telegram import Update,  InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Updater, CommandHandler, CallbackContext, MessageHandler, Filters

import tracemalloc
tracemalloc.start()


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
def start(update: Update, context: CallbackContext):
    update.message.reply_text("Hello! Welcome to the CKODON Library Archive. You can request videos, PDFs, and information. What can I do for you today?")
    send_inline_buttons(update, context)
    
    

# Creating buttons
def send_inline_buttons(update: Update, context: CallbackContext):
    buttons = [
        [InlineKeyboardButton("Videos", callback_data="btn1"),
        InlineKeyboardButton("PDFs", callback_data="btn2")],
        [InlineKeyboardButton("More Info", callback_data="btn3")]
    ]
    reply_markup = InlineKeyboardMarkup(buttons)
    update.message.reply_text("Click the button below:", reply_markup=reply_markup)



    
# Connects users to the Google drive via URL. Still working on the button
def get_video(update: Update, context: CallbackContext):
    buttons = [
        [InlineKeyboardButton("Watch Video", callback_data="watch_video")]
    ]
    reply_markup = InlineKeyboardMarkup(buttons)
    update.message.reply_text("Click the button below to watch the video:", reply_markup=reply_markup)



# Workings on PDF
def get_pdf(update: Update, context: CallbackContext):
    # Creating buttons for the various PDFs. Total number of PDFs = total number of buttons + button for all PDFS
    buttons = []
    for pdf in pdf_list:
        buttons.append([InlineKeyboardButton(pdf["title"], callback_data=pdf["url"])])
    
    buttons.append([InlineKeyboardButton("Get All PDFs", callback_data="all_pdfs")])
    
    reply_markup = InlineKeyboardMarkup(buttons)
    update.message.reply_text("Please choose a PDF:", reply_markup=reply_markup)

# Response to the button clicked by the user
def button_click(update: Update, context: CallbackContext):
    query = update.callback_query
    button_data = query.data
    
    # For video button
    if button_data == "watch_video":
        video_link = "YOUR_VIDEO_LINK"  
        video_message = "Click the link to watch the video:\n" + video_link
        query.message.reply_text(video_message)
        
        
    elif button_data == "all_pdfs":
        # Send all PDFs as documents
        for pdf in pdf_list:
            query.message.reply_document(document=pdf["url"], caption=pdf["title"])

    else:
        # Send the selected PDF
        for pdf in pdf_list:
            if button_data == pdf["url"]:
                query.message.reply_document(document=pdf_url, caption="Selected PDF")
                break
    




def get_info(update: Update, context: CallbackContext):
    query = update.message.text
    # Process the query and provide relevant information
    response = "Here's the information you requested: ..."
    update.message.reply_text(response)

def main():
    updater = Updater(token=TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    start_handler = CommandHandler('start', start)
    dispatcher.add_handler(start_handler)
    
    get_video_handler = CommandHandler('getvideo', get_video)
    dispatcher.add_handler(get_video_handler)
    
    get_pdf_handler = CommandHandler('getpdf', get_pdf)
    dispatcher.add_handler(get_pdf_handler)
    
    get_info_handler = MessageHandler(Filters.text & ~Filters.command, get_info)
    dispatcher.add_handler(get_info_handler)

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
    # asyncio.run(main())
    