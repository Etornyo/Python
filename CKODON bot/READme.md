Certainly! Here's a sample README file for your Telegram bot code:

---

# CKODON Library Archive Bot

The CKODON Library Archive Bot is a Telegram bot that allows users to request videos, PDFs, and information from the CKODON Library's collection. Users can interact with the bot to watch videos, access PDFs, and receive relevant information.

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Usage](#usage)
- [Installation](#installation)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Description

The CKODON Library Archive Bot is a Python-based Telegram bot that utilizes the `python-telegram-bot` library to provide users with easy access to videos, PDFs, and information from the CKODON Library's collection. The bot responds to user commands and interactions to fulfill their requests.

## Features

- **Video Requests**: Users can request to watch videos from the CKODON Library's collection. The bot provides a button that allows users to directly access the video link.

- **PDF Retrieval**: Users can request PDFs from the library's collection. The bot displays a list of available PDFs along with an option to get all PDFs in one go.

- **Information Retrieval**: Users can request specific information, and the bot provides relevant responses.

## Usage

1. Start the bot by sending the `/start` command.
2. Use the `/getvideo` command to request to watch videos.
3. Use the `/getpdf` command to request PDFs. Choose specific PDFs or request all available PDFs.
4. Send any text message to get relevant information.

## Installation

To run the CKODON Library Archive Bot, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies using the `requirements.txt` file.
3. Replace the placeholders for the bot token and Google Drive API key in the code.
4. Run the `Ckodon_lib_bot.py` script using Python.

## Configuration

In the `Ckodon_lib_bot.py` script, you need to provide your Telegram bot token and Google Drive API key. Replace the placeholders in the code with your actual keys.

```python
TOKEN = 'YOUR_BOT_TOKEN'
GOOGLE_DRIVE_API_KEY = 'YOUR_GOOGLE_DRIVE_API_KEY'
```

## Dependencies

The CKODON Library Archive Bot relies on the following libraries:

- `python-telegram-bot`: A Python wrapper for the Telegram Bot API. Install using `pip install python-telegram-bot`.

## Contributing

Contributions to the CKODON Library Archive Bot are welcome! If you find any issues or want to add new features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

---

Feel free to customize this README according to your project's specific details and requirements.