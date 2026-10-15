# Gemini Sentence Assist

Use Google Gemini to add English translations and short grammar notes to Japanese sentence cards, in bulk.

## Features

- **Tools → Gemini Sentence Assist…** or Browser right-click **Gemini: assist selected notes…**
- Translation and/or grammar note into fields you choose
- Pick any Gemini model (refresh the list from Google); flash-lite is cheap and fast
- Custom system / translation / grammar prompts
- Skips filled fields unless overwrite is on; cancellable background run
- Debug log in the add-on's `user_files/gemini.log` (no API key is logged)

## Requirements

- Anki 23.10+ (Qt6)

## API key

Your own Gemini API key from https://aistudio.google.com/apikey — enter it in the add-on dialog or **Tools → Add-ons → Config**. Usage is billed/limited by Google on your key.

The add-on ships with an empty key. Nothing is sent anywhere until you set one.

## Install

### From AnkiWeb (recommended)
1. In Anki: **Tools → Add-ons → Get Add-ons…**
2. Enter code: **1196353080** (AnkiWeb page: https://ankiweb.net/shared/info/1196353080)
3. Restart Anki.

### Manually from GitHub Releases
1. Download `gemini-sentence-assist.ankiaddon` from https://github.com/arturkoziara/anki-gemini-sentence-assist/releases/latest
2. Double-click it, or in Anki use **Tools → Add-ons → Install from file…**
3. Restart Anki.

## Configuration

See `config.md` (also shown in **Tools → Add-ons → Config**).

## Feedback

Report bugs and ideas at https://github.com/arturkoziara/anki-gemini-sentence-assist/issues

## License

MIT
