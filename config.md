# Gemini Sentence Assist

- **gemini_api_key** — Google AI Studio API key ([aistudio.google.com/apikey](https://aistudio.google.com/apikey))
- **gemini_model** — selected model id (refresh via the dialog; defaults to `gemini-2.5-flash-lite`)
- **gemini_models** — last refreshed model list from Google `models.list`
- **gemini_sentence_field** — JP sentence source field (default `Sentence`)
- **gemini_translation_field** — EN translation destination (default `Sentence Meaning`)
- **gemini_grammar_field** — short grammar note destination (default `Notes`)
- **gemini_system_instructions** / **gemini_translation_instructions** / **gemini_grammar_instructions** — editable prompt text (empty = built-in defaults)

## Wizard

1. **Setup** — API key, model, From → To field mapping (optional prompt edits)
2. **Find notes** — Find needing fill / list all, or open from Browser right‑click
3. **Generate** — Preview selected (1) → Apply preview, or Fill selected to write

Use **Tools → Gemini Sentence Assist…**, or in **Browse** right‑click → **Gemini: assist selected notes…**.

Turn on **Overwrite** on Setup to regenerate cards that already have EN/grammar text.
