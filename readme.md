# NLU Chatbot for Symptom-Based Healthcare Assistance

This project implements a **Natural Language Understanding (NLU) based chatbot** designed to recognize user symptoms and provide appropriate healthcare suggestions. Built with Python, TensorFlow, and NLTK, it classifies user inputs into predefined intents and responds accordingly.

---

## Features

- **Symptom detection:** Understands and identifies medical symptom-related queries.
- **Multi-symptom handling:** Recognizes if multiple symptoms are mentioned and advises accordingly.
- **Intent classification:** Uses a feedforward neural network trained on custom intents.
- **Interactive API:** FastAPI endpoint to integrate chatbot functionality.

---

## Project Structure

- `model.py` — Preprocesses data, builds and trains the neural network model, saves trained model and vocabulary.
- `chatbot.py` — Loads model and resources, processes user inputs, predicts intent, and generates responses.
- `main.py` — FastAPI app exposing a `/chat` POST endpoint for chatbot interaction.
- `intents.json` — JSON file containing intents, patterns, and responses.

---

## Usage

1. Train the model by running `model.py` (requires `intents.json`).
2. Start the API server with `main.py`.
3. Send POST requests with a JSON body containing `"message": "<your text>"` to `/chat`.
4. Receive chatbot responses and intent tags in JSON format.

---

## Dependencies

- Python 3.7+
- TensorFlow
- NLTK
- FastAPI
- Uvicorn (for running FastAPI)

---

## Installation

```bash
pip install tensorflow nltk fastapi uvicorn
python -m nltk.downloader punkt wordnet
```

## Example API Request

```bash
curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"message": "I have a headache and fever"}'
```

## Example Response

```json
{
  "response": "You have multiple symptoms. It’s better to book an appointment with a doctor. Call this number for assistance: 123-456-7890.",
  "tag": "multiple_symptoms"
}
```

## Acknowledgments

- Based on Natural Language Understanding techniques.
- Uses NLTK for preprocessing and TensorFlow for neural network modeling.

---
