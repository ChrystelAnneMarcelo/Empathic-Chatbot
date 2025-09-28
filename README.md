# Empathic Chatbot

A simple rule-based + sentiment-analysis chatbot prototype that demonstrates empathic responses.  
The system runs on a **text-based command-line interface** and responds differently depending on user emotions such as sadness, happiness, stress, or neutral.

---

## Features

- Text-based interface (run in terminal/command line).
- Detects **sentiment polarity** using [TextBlob](https://textblob.readthedocs.io/en/dev/).
- Extra **keyword detection** for stress-related emotions (e.g., "overwhelmed", "anxious").
- Provides empathic, validating responses instead of generic replies.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Empathic-Chatbot.git
cd Empathic-Chatbot
```

### 2. Install dependencies

Make sure you have Python 3 installed. Then run:

```bash
pip3 install textblob
python3 -m textblob.download_corpora
```

### 3. Run the chatbot

```bash
python3 empathic_chatbot.py
```

---

## Demo Video

Watch a demo of the chatbot in action:  
[Demo Video Link]

---

## References

- [TextBlob Documentation](https://textblob.readthedocs.io/en/dev/)
