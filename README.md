# Support Ticket Classifier

This project is a complete, end-to-end NLP pipeline to automatically classify customer support tickets into three categories: **Billing**, **Technical**, or **Other**. The solution includes data preparation, model training, and a production-ready REST API, all containerized with Docker for full reproducibility.


## The Modern NLP Approach

This project deliberately uses a modern, transformer-based approach over classic NLP techniques for superior performance and simplicity.

* **Why Transformers?** Models like **DistilBERT** are pre-trained on vast amounts of text, allowing them to understand context and nuance far better than traditional methods like TF-IDF.
* **Minimal Preprocessing:** We avoid aggressive text cleaning (like stop-word removal) because transformers perform better with natural, contextual text. Our preprocessing is handled cleanly by the model's tokenizer.
* **Supervised Learning:** As we have a dataset with pre-defined labels, this is a **supervised classification** task, which is the correct approach for this problem.

## Getting Started (Docker Recommended)

The entire application is containerized, making setup incredibly simple.

### Prerequisites

* Docker Desktop installed and running.
* Your dataset named `support_tickets.csv` placed in the `data/` folder.

### Run the Application

1. **Build the Docker image:** From the project root, run:

```bash
docker build -t ticket-classifier .
```

2. **Run the container:** This command runs the entire pipeline: data prep, model training, and then launches the API.

```bash
docker run -p 8000:8000 ticket-classifier
```

The API will now be available at `http://localhost:8000`.

## API Usage

The API exposes a `/predict` endpoint to classify new tickets.

### Example Request (`curl`)

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "My new wireless mouse fails to connect via Bluetooth. My laptop cannot discover it."
}'
```

### Example Response

```json
{
  "predictions": [
    {
      "label": "Technical",
      "confidence": 0.9947
    },
    {
      "label": "Other",
      "confidence": 0.0031
    },
    {
      "label": "Billing",
      "confidence": 0.0022
    }
  ]
}
```

## Project Structure

```
support-ticket-classifier/
├── Dockerfile              # Instructions to build the Docker image
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── .dockerignore           # Files to exclude from the Docker image
├── data/
│   └── support_tickets.csv # The raw dataset
└── src/
    ├── config.py           # Central configuration file
    ├── explore_data.py     # Data sampling and EDA script
    ├── train_baseline.py   # Script to train a simple benchmark model
    ├── train_transformer.py  # Main model training script
    └── api.py              # FastAPI application for serving the model
```

## Considerations & Future Work

* **Model Choice:** DistilBERT offers an excellent balance of speed and performance. For even higher accuracy in a production system, a larger model like `RoBERTa` could be fine-tuned.
* **Dataset Limitations:** The model's ability to generalize is constrained by the small 300-sample dataset. The immediate next step for improving this model would be to build a much larger (10,000+ examples) and cleaner training set.
* **Deployment:** For a production environment, this container would be deployed on a cloud service like AWS ECS or Google Cloud Run, managed by an orchestration system like Kubernetes for scalability and zero-downtime deployments.
