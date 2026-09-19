from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer


OUTPUT_DIR = Path("w7d1_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


DOCUMENTS = {
    "01_machine_learning_fundamentals.pdf": (
        "Machine Learning Fundamentals",
        [
            "Machine learning is a branch of artificial intelligence that enables "
            "systems to learn patterns from data and make predictions or decisions.",
            "Supervised learning uses labeled examples. Common supervised learning "
            "tasks include classification and regression.",
            "Classification predicts a discrete category, such as spam or not spam.",
            "Regression predicts a continuous numerical value, such as house price.",
            "Unsupervised learning works with data without target labels. Clustering "
            "is a common unsupervised learning technique.",
            "Overfitting happens when a model learns the training data too closely "
            "and performs poorly on unseen data.",
            "A train-test split separates data into training and evaluation sets so "
            "that generalization can be measured."
        ],
    ),
    "02_neural_networks.pdf": (
        "Neural Networks",
        [
            "A neural network is a machine learning model composed of interconnected "
            "layers of computational units called neurons.",
            "A basic neural network contains an input layer, one or more hidden "
            "layers, and an output layer.",
            "The activation function introduces non-linearity into a neural network. "
            "ReLU is commonly used in hidden layers.",
            "During training, backpropagation calculates gradients of the loss with "
            "respect to model parameters.",
            "An optimizer updates weights using the calculated gradients. Adam is a "
            "widely used optimization algorithm.",
            "The learning rate controls the size of parameter updates during training.",
            "A convolutional neural network, or CNN, is especially useful for "
            "processing images because convolution layers can learn spatial features."
        ],
    ),
    "03_natural_language_processing.pdf": (
        "Natural Language Processing",
        [
            "Natural language processing, or NLP, focuses on enabling computers to "
            "process and understand human language.",
            "Tokenization divides text into smaller units called tokens.",
            "Text classification assigns labels to text. Sentiment analysis is an "
            "example of text classification.",
            "Word embeddings represent words or tokens as numerical vectors that "
            "capture useful semantic relationships.",
            "Transformers use attention mechanisms to model relationships between "
            "tokens in a sequence.",
            "Large language models are neural language models trained on large text "
            "datasets to perform tasks such as generation and question answering.",
            "Retrieval can provide relevant external documents to an NLP system "
            "before an answer is generated."
        ],
    ),
    "04_computer_vision.pdf": (
        "Computer Vision",
        [
            "Computer vision is an area of artificial intelligence concerned with "
            "extracting useful information from images and video.",
            "Image classification assigns an image to one or more predefined "
            "categories.",
            "Object detection identifies objects and their locations, commonly "
            "using bounding boxes.",
            "Image segmentation assigns labels to pixels or regions of an image.",
            "Convolutional layers detect local visual patterns such as edges, "
            "textures, and shapes.",
            "Data augmentation creates modified training examples such as rotated, "
            "cropped, or flipped images.",
            "Accuracy, precision, recall, and mean average precision are examples of "
            "metrics used to evaluate computer vision models."
        ],
    ),
    "05_retrieval_augmented_generation.pdf": (
        "Retrieval-Augmented Generation",
        [
            "Retrieval-augmented generation, or RAG, combines information retrieval "
            "with language generation.",
            "A RAG system first retrieves relevant documents from a knowledge source "
            "and then provides those documents as context to a language model.",
            "BM25 is a sparse retrieval method based primarily on term matching and "
            "term frequency statistics.",
            "Dense retrieval represents queries and documents as vectors and "
            "compares their semantic similarity.",
            "Embeddings are numerical vector representations of text that can be "
            "used for semantic retrieval.",
            "Chunking divides long documents into smaller passages before indexing.",
            "Good retrieval quality is important because the generator can only use "
            "the retrieved context that is supplied to it."
        ],
    ),
}


def create_pdf(filename, title, paragraphs):
    path = OUTPUT_DIR / filename

    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        title=title,
        author="Cynaris AI/ML Internship - W7D1",
    )

    styles = getSampleStyleSheet()
    story = [
        Paragraph(title, styles["Title"]),
        Spacer(1, 12),
    ]

    for paragraph in paragraphs:
        story.append(Paragraph(paragraph, styles["BodyText"]))
        story.append(Spacer(1, 10))

    document.build(story)
    print(f"Created: {path}")


def main():
    for filename, (title, paragraphs) in DOCUMENTS.items():
        create_pdf(filename, title, paragraphs)

    print(f"\nCreated {len(DOCUMENTS)} PDF documents in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()