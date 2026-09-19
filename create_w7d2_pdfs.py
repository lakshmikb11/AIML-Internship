from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


OUTPUT_DIR = Path("w7d2_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

documents = [
    (
        "01_python_programming.pdf",
        "Python Programming",
        [
            "Python is a high-level programming language widely used for automation, data analysis, and machine learning.",
            "Variables store values such as numbers, strings, and lists.",
            "Functions group reusable instructions and can accept parameters and return values.",
            "Lists are ordered collections that can contain multiple values.",
            "Dictionaries store information as key-value pairs.",
        ],
    ),
    (
        "02_data_science.pdf",
        "Data Science",
        [
            "Data science combines statistics, programming, and domain knowledge to extract useful information from data.",
            "Data preprocessing includes cleaning missing values, removing duplicates, and transforming features.",
            "Exploratory data analysis helps identify patterns, distributions, and relationships in datasets.",
            "Feature engineering creates useful input variables for machine learning models.",
            "Data visualization can communicate trends and relationships clearly.",
        ],
    ),
    (
        "03_machine_learning_models.pdf",
        "Machine Learning Models",
        [
            "Linear regression predicts continuous numerical values using relationships between features and a target.",
            "Logistic regression is commonly used for classification problems.",
            "Decision trees make predictions using a sequence of feature-based decisions.",
            "Random forests combine multiple decision trees to improve predictive performance.",
            "Model evaluation should use metrics appropriate for the problem type.",
        ],
    ),
    (
        "04_deep_learning.pdf",
        "Deep Learning",
        [
            "Deep learning uses neural networks with multiple layers to learn representations from data.",
            "Neurons combine inputs with weights and apply an activation function.",
            "ReLU is a commonly used activation function in neural networks.",
            "Backpropagation calculates gradients that help update model parameters.",
            "Convolutional neural networks are particularly useful for image-related tasks.",
        ],
    ),
    (
        "05_information_retrieval.pdf",
        "Information Retrieval",
        [
            "Information retrieval systems find relevant documents in response to a user query.",
            "BM25 is a lexical retrieval method based on term frequency and document statistics.",
            "Dense retrieval represents documents and queries as numerical embeddings.",
            "Semantic similarity allows dense retrieval to find related text even when exact words differ.",
            "Retrieval quality can be evaluated using metrics such as Precision@k and Recall@k.",
        ],
    ),
]


def create_pdf(filename, title, paragraphs):
    path = OUTPUT_DIR / filename

    styles = getSampleStyleSheet()
    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        title=title,
    )

    story = [
        Paragraph(title, styles["Title"]),
        Spacer(1, 12),
    ]

    for paragraph in paragraphs:
        story.append(Paragraph(paragraph, styles["BodyText"]))
        story.append(Spacer(1, 8))

    document.build(story)
    print(f"Created: {path}")


def main():
    print("=" * 70)
    print("W7D2: CREATING PDF DOCUMENTS")
    print("=" * 70)

    for filename, title, paragraphs in documents:
        create_pdf(filename, title, paragraphs)

    print()
    print(f"Created {len(documents)} PDF documents in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()