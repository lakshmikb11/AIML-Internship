from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

OUTPUT_PATH = "./data/w5d4/ai_ml_reference.pdf"

doc = SimpleDocTemplate(
    OUTPUT_PATH,
    pagesize=A4,
    rightMargin=50,
    leftMargin=50,
    topMargin=50,
    bottomMargin=50,
)

styles = getSampleStyleSheet()
story = []

story.append(Paragraph("AI/ML Semantic Search Reference", styles["Title"]))
story.append(Spacer(1, 12))

sections = [
    (
        "Machine Learning",
        "Machine learning enables computer systems to learn patterns from data "
        "and use those patterns to make predictions or decisions."
    ),
    (
        "Supervised Learning",
        "Supervised learning uses labeled training data. Classification predicts "
        "discrete categories, while regression predicts continuous numerical values."
    ),
    (
        "Overfitting",
        "Overfitting happens when a model learns the training data too closely "
        "and does not generalize well to unseen data."
    ),
    (
        "Preventing Overfitting",
        "Overfitting can be reduced using cross-validation, regularization, "
        "reducing unnecessary features, limiting model complexity, and collecting "
        "more representative training data."
    ),
    (
        "Model Evaluation",
        "Common evaluation measures include accuracy, precision, recall, and F1-score. "
        "A separate test set can be used to evaluate final performance on unseen data."
    ),
    (
        "Feature Scaling",
        "Feature scaling transforms numerical features to comparable ranges. "
        "It can be important for algorithms that depend on distances or feature magnitudes."
    ),
    (
        "Decision Trees and Random Forests",
        "Decision trees make predictions using feature-based splits. Random forests "
        "combine multiple decision trees to improve generalization."
    ),
    (
        "Support Vector Machines",
        "Support Vector Machines identify decision boundaries that separate classes. "
        "They can be used for classification tasks."
    ),
    (
        "Neural Networks",
        "Neural networks use layers of interconnected computational units to learn "
        "patterns from data."
    ),
]

for heading, text in sections:
    story.append(Paragraph(heading, styles["Heading2"]))
    story.append(Paragraph(text, styles["BodyText"]))
    story.append(Spacer(1, 10))

doc.build(story)

print(f"Created PDF: {OUTPUT_PATH}")