from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


output_path = "./data/w5d3/ai_ml_reference.pdf"

doc = SimpleDocTemplate(
    output_path,
    pagesize=A4,
    rightMargin=50,
    leftMargin=50,
    topMargin=50,
    bottomMargin=50,
)

styles = getSampleStyleSheet()

title_style = styles["Title"]
heading_style = styles["Heading2"]
body_style = styles["BodyText"]

story = []

story.append(Paragraph("AI/ML Reference Notes", title_style))
story.append(Spacer(1, 20))

sections = [
    (
        "Machine Learning",
        "Machine learning enables computers to learn patterns from data and use those patterns to make predictions or decisions. "
        "A typical machine learning workflow includes collecting data, preprocessing it, selecting features, training a model, "
        "evaluating the model, and deploying the model."
    ),
    (
        "Supervised Learning",
        "Supervised learning uses labeled training data. Classification is used when the target consists of categories, "
        "while regression is used when the target is a continuous numerical value. Examples include spam classification and house-price prediction."
    ),
    (
        "Overfitting",
        "Overfitting occurs when a model learns the training data too closely, including patterns that do not generalize to new data. "
        "An overfit model can have excellent training performance but weaker validation or test performance."
    ),
    (
        "Preventing Overfitting",
        "Several techniques can reduce overfitting. Cross-validation provides a more reliable estimate of generalization performance. "
        "Regularization adds a penalty for model complexity. Other approaches include reducing unnecessary features, limiting model complexity, "
        "and collecting more representative training data."
    ),
    (
        "Model Evaluation",
        "Classification models can be evaluated using accuracy, precision, recall, F1-score, and ROC-AUC. "
        "A confusion matrix shows the numbers of correct and incorrect predictions for each class. "
        "Regression models can be evaluated using metrics such as mean absolute error, mean squared error, and R-squared."
    ),
    (
        "Feature Scaling",
        "Feature scaling transforms numerical features to comparable ranges. Standardization commonly transforms a feature so that it has "
        "approximately zero mean and unit variance. Scaling is particularly useful for algorithms based on distances or feature magnitudes."
    ),
    (
        "Decision Trees and Random Forests",
        "A decision tree makes predictions by recursively splitting data according to feature conditions. "
        "Random forests combine predictions from multiple decision trees and can provide better generalization than a single tree."
    ),
    (
        "Support Vector Machines",
        "Support Vector Machines, or SVMs, find decision boundaries that separate classes. "
        "The kernel method allows SVMs to model nonlinear relationships by mapping data into a different feature space."
    ),
    (
        "Neural Networks",
        "Neural networks contain interconnected layers of computational units called neurons. "
        "During training, the network adjusts its parameters to reduce prediction error. "
        "Deep neural networks contain multiple hidden layers and can learn complex representations."
    ),
]

for heading, text in sections:
    story.append(Paragraph(heading, heading_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(text, body_style))
    story.append(Spacer(1, 15))

doc.build(story)

print(f"PDF created successfully: {output_path}")