# W5D5: Ollama Model Comparison

## Models Compared

- llama3.2:3b
- qwen2.5:3b

Both models were tested using the same three AI/ML questions.

## Question 1: Supervised Learning

### llama3.2:3b
- Explained supervised learning using labeled data.
- Used a customer purchase prediction example.
- Included a small tabular dataset example.
- Provided a more detailed explanation of input features and output labels.

### qwen2.5:3b
- Explained supervised learning using labeled data.
- Used a spam email classification example.
- Focused on the relationship between inputs and correct outputs.
- Used a simple real-world example suitable for beginners.

### Observation
Both models correctly explained the main concept. The examples and level of detail differed.

---

## Question 2: Overfitting

### llama3.2:3b
- Explained overfitting as learning training data too closely.
- Used house-price prediction as an example.
- Discussed regularization, data augmentation, early stopping, ensemble methods, and model simplification.
- Included a Python example using L1 regularization.

### qwen2.5:3b
- Explained overfitting as learning training-data details and noise.
- Used image recognition as an example.
- Discussed increasing training data, cross-validation, regularization, dropout, and ensemble methods.
- Focused mainly on conceptual explanations.

### Observation
Both models identified overfitting and several common mitigation techniques. llama3.2:3b provided a code example, while qwen2.5:3b focused more on conceptual explanations.

Note: Some generated details should still be checked against trusted technical documentation before being used in production or educational material.

---

## Question 3: Classification vs Regression

### llama3.2:3b
- Explained classification as predicting categories.
- Explained regression as predicting continuous numerical values.
- Used spam detection and house-price prediction examples.
- Included a direct comparison of output types.

### qwen2.5:3b
- Explained regression as predicting continuous numerical values.
- Explained classification as predicting fixed categories.
- Used cookie prediction and weather classification examples.
- Summarized the distinction clearly at the end.

### Observation
Both models correctly identified the main distinction: classification predicts categories, while regression predicts continuous numerical values. They used different examples and explanation styles.

---

## Overall Comparison

| Aspect | llama3.2:3b | qwen2.5:3b |
|---|---|---|
| Explanation style | Detailed, with extended examples | Structured, concise conceptual explanations |
| Examples | Customer purchase, house prices, spam | Spam, image recognition, cookies, weather |
| Code/example depth | Included code for overfitting | Primarily conceptual |
| Beginner explanations | Clear with detailed walkthroughs | Clear with simple examples |
| Response structure | Longer and more elaborated | More compact and organized |

## Conclusion

Both local Ollama models successfully answered the same three AI/ML questions. The responses showed differences in examples, response length, structure, and use of code. The comparison demonstrates that local LLMs can provide useful answers while producing different styles of explanation for the same prompts.

## Manual Verification

- Same three questions were used for both models.
- llama3.2:3b generated responses successfully.
- qwen2.5:3b generated responses successfully.
- Response differences were manually reviewed and documented.
