# W5D2: Llama 3.2 3B vs Qwen 2.5 3B

## Objective

Compare the response quality of the local Ollama models `llama3.2:3b` and `qwen2.5:3b` using the same system prompt and the same three questions.

## Test Setup

- **API:** `http://localhost:11434/api/chat`
- **Model 1:** `llama3.2:3b`
- **Model 2:** `qwen2.5:3b`
- **System Prompt:**  
  "You are an AI/ML tutor for beginners. Explain concepts clearly using simple language. Give a short example when useful. Structure your answers with concise points and avoid unnecessary jargon."

### Questions Tested

1. What is supervised learning?
2. Explain overfitting in simple terms.
3. What is the difference between classification and regression?

## Observations

### 1. Supervised Learning

**Llama 3.2 3B**

- Explained supervised learning using labeled data.
- Used a step-by-step explanation of the training process.
- Included a dog-and-cat image classification example.
- Used headings and bullet points to organize the response.

**Qwen 2.5 3B**

- Explained supervised learning using labeled input-output pairs.
- Described how the model learns patterns by minimizing prediction errors.
- Used a spam-email example.
- Presented the explanation in a concise, structured format.

**Observation:** Both models explained the concept clearly and included practical examples. Llama provided a more step-by-step explanation, while Qwen was more compact.

### 2. Overfitting

**Llama 3.2 3B**

- Explained that overfitting occurs when a model learns noise and becomes too specialized to the training data.
- Described causes and consequences.
- Included several ways to reduce overfitting, such as simpler models, regularization, early stopping, and additional data.
- Used a movie-preference example.

**Qwen 2.5 3B**

- Explained overfitting as learning training details and noise too closely.
- Clearly contrasted strong training performance with weaker performance on unseen data.
- Mentioned model complexity and excessive training as possible causes.
- Included a cat-recognition example and prevention methods.

**Observation:** Both models covered the main idea clearly. Llama gave more supporting points, while Qwen presented the explanation more directly.

### 3. Classification vs Regression

**Llama 3.2 3B**

- Defined classification as predicting categorical labels.
- Defined regression as predicting continuous numerical values.
- Used customer purchase and house-price examples.
- Used a structured comparison with goals, outputs, and examples.

**Qwen 2.5 3B**

- Defined classification as predicting discrete outputs.
- Defined regression as predicting continuous outputs.
- Provided examples involving spam detection, cancer classification, house prices, travel time, and temperature.
- Used a concise numbered format.

**Observation:** Both models clearly distinguished categorical/discrete predictions from continuous numerical predictions. Llama gave a more detailed explanation, while Qwen was shorter and example-focused.

## Overall Qualitative Comparison

| Aspect | Llama 3.2 3B | Qwen 2.5 3B |
|---|---|---|
| Explanation style | Detailed and step-by-step | Compact and direct |
| Structure | Headings, bullets, numbered points | Bullets and numbered points |
| Examples | Practical examples included | Practical examples included |
| Beginner readability | Clear and explanatory | Clear and concise |
| Response length | Generally longer | Generally shorter |
| Technical detail | More supporting detail | Focused on core concepts |

## Conclusion

Both local models successfully followed the custom system prompt and produced clear explanations for the three AI/ML questions.

In this qualitative test, **Llama 3.2 3B** tended to provide more detailed, step-by-step explanations, while **Qwen 2.5 3B** tended to provide more compact and direct responses with practical examples.

This comparison is based only on the three questions tested and is a qualitative observation rather than a formal benchmark.



