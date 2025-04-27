# Model Card for promptguard-cnn-v1

A lightweight TextCNN model designed to classify LLM user prompts as either harmful or neutral, intended as a first-pass filter for AI safety.


## Model Details

### Model Description

This CNN detects harmful prompts before they reach a large language model (LLM), providing an extra layer of defense. It uses 3 convolutional layers with different kernel sizes to capture local patterns in user inputs, followed by max pooling and dropout regularization.


- **Developed by:** Caterina, Ritu, Zara 
- **Model type:** Text Classification Model (CNN)
- **Language(s) (NLP):** English
- **License:** MIT License

- **Repository:** https://github.com/esturzu/gcg-security

## Uses

This section addresses how the model is intended to be used, including foreseeable users and those affected.

### Direct Use

The model can be used directly to classify user prompts as either **harmful** or **neutral** without any additional fine-tuning.  
Direct use cases include:
- Pre-screening prompts in LLM-based systems to block harmful inputs before model execution.
- Integrating into chatbot interfaces or content moderation pipelines to detect unsafe user messages.
- Running lightweight safety filters on text submissions in online platforms.

The model is intended for users such as AI safety researchers, chatbot developers, platform moderators, and organizations deploying public-facing LLMs.

### Downstream Use

The model may be fine-tuned or adapted for specific domains where definitions of "harmful" vary, such as:
- Educational AI platforms that require stricter safety filters.
- Healthcare conversational agents needing domain-specific content restrictions.
- Region-specific moderation tools adapting to local norms and regulations.

It can also be plugged into larger safety ecosystems that perform layered screening (e.g., input filters + output monitors).


### Out-of-Scope Use

The model is **not** intended for:
- Making legal, employment, healthcare, or law enforcement decisions without human oversight.
- Classifying highly nuanced or legally sensitive prompts without domain-specific retraining.
- Use cases requiring zero-error tolerance, such as criminal investigations or medical triage systems.

The model was trained on general English prompts and may underperform on non-English text or prompts written with adversarial evasion techniques.

## Bias, Risks, and Limitations

While the model achieves strong accuracy and F1 scores, several limitations exist:
- **Bias in Training Data**: Definitions of "harmful" are based on curated datasets and may reflect dataset creator biases.
- **Subtlety Limits**: Some highly subtle or adversarial prompts may bypass classification.
- **Language and Cultural Bias**: The model was trained on English prompts and may not generalize well to other languages or culturally specific phrasing.
- **Overblocking Risk**: There is a risk of overblocking benign prompts that use certain flagged terms without harmful intent.

Additionally, the model’s binary classification approach does not provide granular risk scoring or explainability for why a prompt is labeled harmful.

### Recommendations

Users should:
- Combine model outputs with human review for high-stakes applications.
- Regularly revalidate model performance if deployed in evolving environments (e.g., new internet slang, adversarial trends).
- Consider additional domain-specific fine-tuning if applying the model in sensitive areas like healthcare, education, or law.
- Be aware that no model can guarantee 100% detection of harmful content, and use layered defense mechanisms where appropriate.

## How to Get Started with the Model

Use the code shared in wrapper2.ipynb to get started with the model. 

## Training Details

### Training Data

The training dataset consisted of approximately 10,000 prompts, evenly divided between harmful (~4,959) and neutral (~5,000) examples.  
Prompts were collected and adapted from several sources:
- **Categorical Harmful QA** (Hugging Face)
- **S-Eval** (Hugging Face)
- **ChatGPT-generated prompts**
- **Alpaca Dataset** (tatstu-lab)

The training data was carefully curated to ensure clean, balanced labels.  
Harmful prompts included unethical requests, misinformation, and logic traps, while neutral prompts spanned a broad range of benign topics to avoid overblocking safe queries.

{{INSERT DATASET CARD HERE)}}

### Training Procedure

The model was trained using a 3-layer convolutional neural network (CNN) architecture optimized for binary classification (harmful vs. neutral prompts).  
Training was conducted over 5 epochs with supervised learning using cross-entropy loss.

#### Preprocessing 

- Text was tokenized using a **simple whitespace split**.
- All text inputs were **lowercased**.
- Special characters and excessive whitespace were stripped.
- Inputs were **padded or truncated** to a maximum sequence length suitable for the CNN model.

Minimal preprocessing was used to preserve the natural structure of the prompts while ensuring model compatibility.

#### Training Hyperparameters

- **Optimizer:** Adam
- **Loss Function:** Cross-Entropy Loss
- **Learning Rate:** 1e-3
- **Batch Size:** 64
- **Epochs:** 5
- **Dropout Rate:** 0.5
- **Embedding Dimension:** 100
- **Convolution Kernel Sizes:** 3, 4, 5
- **Number of Filters per Kernel:** 100
- **Training regime:** fp32 (full precision 32-bit floating point)

## Evaluation

This section describes the evaluation protocols and provides the results.

### Testing Data, Factors & Metrics

#### Testing Data

The testing data consisted of a held-out subset from the combined dataset used for training.  
Sources for prompts included:
- **Categorical Harmful QA Dataset** (Hugging Face)
- **S-Eval Dataset** (Hugging Face)
- **ChatGPT-generated prompts**
- **Alpaca Dataset** (tatstu-lab)

The dataset contained approximately equal numbers of harmful and neutral prompts (~5,000 each), ensuring a balanced evaluation.  
Prompts were varied across topic domains to prevent overfitting to specific phrasing patterns.  
{INSERT DATASET CARD HERE}

#### Factors

Evaluation considered the following factors:
- **Prompt Type**: harmful vs. neutral prompts were evaluated separately.
- **Prompt Complexity**: simple (e.g., "List steps to bake.") vs. complex (e.g., "Explain how to bypass digital security systems.")
- **Subtle Wording Variations**: the model's ability to correctly classify prompts that were phrased similarly but had different intent.
- **Dataset Source Variation**: prompts originated from multiple datasets with slightly different construction styles (e.g., human-written vs. adversarially generated).

These factors helped assess whether the model generalized across diverse prompt styles and subtle adversarial attacks.


#### Metrics

The primary evaluation metrics used were:
- **Accuracy**: Measures the overall percentage of correctly classified prompts (harmful or neutral).
- **Precision**: Measures the proportion of predicted harmful prompts that were actually harmful (to reduce false positives).
- **Recall**: Measures the proportion of actual harmful prompts that were correctly identified (to reduce false negatives).
- **F1 Score**: Harmonic mean of precision and recall, balancing the two.

These metrics were chosen to ensure both types of errors (missing harmful prompts and wrongly flagging safe prompts) were minimized, given the safety-critical nature of the task.

{{ testing_metrics | default("[More Information Needed]", true)}}

### Results

The TextCNN achieved strong and stable performance across all training epochs:

| Epoch | Train Loss | Train Accuracy | Train Precision | Train Recall | Train F1 | Val Accuracy | Val Precision | Val Recall | Val F1 |
|:-----:|:----------:|:--------------:|:---------------:|:------------:|:--------:|:------------:|:-------------:|:----------:|:------:|
| 1 | 63.6247 | 87.42% | 85.88% | 89.42% | 87.62% | 95.18% | 93.52% | 97.04% | 95.25% |
| 2 | 27.1779 | 95.45% | 94.85% | 96.08% | 95.46% | 96.65% | 95.54% | 97.85% | 96.68% |
| 3 | 13.7224 | 97.65% | 97.47% | 97.81% | 97.64% | 96.05% | 93.85% | 98.52% | 96.13% |
| 4 | 9.1169 | 98.44% | 98.53% | 98.33% | 98.43% | 97.19% | 96.68% | 97.72% | 97.19% |
| 5 | 5.2029 | 99.18% | 99.08% | 99.28% | 99.18% | 97.26% | 97.31% | 97.18% | 97.24% |

The model quickly achieved high training and validation accuracy by Epoch 2, with minimal overfitting. Validation precision, recall, and F1 scores remained high throughout training, demonstrating strong generalization to unseen prompts.

#### Summary

The TextCNN model achieved a final validation accuracy of approximately **97.26%**, with a **precision of 97.31%**, **recall of 97.18%**, and **F1 score of 97.24%** on unseen data.  
It demonstrated consistent learning across epochs, strong generalization, and robust performance for classifying harmful versus neutral prompts.

## Model Examination [optional]

No specific model interpretability techniques (e.g., saliency maps, attention analysis) were applied.  
Evaluation focused on validation accuracy, loss reduction, and generalization consistency.


### Model Architecture and Objective

The model is a convolutional neural network (CNN) for binary text classification (harmful vs. neutral prompts).  
Architecture details:
- 100-dimensional word embeddings
- 3 convolutional layers with kernel sizes 3, 4, and 5
- 100 filters per convolution
- Max pooling
- Dropout layer with 0.5 probability
- Final fully connected (dense) layer mapping to two classes (harmful, neutral)

The objective is to pre-screen LLM user prompts for potentially harmful content, providing an additional safety layer before the prompt reaches an LLM.

### Compute Infrastructure

The model was trained on a local machine using a single CPU.  
Training was completed using a standard deep learning framework (PyTorch). 

#### Hardware

{{ hardware_requirements | default("[More Information Needed]", true)}}

#### Software

- Python 3.10
- PyTorch 2.0

## Model Card Authors [optional]

Ritu Gupta 

## Model Card Contact

ritumarygupta@gmail.com
