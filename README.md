
# Sentiment and News Categorization using Artificial Neural Networks (ANN)

This repository contains a project that applies artificial neural networks (ANN) to perform sentiment analysis on e-commerce product comments and news categorization. The models were developed using two different datasets, focusing on analyzing sentiments in online comments and categorizing news articles into predefined categories. 

## Project Overview (December 2023)

### 1. **Sentiment Analysis**
   - **Dataset**: The project uses Kaggle's *e-ticaret_urun_comments* dataset, which contains product reviews labeled as:
     - 0: Negative
     - 1: Positive
     - 2: Neutral
   - **Goal**: Develop an ANN model to classify the sentiments expressed in product reviews. 
   
### 2. **News Categorization**
   - **Dataset**: This part of the project uses a dataset containing news articles and their associated categories. 
   - **Goal**: Build an ANN model to classify news articles into predefined categories using word embeddings generated with Word2Vec.

## Features

- **Sentiment Analysis Model**:
  - Tokenization using Keras `Tokenizer`.
  - Sequential neural network with embedding and dense layers.
  - Model evaluated with accuracy metrics on the test set.
  
- **News Categorization Model**:
  - Tokenization using Word2Vec for embedding representation.
  - Neural network using Word2Vec embeddings and dense layers.
  - Dropout layers for regularization.
  - Evaluation of model performance on the test set.

## Datasets

- **Sentiment Analysis Dataset**: [e-ticaret_urun_comments](https://www.kaggle.com) 
- **News Categorization Dataset**: News dataset with related categories.

## Project Workflow

### Sentiment Analysis

1. **Load Dataset**: The dataset is loaded from a CSV file with proper delimiter handling.
2. **Preprocessing**:
   - Label encoding for target variable (`y`).
   - Tokenization and padding for input text (`X`).
3. **Model Architecture**:
   - Embedding layer to learn word representations.
   - Dense and softmax layers for classification.
4. **Training**:
   - The model is compiled using Adam optimizer and categorical cross-entropy as the loss function.
   - Trained on the preprocessed data for 10 epochs with validation.
5. **Evaluation**:
   - Test data accuracy is printed after model evaluation.

### News Categorization

1. **Load Dataset**: Load and preprocess the dataset.
2. **Word2Vec Tokenization**: Text data is tokenized using Word2Vec, and embedding vectors are generated for each word.
3. **Model Architecture**:
   - ANN model built using Word2Vec embeddings, dense layers, and dropout for regularization.
   - Softmax activation for final classification into 7 categories.
4. **Training**: 
   - The model is compiled using Adam optimizer and sparse categorical cross-entropy.
   - Trained on padded sequences for 10 epochs.
5. **Evaluation**:
   - Test accuracy is computed and printed.

## Requirements

To run this project, you will need the following libraries:

```bash
pip install tensorflow
pip install keras
pip install gensim
pip install scikit-learn
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Sentiment-News-ANN.git
   ```
   
2. Navigate to the project directory:
   ```bash
   cd Sentiment-News-ANN
   ```

3. Run the Jupyter Notebook files to execute the models:
   - Sentiment Analysis: `Sentiment_Analysis.ipynb`
   - News Categorization: `News_Categorization.ipynb`

4. Use `tokenization.py` for the tokenization process in the ANN models.

## Model Performance

- **Sentiment Analysis Model Accuracy**: `~XX%`
- **News Categorization Model Accuracy**: `~XX%`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Datasets from Kaggle.
- TensorFlow and Keras libraries.
