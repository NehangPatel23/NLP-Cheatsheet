# **NLP Comprehensive Cheat Sheet**

## **1. Introduction to NLP**

- **Definition**: Natural Language Processing (NLP) is a subfield of artificial intelligence focused on the interaction between computers and human language. It involves enabling computers to understand, interpret, and generate human language.

- **Applications**:
  - **Text Classification**: Categorizing text into predefined categories (e.g., spam detection).
  - **Sentiment Analysis**: Determining the sentiment expressed in text (positive, negative, neutral).
  - **Machine Translation**: Translating text from one language to another (e.g., Google Translate).
  - **Chatbots & Virtual Assistants**: Engaging in conversation and providing information (e.g., Siri, Alexa).
  - **Named Entity Recognition (NER)**: Identifying and classifying entities (people, organizations, locations) in text.

<br>

## **2. Levels of Language Analysis**

1. **Morphology**:
   - **Definition**: Study of the structure of words and their meaningful components.
   - **Example**:
     - Words "dogs" → root "dog" + suffix "s" (indicating plural).
     - Morphemes: "un-" (prefix), "happy" (root), "ness" (suffix).

2. **Syntax**:
   - **Definition**: Study of sentence structure and the rules that govern the formation of sentences.
   - **Example**:
     - Sentence structure: "The cat sits on the mat."
     - Parse trees can visually represent the syntactic structure of sentences.

3. **Semantics**:
   - **Definition**: Study of meaning in language and how words and phrases contribute to meaning.
   - **Example**:
     - The word "bank" can refer to a financial institution or the side of a river (lexical ambiguity).
     - Sentences can have different meanings based on the arrangement of words (e.g., "The chicken is ready to eat" vs. "The chicken is ready to eat us").

4. **Discourse**:
   - **Definition**: Analysis of longer texts and the relationships between sentences and how they create meaning.
   - **Example**:
     - Pronoun resolution: "John went to the bank. He withdrew money." (Resolving "he" to "John").
     - Co-reference resolution: Identifying when different words refer to the same entity in discourse.

5. **Pragmatics**:
   - **Definition**: Study of context-dependent meaning and how language is used in practical situations.
   - **Example**:
     - "Can you pass the salt?" implies a request rather than merely asking about ability.

<br>

## **3. Challenges in NLP**

- **Ambiguity**:
  - **Lexical Ambiguity**: A word has multiple meanings (e.g., "bat" as an animal or a sports implement).
  - **Syntactic Ambiguity**: A sentence can be parsed in different ways (e.g., "I saw the man with the telescope").
  
- **Context Dependence**: The meaning of phrases can change based on context (e.g., "I'm cold" may refer to temperature or emotional state).

- **Variability**: Variations in language usage (slang, regional dialects) can complicate processing.

<br>

## **4. N-gram Models**

- **Definition**: Statistical models that predict the next word in a sequence based on the previous $n-1$ words.

- **Types**:
  - **Unigram**: $P(w_i)$
    - **Example**: Probability of the word "dog" occurring in a corpus.
  
  - **Bigram**: $P(w_i | w_{i-1})$
    - **Example**:
      $$P(\text{the} | \text{dog}) = \frac{\text{Count(dog, the)}}{\text{Count(dog)}}$$

  - **Trigram**: $P(w_i | w_{i-2}, w_{i-1})$
    - **Example**:
      $$P(\text{barks} | \text{the}, \text{dog}) = \frac{\text{Count(the, dog, barks)}}{\text{Count(the, dog)}}$$

- **General Formula**:
  
  $$P(w_1, w_2, \dots, w_n) = P(w_1) P(w_2 | w_1) \cdots P(w_n | w_{n-1})$$
  
  - **Example**: For the sentence "I am happy":
    
    $$P(\text{I}, \text{am}, \text{happy}) = P(\text{I}) P(\text{am} | \text{I}) P(\text{happy} | \text{am})$$

- **Markov Assumption**:
  - Only the last $n-1$ words are used to predict the next word. This simplifies the model but may lose long-range dependencies.

- **Chain Rule:** The joint probability of a sequence can be computed using:
  
$$P(w_1, w_2, \dots, w_n) = P(w_1)P(w_2 | w_1) \cdots P(w_n | w_1, w_2, \dots, w_{n-1})$$

<br>

## **5. Smoothing Techniques**

- **Purpose**: Address zero probabilities in n-grams when an observed word sequence does not appear in the training data.

- **Laplace (Add-1) Smoothing**:
  
  $$P(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i) + 1}{C(w_{i-1}) + |V|}$$
  
  - **Example**:
    - If "the dog" appears 5 times and "the" appears 10 times with a vocabulary size of 1000:
      
      $$P(\text{dog} | \text{the}) = \frac{5 + 1}{10 + 1000} = \frac{6}{1010}$$

- **Add-α Smoothing**:
  
  $$P(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i) + \alpha}{C(w_{i-1}) + \alpha |V|}$$

  - **Example**:
    - For $\alpha = 0.5$, if "happy" follows "very" 4 times in the corpus:
      
      $$P(\text{happy} | \text{very}) = \frac{4 + 0.5}{10 + 0.5 \times 1000} = \frac{4.5}{510}$$

- **Linear Interpolation Smoothing**:
  
  $$P(w_i | w_{i-1}) = \lambda_1 P_{1}(w_i | w_{i-1}) + \lambda_2 P_{2}(w_i | w_{i-2}, w_{i-1}) + \ldots$$
  
  - **Example**: Combining unigram, bigram, and trigram probabilities with weights $\lambda_1, \lambda_2$ such that $\lambda_1 + \lambda_2 + \ldots = 1$.

- **Handling Unknown Words**: Assign a small probability to unseen words using techniques like introducing an `<UNK>` token in training.
  - **Practical Example**: When training a model on text, any word not in the vocabulary is replaced with `<UNK>`, and the model learns probabilities for this token.

<br>

## **6. Naive Bayes Classifier**

- **Formula**:
  
  $$P(y | X) = \frac{P(X | y) P(y)}{P(X)}$$
  
- **Decision Rule**:
  
  $$y^* = \arg\max_y P(y) \prod P(w_i | y)$$
  
- **Example**: Classifying emails as spam or not:
  - $P(\text{spam}) = 0.3, P(\text{free} | \text{spam}) = 0.5, P(\text{money} | \text{spam}) = 0.7$
  - Calculate:
    
    $$P(\text{spam} | \text{email}) = 0.3 \times 0.5 \times 0.7 = 0.105$$

- **Handling Unknown Words**: Assign a small probability for unseen words during classification to prevent zero probabilities.
  - **Example**: Use a small constant $\epsilon$ (epsilon) for unknown words.

- **Stop Words**: Often removed during preprocessing as they carry less semantic meaning (e.g., "and", "the", "is").

- **Log Probability**: To avoid numerical underflow, probabilities are often converted into log space:
  
  $$\log P(X | y) = \log P(w_1 | y) + \log P(w_2 | y) + \cdots$$
  
  - **Example**: If the probabilities for three words are 0.5, 0.3, and 0.1, we calculate:
    
    $$\log P(X | y) = \log(0.5) + \log(0.3) + \log(0.1) = -0.69 - 1.20 - 2.30 = -4.19$$

<br>

## **7. Evaluation Metrics**

- **Accuracy**:
  
  $$\text{Accuracy} = \frac{\text{True Positives} + \text{True Negatives}}{\text{Total Instances}}$$

  - **Example**:
    If a model correctly classifies 80 out of 100 emails:
      
    $$\text{Accuracy} = \frac{80}{100} = 0.8$$

- **Precision**:
  
  $$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$$
  
  - **Example**:
    If the model predicts 40 spam emails correctly but also classifies 10 non-spam as spam:
      
      $$\text{Precision} = \frac{40}{40 + 10} = 0.8$$

- **Recall**:
  
  $$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$
  
  - **Example**:
    If 50 emails are actually spam, but the model only identifies 40 correctly:
      
      $$\text{Recall} = \frac{40}{40 + 10} = 0.8$$

- **F1 Score**:
  
  $$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
  
  - **Example**: Given Precision = 0.8 and Recall = 0.8:
    
    $$F1 = 2 \times \frac{0.8 \times 0.8}{0.8 + 0.8} = 0.8$$

- **Micro and Macro-Averaging**:
  - **Micro-Averaging**: Aggregates contributions from all classes to compute the average metric across the entire dataset.
    - **Example:** Given a confusion matrix that presents how many items that have labels "Comedy", "Action", and "Romance":
      
    $$\text{Micro-averaged Recall} = \dfrac{TP_{action} + TP_{comedy} + TP_{romance}}{TP_{action} + TP_{comedy} + TP_{romance} + FN_{action} + FN_{comedy} + FN_{romance}}$$

      $$\text{Micro-averaged Precision} = \dfrac{TP_{action} + TP_{comedy} + TP_{romance}}{TP_{action} + TP_{comedy} + TP_{romance} + FP_{action} + FP_{comedy} + FP_{romance}}$$
    
  - **Macro-Averaging**: Computes the metric independently for each class and then takes the average, treating all classes equally.
    - **Example:** Given a confusion matrix that presents how many items that have labels "Comedy", "Action", and "Romance":
      
      $$\text{Macro-averaged Recall} = \dfrac{1}{3} \times (Recall_{action} + Recall_{comedy} + Recall_{romance})$$
      
      $$\text{Macro-averaged Precision} = \dfrac{1}{3} \times (Precision_{action} + Precision_{comedy} + Precision_{romance})$$

- **Cross-Validation**: Divides the dataset into multiple parts (folds), training and testing the model on different segments to ensure robustness and avoid overfitting.
  - **Example**: In 5-fold cross-validation, the data is split into five equal parts, and the model is trained on four parts while testing on the fifth. This process is repeated five times, and the results are averaged.

- **Log Loss**: A metric for evaluating the performance of classification models, particularly in the context of probabilities:
  
  $$\text{Log Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left[y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)\right]$$

<br>

## **8. Perplexity and Cross-Entropy**

- **Perplexity**: Measures how well a probability distribution predicts a sample. Lower perplexity indicates a better model.
  
  $$\text{Perplexity} = 2^{-\frac{1}{N} \sum \log P(w_i | w_1, ..., w_{i-1})}$$
  
  - **Example**: For a model predicting 0.1 for each word in a 5-word sentence:
    
    $$\text{Perplexity} = 2^{-\frac{1}{5} \log_2 (0.1^5)} \approx 10$$

- **Cross-Entropy**: Quantifies the difference between two probability distributions. It measures the performance of a model whose output is a probability value between 0 and 1.
  
  $$H(p, q) = - \sum p(x) \log q(x)$$
  
  - **Example**: If the true distribution $p(x)$ has probabilities [0.6, 0.4] and the predicted distribution $q(x)$ has probabilities [0.7, 0.3]:
    
    $$H(p, q) = -[0.6 \log(0.7) + 0.4 \log(0.3)]$$

<br>

## **9. Zipf's Law and Heaps' Law**

- **Zipf’s Law**: Describes the distribution of word frequencies in a corpus. It states that the frequency of a word is inversely proportional to its rank in the frequency table.
  - **Example**: The second most frequent word occurs approximately half as often as the most frequent word.
  
- **Heaps’ Law**: Predicts that the vocabulary size in a corpus grows with the size of the corpus.
  - **Formula**:
    
    $V(n) = k n^β$
    
    where $V(n)$ is the vocabulary size, $n$ is the number of words in the corpus, and $k$ and $β$ are constants.

<br>

## **10. Viterbi Algorithm**

- **Purpose**: Used to find the most likely sequence of hidden states in a Hidden Markov Model (HMM).

- **Definitions**:
  - **States**: The hidden states in the model.
  - **Observations**: The observed outputs.
  - **Transition Probabilities**: The probabilities of moving from one state to another.
  - **Emission Probabilities**: The probabilities of observing a certain output from a given state.
  
- **Steps**:
  1. **Initialization**: Set the initial probabilities for each state based on prior knowledge or uniform distribution.
   
     $$\delta_1(j) = \pi_j \cdot b_j(o_1)$$
  
    Where $\pi_j$ is the initial probability of state $j$ and $b_j(o_1)$ is the emission probability of the first observation given state $j$.

  2. **Recursion**: For each state at time $t$, compute the maximum probability of reaching that state using:

     $$\delta_t(j) = \max_i [\delta_{t-1}(i) \times a_{ij}] \times b_j(o_t)$$
     
  4. **Termination**: Choose the state with the highest probability at the final time step.
  5. **Backtracking**: Trace back through the states to find the most likely sequence.

- **Example**: For POS tagging:
  - **States**: Noun, Verb, etc.
  - **Observations**: Words in a sentence.
  - Use transition probabilities (e.g., $a_{ij}$: probability of transitioning from state $i$ to state $j$) and emission probabilities (e.g., $b_j(o_t)$: probability of word $o_t$ given state $j$) to determine the most likely sequence of POS tags.

<br>

## **11. Hidden Markov Models (HMM) Concepts**

**Forward Algorithm**: This algorithm computes the probability of an observation sequence given a model:

$$\alpha_t(j) = \sum_i \alpha_{t-1}(i) \cdot a_{ij} \cdot b_j(o_t)$$

<br>

## **12. Word Representations**

- **Word Representations**:
  - **One-Hot Encoding**: Represents each word as a binary vector. Not efficient for large vocabularies due to sparsity.
  - **Word Embeddings**: Dense vector representations of words that capture semantic meanings (e.g., Word2Vec, GloVe).
    - **Example**: In Word2Vec, words with similar meanings are closer in vector space.

<br>

## **13. Error Analysis and Cross-Validation**

- **Error Analysis**: Involves examining the types of errors made by models (false positives vs. false negatives) to better understand model weaknesses.
  - **Word Error Rate (WER)**: Measures how often words are predicted incorrectly.
  - **Confusion Matrix**: Used to analyze classification errors across multiple classes.

- **Cross-Validation**: Divides the dataset into multiple parts (folds), training and testing the model on different segments to ensure robustness and avoid overfitting.
  - **Example**: In 5-fold cross-validation, the data is split into five equal parts, and the model is trained on four parts while testing on the fifth. This process is repeated five times, and the results are averaged.

<br>

## **14. Probabilistic Classifiers**

- **Definition**: Probabilistic classifiers model the relationship between features and class labels using probabilities.
- **Joint Probability**: For generative classifiers, the joint probability of features and class labels can be expressed as:
  
$$P(X, Y) = P(Y) P(X | Y)$$

- **Example**: In a spam classification task, $Y$ could be “spam” or “not spam,” while $X$ represents the features extracted from the email content.

<br>

## **15. Conditional Random Fields (CRF)**

- **Definition**: Conditional Random Fields are a type of discriminative model used for structured prediction, particularly in tasks like sequence labeling.
- **Partition Function**:
  
$$ Z(\mathbf{x}) = \sum_{\mathbf{y}} \exp\left(\sum_k \lambda_k f_k(\mathbf{x}, \mathbf{y})\right) $$

- **Example**: In a named entity recognition task, $\mathbf{x}$ could represent the sequence of words, and $\mathbf{y}$ could represent the corresponding tags.

<br>

## **16. Neural Networks and Machine Learning Basics**

- **Definition**: Neural networks are a class of machine learning models inspired by the structure of the human brain, used for a variety of tasks, including classification and regression.
- **Cost Function**: A common cost function used for training neural networks is the cross-entropy loss, defined as:
  
$$ L(y, \hat{y}) = -\sum_{i=1}^{N} \left[y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)\right] $$

- **Example**: This formula helps in measuring the difference between the predicted probabilities and the actual class labels in classification tasks.

<br>

## **17. Statistical POS Tagging**

- **Definition**: Statistical Part-of-Speech (POS) tagging uses probabilistic models to assign parts of speech to each word in a sentence.
- **Probability Model**:
  
$$ P(y | x) = \frac{P(x | y) P(y)}{P(x)} $$

- **Example**: Here, $y$ represents the sequence of POS tags, and $x$ represents the words in the sentence.

<br>

## **18. TF-IDF (Term Frequency-Inverse Document Frequency)**

- **Term Frequency (TF):**

    The term frequency of a term $t$  in a document $d$  is calculated as:
  
    $$\text{TF}(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d}$$

- **Inverse Document Frequency (IDF):**

    The inverse document frequency of a term  $t$  is calculated as:
  
    $$\text{IDF}(t) = \log\left(\frac{N}{\text{DF}(t)}\right)$$
  
    Where:
  - $N$  = Total number of documents in the corpus.
  - $\text{DF}(t)$  = Number of documents containing the term  $t$ .

- **TF-IDF Score:**

    The TF-IDF score for a term  $t$  in document  $d$  is calculated as:
  
    $$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

<br>

## **19. Pointwise Mutual Information (PMI)**

- **Probability Definitions:**
  - $P(x)$: The probability that a sentence contains at least one instance of the term  $x$.
  - $P(x, y)$: The probability that a sentence contains at least one instance of both terms  $x$  and  $y$.

- **PMI Calculation:**

    The PMI between terms  $x$  and  $y$  is calculated as:
  
    $$\text{PMI}(x, y) = \log_2\left(\frac{P(x, y)}{P(x) \times P(y)}\right)$$
  
    This measures how much more likely the two terms are to co-occur than would be expected by chance.

<br>

## **20. Neural Network Calculations**

- **Sigmoid Activation Function:**

    The output  $Y$  of a neuron using the sigmoid activation function is calculated as:
  
    $$Y = \sigma(Z) = \frac{1}{1 + e^{-Z}}$$
  
    Where  $Z$  is the weighted sum of inputs plus bias:
  
    $$Z = W \cdot X + b$$
  
    Where:
  - $W$  = Weight vector.
  - $X$  = Input vector.
  - $b$  = Bias.

- **Tanh Activation Function:**

    The output  $Y$  of a neuron using the tanh activation function is calculated as:
  
    $$Y = \tanh(Z) = \frac{e^{Z} - e^{-Z}}{e^{Z} + e^{-Z}}$$

- **Softmax Function:**

    The softmax function is used to convert raw scores into probabilities:
  
    $$\text{softmax}(Z_i) = \frac{e^{Z_i}}{\sum_{j} e^{Z_j}}$$
  
    Where  $Z_i$  is the  $i^{th}$  element of the input vector  $Z$.

- **Cross-Entropy Loss Function:**

    The cross-entropy loss between the true label $y$  and the predicted probabilities $\hat{y}$ is calculated as:
  
    $$L(y, \hat{y}) = -\sum_{i=1}^{N} \left[y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)\right]$$
