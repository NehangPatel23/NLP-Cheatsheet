# NLP Comprehensive Study Guide

Author: Nehang Patel

Source: [NLP Lecture Slides](https://github.com/NehangPatel23/NLP-Cheatsheet/blob/main/NLP%20Lecture%20Slides.pdf)

<br>

## Table of Contents

- **[I - What is Natural Language Processing (NLP)?](#i---what-is-natural-language-processing-nlp)**
  - [1. How do we communicate with computers?](#1-how-do-we-communicate-with-computers)
  - [2. Applications of NLP](#2-applications-of-nlp)
  - [3. Levels of Analysis and Knowledge in NLP](#3-levels-of-analysis-and-knowledge-in-nlp)
  - [4. Why is NLP Difficult?](#4-why-is-nlp-difficult)
  - [5. Dealing with Ambiguity in NLP](#5-dealing-with-ambiguity-in-nlp)

- **[II - Corpus and Morphology](#ii---corpus-and-morphology)**
  - [1. The Traditional NLP Pipeline](#1-the-traditional-nlp-pipeline)
  - [2. Corpus in NLP](#2-corpus-in-nlp)
  - [3. Text Normalization and Tokenization](#3-text-normalization-and-tokenization)
  - [4. Morphology: Understanding Word Structure](#4-morphology-understanding-word-structure)
  - [5. Handling Morphological Variants and Normalization Errors](#5-handling-morphological-variants-and-normalization-errors)
  - [6. Summary](#summary---corpus--morphology)
  
- **[III - N-gram Language Models](#iii---n-gram-language-models)**
  - [1. Introduction to Language Models](#1-introduction-to-language-models)
  - [2. Probability Theory Recap](#2-probability-theory-recap)
  - [3. Language Models in NLP](#3-language-models-in-nlp)
  - [4. N-gram Language Models](#4-n-gram-language-models)
  - [5. Computing N-gram Probabilities](#5-computing-n-gram-probabilities)
  - [6. Practical Considerations for N-gram Models](#6-practical-considerations-for-n-gram-models)
  - [7. Evaluation of Language Models](#7-evaluation-of-language-models)
  - [8. Summary](#summary---n-grams)

- **[IV - Text Classification - Naive Bayes](#iv---text-classification---naive-bayes)**
  - [1. Naive Bayes Classifier for Text Classification](#naive-bayes-classifier-for-text-classification)
  - [2. Example - Sentiment Analysis](#example-sentiment-analysis)
  - [3. Dealing with Unseen Words and Zero Probabilities](#dealing-with-unseen-words-and-zero-probabilities)
  - [4. Practical Considerations](#practical-considerations)
  - [5. Why Use Naive Bayes?](#why-use-naive-bayes)
  - [6. Summary](#summary---naive-bayes-classification)

- **[V - Logistic Regression](#v---logistic-regression)**
  - [1. Training a Logistic Regression Model](#training-a-logistic-regression-model)
  - [2. Cross-Validation and Evaluating Classifiers](#cross-validation-and-evaluating-classifiers)
  - [3. Evaluation Metrics](#evaluation-metrics)
  - [4. Word Error Rate (WER)](#word-error-rate-wer-1)
  - [5. Probabilistic Classifiers](#probabilistic-classifiers)
  - [6. Sigmoid Function](#the-sigmoid-function)
  - [7. Multinomial Logistic Regression](#multinomial-logistic-regression)
  - [8. The Softmax Function](#softmax-function)
  - [9. Logistic Regression vs. Naive Bayes](#logistic-regression-vs-naive-bayes)
  - [10. Gradient Descent](#gradient-descent)
  - [11. Summary](#summary---logistic-regression)

- **[VI - Part-of-Speech Tagging](#vi---part-of-speech-tagging)**
  - [1. Creating a POS Tagger](#creating-a-pos-tagger)
  - [2. Inter-Annotator Agreement (IAA)](#inter-annotator-agreement-iaa)
  - [3. Kappa Statistic](#the-kappa-statistic)
  - [4. Rule-Based POS Tagging](#rule-based-pos-tagging)
  - [5. Statistical POS Tagging](#statistical-pos-tagging)
  - [6. POS Tagging with Generative Models](#pos-tagging-with-generative-models)
  - [7. Hidden Markov Model (HMM)](#hidden-markov-model-hmm)

- **[VII - Viterbi](#vii---viterbi)**
  - [1. Formalizing the Tagging Problem](#formalizing-the-tagging-problem)
  - [2. Hidden Markov Model (HMM) Overview](#hidden-markov-model-hmm-overview)
  - [3. The Viterbi Algorithm](#the-viterbi-algorithm)
  - [4. Example of Viterbi Algorithm](#example-of-viterbi-algorithm)
  - [5. Greedy Algorithm vs. Viterbi](#greedy-algorithm-vs-viterbi)
  - [6. Why Enumeration Won’t Work](#why-enumeration-wont-work)
  - [7. Applications of the Viterbi Algorithm](#applications-of-the-viterbi-algorithm)
  - [8. Probabilistic Tag Assignment](#probabilistic-tag-assignment)
  - [9. Forward and Backward Probabilities](#forward-and-backward-probabilities)
  - [10. Combining Forward and Backward Probabilities](#combining-forward-and-backward-probabilities)
  - [11. Example of Forward and Backward Probabilities](#example-of-forward-and-backward-probabilities)
  - [12. Summary](#summary---viterbi-algorithm)

- **[VIII - Sequence Labelling](#viii---sequence-labelling)**
  - [1. Shallow Parsing](#1-shallow-parsing)
  - [2. Named Entity Recognition (NER)](#2-named-entity-recognition-ner)
  - [3. Rule-Based vs. Machine Learning NER](#3-rule-based-vs-machine-learning-ner)
  - [4. Common Types of Rules for NER](#4-common-types-of-rules-for-ner)
  - [5. Machine Learning Models for Sequence Labeling](#5-machine-learning-models-for-sequence-labeling)
  - [6. BIO Tagging for NER](#6-bio-tagging-for-ner)
  - [7. Hidden Markov Models (HMM) for NER](#7-hidden-markov-models-hmm-for-ner)
  - [8. Maximum Entropy Markov Models (MEMM)](#8-maximum-entropy-markov-models-memm)
  - [9. Conditional Random Fields (CRF)](#9-conditional-random-fields-crf)
  - [10. Summary - HMM vs. MEMM vs. CRF](#10-summary---hmm-vs-memm-vs-crf)
  - [11. Example NER System: MENERGI](#example-ner-system-menergi)

- **[IX - Lexical Semantics](#ix---lexical-semantics)**
  - [1. Introduction to Lexical Semantics](#1-introduction-to-lexical-semantics)
  - [2. Word Senses](#2-word-senses)
  - [3. Homonymy and Polysemy](#3-homonymy-and-polysemy)
  - [4. Metonymy](#4-metonymy)
  - [5. Synonyms and Antonyms](#5-synonyms-and-antonyms)
  - [6. Hypernymy and Hyponymy](#6-hypernymy-and-hyponymy)
  - [7. Meronymy and Holonymy](#7-meronymy-and-holonymy)
  - [8. WordNet](#8-wordnet)
  - [9. Word Similarity](#9-word-similarity)
  - [10. Path-Based Similarity](#10-path-based-similarity)
  - [11. Dictionary-Based Similarity](#11-dictionary-based-similarity)
  - [12. Word Sense Disambiguation (WSD)](#12-word-sense-disambiguation-wsd)
  - [13. Lesk Algorithm](#13-lesk-algorithm)
  - [14. Bootstrapping Algorithm for Word Sense Disambiguation](#14-bootstrapping-algorithm-for-word-sense-disambiguation)
  - [14. Summary](#summary---lexical-semantics)

- **[X - Distributional Representations](#x---distributional-representations)**
  - [1. The Distributional Hypothesis](#1-the-distributional-hypothesis)
  - [2. Distributional Similarity](#2-distributional-similarity)
  - [3. Distance and Similarity Metrics](#3-distance-and-similarity-metrics)
    - [i. Manhattan Distance](#1-manhattan-distance)
    - [ii. Euclidean Distance](#2-euclidean-distance)
    - [3. Jaccard Similarity](#3-jaccard-similarity)
    - [4. Cosine Similarity](#4-cosine-similarity)
  - [4. Term-Document Matrix](#4-term-document-matrix)
  - [5. TF-IDF](#5-tf-idf-term-frequency-inverse-document-frequency)
  - [6. Word-Word Co-occurrence Matrix](#6-word-word-co-occurrence-matrix)
  - [7. Pointwise Mutual Information (PMI)](#7-pointwise-mutual-information-pmi)
    - [Pointwise Mutual Information (PMI) Example](#pointwise-mutual-information-pmi-example)
  - [8. Sparse vs. Dense Vectors](#8-sparse-vs-dense-vectors)
  - [9. Summary](#summary---distributional-representations)

- **[XI - Word Embeddings](#xi---word-embeddings)**
  - [1. Sparse vs. Dense Vectors](#1-sparse-vs-dense-vectors)
  - [2. How to Obtain Dense Vectors?](#2-how-to-obtain-dense-vectors)
  - [3. Word2Vec](#3-word2vec)
  - [4. Training Word Embeddings](#4-training-word-embeddings)
  - [5. Learning via Gradient Descent](#5-learning-via-gradient-descent)
  - [6. Embedding Matrices](#6-embedding-matrices)
  - [7. Evaluating Word Embeddings](#7-evaluating-word-embeddings)
  - [8. Word Embeddings and Historical Change](#8-word-embeddings-and-historical-change)
  - [9. Visualizing Word Embeddings](#9-visualizing-word-embeddings)
  - [10. Resources for Pretrained Word Embeddings](#10-resources-for-pretrained-word-embeddings)
  - [11. Summary](#summary---word-embeddings)

- **[XII - Neural Networks for NLP](#xii---neural-networks-for-nlp)**
  - [1. Neural Networks: Overview](#1-neural-networks-overview)
  - [2. Non-Linear Activation Functions](#2-non-linear-activation-functions)
  - [3. Perceptrons](#3-perceptrons)
  - [4. XOR Problem](#4-xor-problem)
  - [5. Multi-Layer Neural Networks](#5-multi-layer-neural-networks)
  - [6. Backpropagation and Gradient Descent](#6-backpropagation-and-gradient-descent)
  - [7. Activation Functions: Sigmoid, Tanh, and ReLU](#7-activation-functions-sigmoid-tanh-and-relu)
  - [8. Softmax for Multiclass Classification](#8-softmax-for-multiclass-classification)
  - [9. Feedforward Neural Networks for NLP](#9-feedforward-neural-networks-for-nlp)
  - [10. Word Embeddings as Features](#10-word-embeddings-as-features)
  - [11. Neural Language Models](#11-neural-language-models)

- **[XIII - Recurrent Neural Networks (RNNs)](#xiii---recurrent-neural-networks-rnns)**
  - [1. Introduction to Sequence Labeling with Neural Networks](#1-introduction-to-sequence-labeling-with-neural-networks)
  - [2. Recurrent Neural Networks (RNNs)](#2-recurrent-neural-networks-rnns)
  - [3. Illustration of Recurrent Process](#3-illustration-of-recurrent-process)
  - [4. RNN vs. Feedforward Neural Networks](#4-rnn-vs-feedforward-neural-networks)
  - [5. RNNs for Language Modeling](#5-rnns-for-language-modeling)
  - [6. Improvements Over Feedforward Neural Networks](#6-improvements-over-feedforward-neural-networks)
  - [7. Training an RNN Language Model](#7-training-an-rnn-language-model)
  - [8. Weight Tying in RNNs](#8-weight-tying-in-rnns)
  - [9. RNNs for Other NLP Tasks](#9-rnns-for-other-nlp-tasks)
  - [10. Stacked and Bidirectional RNNs](#10-stacked-and-bidirectional-rnns)
  - [11. RNN Variants: LSTMs and GRUs](#11-rnn-variants-lstms-and-grus)
  - [12. Vanishing and Exploding Gradients](#12-vanishing-and-exploding-gradients)
  - [13. LSTMs](#13-lstms)
  - [14. How LSTM Solves Vanishing Gradients](#14-how-lstm-solves-vanishing-gradients)
  
- **[XIV - Machine Translation and Seq2Seq Models](#xiv---machine-translation-and-seq2seq-models)**
  - [1. Machine Translation (MT)](#1-machine-translation-mt)
  - [2. The Vauquois Triangle](#2-the-vauquois-triangle)
  - [3. Rule-Based Machine Translation (RBMT)](#3-rule-based-machine-translation-rbmt)
  - [4. Statistical Machine Translation (SMT)](#4-statistical-machine-translation-smt)
  - [5. Evaluation of Machine Translation](#5-evaluation-of-machine-translation)
  - [6. BLEU Score](#6-bleu-score)
  - [7. Neural Machine Translation (NMT)](#7-neural-machine-translation-nmt)
  - [8. Training Seq2Seq Models](#8-training-seq2seq-models)
  - [9. Decoding in Seq2Seq Models](#9-decoding-in-seq2seq-models)
  - [10. Pros and Cons of NMT](#10-pros-and-cons-of-nmt)
  - [11. Beam Search Decoding](#11-beam-search-decoding)

---

<br>

## I - What is Natural Language Processing (NLP)?

NLP, or Natural Language Processing, is a subfield of **linguistics**, **computer science**, and **artificial intelligence**. The goal of NLP is to enable computers to **understand**, **interpret**, and **respond to human language** in a way that is valuable. This typically involves building computational models that mimic aspects of how humans understand and generate language.

### 1. **How do we communicate with computers?**

Computers don’t inherently understand human language. Instead, they process inputs as a series of instructions in **machine language**. Initially, human-computer communication was highly structured, based on specific **commands** that the computer could recognize (e.g., using keyboards to issue commands like "open file").

However, one of the ultimate goals in computer science is for machines to understand **natural language**, which is the type of language humans use in everyday speech (e.g., English, Spanish, Mandarin). Imagine a world where you could simply talk to a machine as you do with another person, and it would comprehend and respond appropriately.

### 2. **Applications of NLP**

NLP has a wide variety of applications that impact our daily lives, from personal assistants like **Siri** and **Alexa** to language translation tools like **Google Translate**. Below are some notable applications of NLP:

- **Virtual Assistants:** These include software like Siri, Alexa, and Google Assistant, which can perform tasks or provide information based on voice commands.
- **Machine Translation:** This involves converting text or speech from one language to another. For example, Google Translate.
- **Question Answering:** Systems like IBM's **Watson**, which can interpret and answer questions posed in natural language (e.g., playing Jeopardy), fall under this category.
- **Text Classification:** This involves assigning predefined categories to a piece of text. For example, spam detection in emails, sentiment analysis of reviews.
- **Information Extraction:** Automatically identifying structured information from unstructured text. For example, finding names, dates, and locations in documents.
- **Sentiment Analysis:** Determining the sentiment expressed in a piece of text, whether it is positive, negative, or neutral.
- **Text Summarization:** Condensing long pieces of text into shorter summaries.
  
Each of these applications requires sophisticated computational models and a deep understanding of linguistic structure.

### 3. **Levels of Analysis and Knowledge in NLP**

Understanding language involves various levels of analysis. Each level builds on the previous one to understand more complex aspects of language:

- **Morphology:** This refers to how words are constructed. For example, English words often have prefixes (like "un-" in "undo") and suffixes (like "-ing" in "running").
  
- **Syntax:** This is the study of the structural relationships between words in a sentence. In English, syntax helps determine the role each word plays (e.g., subject, verb, object).

- **Semantics:** This level deals with the meaning of words and phrases. It ensures that when we read a sentence, we understand what it is saying.

- **Discourse:** This looks at how sentences are connected in a text, and how meaning is developed across larger units of language like paragraphs or documents.

- **Pragmatics:** This is concerned with the purpose of language in context. For instance, if someone says, “It’s cold in here,” they might not just be stating a fact—they could be indirectly requesting you to close the window.

- **World Knowledge:** A human's understanding of the world and common sense plays a significant role in language processing. For example, if someone says, "I dropped the glass, and it broke," world knowledge tells us that glasses are fragile and typically break when dropped.

### 4. **Why is NLP Difficult?**

Natural language is highly **ambiguous**, making NLP a challenging task. Some examples of ambiguities include:

- **Structural ambiguity:** A sentence like "I saw the man with the telescope" could mean either that you used a telescope to see the man, or that the man had a telescope.
  
- **Lexical ambiguity:** Many words have multiple meanings. For example, the word "bank" can refer to a financial institution or the side of a river.
  
- **Referential ambiguity:** Pronouns like "he" or "it" in a sentence can refer to multiple entities, and understanding the correct reference requires context. For instance, "John saw Terry, and he waved"—who waved?

- **Phonological ambiguity:** Spoken words may sound the same but have different meanings (homophones). For example, "night" and "knight."

Handling these ambiguities requires complex models and often large amounts of training data.

### 5. **Dealing with Ambiguity in NLP**

One of the ways NLP models handle ambiguity is by relying on **statistical methods**. These methods involve training models on large corpora of text data so that they can learn patterns and make predictions based on probability. For example, when Google Translate processes a sentence, it doesn't just look up word translations—it uses statistical models to figure out the most likely translation based on previous data.

<br>

## II - Corpus and Morphology

### **1. The Traditional NLP Pipeline**

When building an NLP system, the process typically follows a series of well-defined steps. Each step deals with progressively more complex language representations. Understanding this **NLP pipeline** is crucial, as each step influences the one after it.

#### **Stages in the Traditional Pipeline:**

1. **Tokenizer/Segmenter:**
   - **Goal:** Break a continuous text stream into discrete units, typically **words** or **sentences**.
   - In languages like **English**, tokenization appears straightforward since words are generally separated by spaces. However, this isn’t true for all languages.
   - **Example of Challenges:**
     - **Contractions:** "didn't" might be split into "did" and "n't," or it might not be split at all.
     - **Punctuation:** Does "U.S.A." get tokenized as "USA," or does it remain with the periods intact?
     - **Non-Space-Segmented Languages:** In languages like Chinese or Japanese, tokenization is more challenging because words aren’t separated by spaces. For instance, in Chinese, the sentence "我爱北京" (“I love Beijing”) doesn’t have spaces, so the system needs to figure out where the word boundaries lie.

2. **Morphological Analyzer/POS-Tagger:**
   - **Goal:** Identify the **structure** of words and their **part of speech** (POS). For example, recognizing whether a word is a **noun**, **verb**, **adjective**, etc., and what **inflections** it has.
   - **POS tagging** disambiguates words based on their role in a sentence. For instance, “run” can be a **noun** ("a good run") or a **verb** ("I run daily").
   - **Morphological analysis** looks into the internal structure of words—breaking them down into their base form and affixes.

3. **Word Sense Disambiguation (WSD):**
   - **Goal:** Resolve the ambiguity of words with multiple meanings based on context.
   - **Example:** Consider the word "bank." Is it referring to a **financial institution** or the **side of a river**? WSD is the task that helps models determine which sense of the word is correct in a particular sentence.

4. **Syntactic/Semantic Parser:**
   - **Goal:** Extract the grammatical structure of sentences and understand the **relationship between words**.
   - **Syntactic Parsing:** Focuses on how words are **structured** in a sentence, such as identifying the **subject**, **object**, and **predicate**.
   - **Semantic Parsing:** Focuses on the **meaning** conveyed by the sentence. For example, the sentence “The cat chased the mouse” and “The mouse was chased by the cat” have different syntactic structures but the same semantic meaning.

5. **Coreference Resolution:**
   - **Goal:** Track entities in a text. When different pronouns or phrases refer to the same entity, the system needs to know this.
   - **Example:** In the sentence "John went to the store, and he bought some apples," "he" refers to "John." A coreference resolution system will make this connection.

The traditional NLP pipeline relies heavily on **rule-based systems** or **statistical models** for each of these steps. Each stage has its **own representations** (e.g., part-of-speech tags, dependency labels) and errors in one step can propagate through the rest of the pipeline.

### **2. Corpus in NLP**

#### **What is a Corpus?**

A **corpus** is simply a collection of **language data** used to train and evaluate NLP systems. The data in the corpus can be raw text or annotated with linguistic features, making it vital for **empirical NLP** (NLP based on actual data rather than intuition or theory).

**Types of Corpora:**

- **Raw Corpora:** This is unprocessed text data with no additional annotations, such as articles pulled from websites or social media posts. Raw corpora serve as a base for unsupervised tasks, such as **language modeling**.
  
- **Annotated Corpora:** Here, linguistic features have been added to the text, such as POS tags, syntactic structures, or semantic annotations. Annotated corpora are often used in **supervised learning** tasks where models are trained to predict annotations based on features learned from the training data.

#### **Corpus Information**

Key characteristics of a corpus that need to be considered include:

1. **Size**: How large is the corpus? Some tasks require **large datasets**, especially deep learning models, which thrive on massive data inputs. A corpus size might be measured in terms of the number of **words**, **sentences**, or **documents**.

2. **Genre**: What type of text is included in the corpus? Different genres introduce different challenges. A corpus of **news articles** will be different from **social media posts**, which might contain more slang, abbreviations, and informal grammar.

3. **Mode of Communication**: Is the corpus composed of **written** or **spoken** language? A spoken language corpus might include **pauses**, **disfluencies** (e.g., "um" or "uh"), and other spoken cues that written language doesn't capture.

4. **Topic**: The corpus could be focused on a particular **domain** like politics, technology, or sports. Domain-specific corpora can help train models tailored to a specific application (e.g., healthcare, legal documents).

5. **Annotation Process**: How was the corpus annotated? This is crucial for **quality control**. The quality and consistency of the annotations determine how well models will perform on tasks like **part-of-speech tagging** or **named entity recognition (NER)**.

#### **Why Do We Need Corpora?**

1. **Evaluation**: To ensure that your NLP system is performing well, you need a **test set**—a portion of the corpus that hasn’t been seen during training. This allows for a fair evaluation of the model’s performance.

2. **Improvement**: For **supervised learning**, a model learns from examples in a corpus. The more examples it sees, the better it gets at generalizing to new data. Corpora are essential for tasks such as **text classification**, **machine translation**, **speech recognition**, etc.

#### **Corpus Selection**

Choosing the right corpus is crucial for developing an effective NLP system. The task you want to accomplish and the **domain** of your system largely dictate the selection of a corpus. For instance, if you are building a spam detection system, it’s important to use a corpus containing both **spam** and **non-spam** emails.

A good rule in NLP is that the **training data** should be as similar as possible to the **test data** to ensure robust performance. If your test data is sampled from a different distribution (e.g., legal text vs. news), your model might not generalize well.

---

### **3. Text Normalization and Tokenization**

Before processing a corpus, we often need to **normalize** and **tokenize** the text. These are essential preprocessing steps that transform raw text into a form that can be analyzed by NLP models.

#### **Text Normalization**

Text normalization involves converting the text into a **standardized form**. This step is particularly important when dealing with diverse data sources such as tweets, emails, or blogs, where informal language is prevalent.

1. **Lowercasing**: Convert all characters to **lowercase** to avoid treating "Apple" and "apple" as two different tokens, unless case distinctions are important (e.g., **named entities**).

2. **Expanding Contractions**: Expand contracted forms like "isn't" to "is not" to ensure that both forms are treated uniformly.

3. **Dealing with Abbreviations and Symbols**: For example, "Mr." and "Dr." need to be properly tokenized and treated distinctly from full words like "mister" or "doctor."

4. **Removing Punctuation**: Sometimes, punctuation (e.g., commas, periods) is removed to simplify the analysis, although punctuation might be essential in tasks like **sentiment analysis**.

#### **Tokenization**

Once text is normalized, we segment it into **tokens**—typically words, but tokens can also include **punctuation**, **hashtags**, or **emoticons**.

1. **English Tokenization**: In English, tokenization is often straightforward due to spaces separating words. However, edge cases include handling **hyphenated words**, **contractions**, or **apostrophes** (e.g., "don’t" vs. "don’t").

2. **Languages Without Spaces**: In languages like **Chinese** or **Japanese**, there are no spaces between words, so tokenization becomes more challenging. In these cases, models often rely on **pre-trained tokenizers** or **language-specific heuristics**.

---

### **4. Morphology: Understanding Word Structure**

Morphology is the study of how words are formed and the internal structure of words. It looks at how **morphemes** (the smallest meaningful units in a language) come together to form words.

#### **What is a Morpheme?**

- **Free Morphemes**: These can stand alone as words. For example, "dog" or "run."
- **Bound Morphemes**: These must attach to other morphemes. For example, **prefixes** like "un-" or **suffixes** like "-ing" are bound morphemes.

#### **Two Types of Morphological Processes:**

1. **Inflectional Morphology**:
   - Inflection involves adding affixes to a word to express different **grammatical features** (e.g., tense, number, or gender).
   - **Examples**:
     - **Tense**: "walk" → "walked"
     - **Pluralization**: "cat" → "cats"
   - Importantly, **inflection** does not change the word’s **part of speech** or basic meaning. A verb remains a verb, and a noun remains a noun.

2. **Derivational Morphology**:
   - Derivation involves adding affixes that **change the meaning** of a word or even its **part of speech**.
   - **Examples**:
     - **Noun to Adjective**: "friend" → "friendly"
     - **Adjective to Noun**: "happy" → "happiness"
   - **Derivational** changes often create entirely new words with new meanings.

#### **Why Does Morphology Matter in NLP?**

Understanding the **morphological structure** of words helps in various NLP tasks:

- **Named Entity Recognition (NER)**: Recognizing entities like people’s names or organizations often relies on recognizing common **prefixes** and **suffixes**.
- **Machine Translation**: Properly translating between languages often requires **morphological analysis**, especially in highly inflected languages like **Russian** or **Turkish**.
- **Information Retrieval**: Morphology helps normalize words into their base forms, improving search results by matching all forms of a word.

#### **Word Normalization Techniques:**

1. **Lemmatization**: Reducing a word to its **base or root form** (lemma), ensuring that all inflected forms are treated as the same concept. For instance, "running" → "run," "ran" → "run."

2. **Stemming**: This is a more heuristic-based approach where **affixes** are chopped off to get the base word. However, this can sometimes result in incorrect stems. For example:
   - "generous" might be reduced to "gener," which is not a valid word.

**Stemming** is faster but less accurate than **lemmatization**, and which method to use depends on the application.

---

### **5. Handling Morphological Variants and Normalization Errors**

Sometimes, **blindly stripping affixes** can lead to errors. For example:

- "Pretend" → "Tend"
- "Mrs." → "Mr"

This shows that even word normalization strategies like stemming or lemmatization aren’t perfect, and systems need to be cautious when reducing words to their base forms.

#### **Morphology in Action**

**Example 1**:

- **Inflectional**: “The boy is playing football.” Here, “playing” is an inflected form of the verb “play” (in the present continuous tense).
  
**Example 2**:

- **Derivational**: "John is a friendly person." Here, "friendly" is a derivation of the noun "friend" that has been turned into an adjective.

---

### **Summary - Corpus & Morphology**

- **Corpus**: A collection of language data that can be either raw or annotated. Corpora are crucial for training, evaluating, and improving NLP models.
- **Morphology**: The study of the structure of words and how smaller units (morphemes) combine to create words.
- **Inflectional vs. Derivational Morphology**: Inflectional changes retain the word’s core meaning, while derivational changes can change the word’s meaning or part of speech.
- **Text Normalization and Tokenization**: Preprocessing steps that involve cleaning text and breaking it down into individual tokens, which can then be analyzed by NLP models.
  
<br>

## III - N-gram Language Models

### **1. Introduction to Language Models**

#### **What is a Language Model?**

A **language model** defines a probability distribution over a sequence of words. Its primary goal is to **predict the next word** in a sequence, given the words that come before it. Language models are essential for various tasks in NLP, including **speech recognition**, **machine translation**, **spelling correction**, **predictive text**, and more.

- **Example**: If you are typing "I love natural language…" into your phone, a language model will predict the next word, perhaps suggesting “processing” because it is a likely continuation in the context of NLP.

But how does a machine know which word is the most **plausible** in a given context? This is where **probabilities** and language models come in.

#### **Challenges Addressed by Language Models**

1. **Word Salad**: A machine needs to distinguish between valid sequences of words and random strings.
   - **Example**: Compare "lamb apple water marry" with "Mary had a little lamb." The latter is a sensible sentence, while the former is just a random jumble.

2. **Error Correction**: A machine can use a language model to correct spelling or grammar.
   - **Example**: If you mistype "fantastci," a language model can recognize that "fantastic" is far more probable and suggest the correction.

3. **Text Generation**: Language models are also used to **generate** new, human-like text. They predict and select the most likely words to form sentences.
   - **Example**: Given the prompt "I love natural language…", a language model might generate “processing” to complete the sentence.

---

### **2. Probability Theory Recap**

Before diving deeper into language models, we need a quick **refresher on probability theory**, as it is fundamental to how language models work.

#### **Sample Space and Events**

- **Sample Space**: The set of all possible outcomes in an experiment. For example, if you flip a coin, the sample space is {Heads, Tails}.
  
- **Event**: A subset of the sample space. If you toss a coin twice, an event might be getting one heads and one tails, which corresponds to the set {HT, TH}.

#### **Probability of an Event**

The **probability of an event** is the sum of the probabilities of the individual outcomes that make up that event. For instance, if you toss a fair coin twice, the probability of getting different outcomes (HT or TH) is:

$$P(\text{Different outcomes}) = P(HT) + P(TH) = \frac{1}{4} + \frac{1}{4} = \frac{1}{2}$$

#### **Joint and Conditional Probability**

- **Joint Probability**: The probability of two events happening together, denoted as $P(A \cap B)$ or $P(A, B)$.
  
- **Conditional Probability**: The probability of an event $A$, given that another event $B$ has occurred, is written as $P(A|B)$ and defined as:

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

This concept is crucial for **language models**, as they often predict the probability of the next word based on the preceding words.

#### **Chain Rule of Probability**

The **chain rule** expresses the joint probability of a sequence of events as a product of conditional probabilities:

$$P(A_1, A_2, \dots, A_n) = P(A_1)P(A_2|A_1)P(A_3|A_1, A_2) \dots P(A_n|A_1, A_2, \dots, A_{n-1})$$

In the context of language modeling, this means the probability of a sequence of words can be broken down into the probability of each word, conditioned on the preceding words.

---

### **3. Language Models in NLP**

#### **The Goal of Language Models**

In many NLP tasks, we need to know how likely a sequence of words is in a particular language, like **English**. For instance, in **speech recognition**, the model needs to decide between multiple possible transcriptions by picking the one that is most probable given the context.

- **Example**: If the system hears something like "fantastic," it needs to choose between possibilities like "fantastic," "fantasy," or "fanatic." The language model helps decide based on which word is more likely in the given context.

---

### **4. N-gram Language Models**

An **N-gram** is simply a contiguous sequence of $N$ words from a given text. For example:

- **1-gram (Unigram)**: A sequence of 1 word (e.g., "Mary," "had," "a").
- **2-gram (Bigram)**: A sequence of 2 words (e.g., "Mary had," "had a").
- **3-gram (Trigram)**: A sequence of 3 words (e.g., "Mary had a," "had a little").

#### **How N-grams Work**

**Independence Assumption:**

N-gram models assume that the probability of a word depends only on the **previous $N-1$ words**. This is also known as the **Markov assumption**—the idea that ***future states (words) depend only on the current state (recent words)***.

- **Unigram Model**: The simplest model, where we assume each word is independent of the others:

  $$P(w_i | w_1, \dots , w_{i-1}) \approx P(w_i)$$
  
  $$P(w_1, w_2, w_3, \dots, w_n) = \prod_{i=1}^{n} P(w_i) = P(w_1)P(w_2)P(w_3) \dots P(w_n)$$

- **Bigram Model**: The probability of a word depends only on the immediately preceding word:
  
  $$P(w_i | w_1, \dots , w_{i-1}) \approx P(w_i|w_{i-1})$$
  
  $$P(w_1, w_2, w_3, \dots, w_n) = \prod_{i=1}^{n} P(w_i|w_{i-1}) = P(w_1)P(w_2|w_1)P(w_3|w_2) \dots P(w_n|w_{n-1})$$

- **Trigram Model**: The probability of a word depends on the two preceding words:
  
  $$P(w_i | w_1, \dots , w_{i-1}) \approx P(w_i|w_{i-2}, w_{i-1})$$
  
  $$P(w_1, w_2, w_3, \dots, w_n) = \prod_{i=1}^{n} P(w_i|w_{i-2}, w_{i-1}) = P(w_1)P(w_2|w_1)P(w_3|w_1, w_2) \dots P(w_n|w_{n-2}, w_{n-1})$$

#### **Example: “Mary had a little lamb”**

- **Unigram Model**:
  
    $$P(\text{Mary had a little lamb}) = P(\text{Mary}) \times P(\text{had}) \times P(\text{a}) \times P(\text{little}) \times P(\text{lamb})$$

    In this model, each word is independent of the others.

- **Bigram Model**:
  
    $$P(\text{Mary had a little lamb}) = P(\text{Mary}) \times P(\text{had} | \text{Mary}) \times P(\text{a} | \text{had}) \times P(\text{little} | \text{a}) \times P(\text{lamb} | \text{little})$$

    Here, each word depends on the word that came before it.

- **Trigram Model**:
  
    $$P(\text{Mary had a little lamb}) = P(\text{Mary}) \times P(\text{had} | \text{Mary}) \times P(\text{a} | \text{Mary, had}) \times P(\text{little} | \text{had, a}) \times P(\text{lamb} | \text{a, little})$$

    Each word depends on the two preceding words, providing more context than a bigram model.

---

### **5. Computing N-gram Probabilities**

#### **Using Frequency Counts**

We estimate the probability of an N-gram based on **counts** from a training corpus. The idea is simple: the more frequently a sequence of words appears in the corpus, the higher its probability.

- **Unigram Probability**:
  
    $$P(w_i) = \frac{C(w_i)}{N}$$

    where $C(w_i)$ is the count of the word $w_i$, and $N$ is the total number of words in the corpus.

- **Bigram Probability**:
  
    $$P(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}$$

    where $C(w_{i-1}, w_i)$ is the count of the word pair $w_{i-1}, w_i$, and $C(w_{i-1})$ is the count of $w_{i-1}$ in the corpus.

#### **Handling Zero Counts (Smoothing)**

One major problem with N-gram models is that some sequences of words may not appear in the training data, leading to **zero probabilities**. To avoid assigning zero probability to unseen events, we use **smoothing techniques**:

- **Add-One (Laplace) Smoothing**

    We add 1 to every possible N-gram count, even those not seen in the corpus.

    Before Smoothing:

    ***For a unigram,***

    $$P(w_i) = \frac{C(w_i)}{N}$$

    where

    $$N = \sum_{x'} C(x')$$

    ***For a bigram,***

    $$P(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}$$

    After Smoothing:

    ***For a unigram,***

    $$P(w_i) = \frac{C(w_i) + 1}{N + |V|}$$

   ***For a bigram,***

    $$P(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i) + 1}{C(w_{i-1}) + |V|}$$

    where $|V|$ is the size of the vocabulary. This ensures no bigram has a probability of zero. We also need to make sure that:
    $$\sum_{w'} P(w'|w_{i-1}) = 1$$

- **Add-$\alpha$ Smoothing**
  
  A generalization of add-one smoothing, where we add a smaller constant $\alpha$ ($\alpha < 1$) instead of 1. This method allows more control over how much probability mass to shift from seen events to unseen ones.

    Before Smoothing: For a bigram,

    $$P(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}$$

    After Smoothing: For a bigram,

    $$P(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i) + \alpha}{C(w_{i-1}) + \alpha|V|}$$

- **Linear Interpolation Smoothing**
  
  Interpolate n-gram model with k-gram model ($k<=n$)

  Bigram:

  $$\hat{P}(w_i|w_{i-1}) = \lambda P(w_i)+(1-\lambda)P(w_i|w_{i-1})$$

  Trigram:

  $$\hat{P}(w_i|w_{i-2}, w_{i-1}) = \lambda_1 P(w_i)+\lambda_2 P(w_i|w_{i-1})+(1-\lambda_1-\lambda_2)P(w_i|w_{i-2}, w_{i-1})$$
  
---

### **6. Practical Considerations for N-gram Models**

#### **Beginning and End of Sequence**

When dealing with N-grams, we need to account for the **beginning** and **end** of a sentence. To do this, we introduce **special tokens**:

- $<s>$: Represents the start of a sentence.
- $</s>$: Represents the end of a sentence.

#### **Log Probabilities**

In practice, multiplying many small probabilities can result in very small numbers, which can lead to **numerical underflow**. To solve this, we often work in **log space**. Instead of multiplying probabilities, we sum their **logarithms**:

$$\log(P(w_1, w_2, w_3)) = \log(P(w_1)) + \log(P(w_2 | w_1)) + \log(P(w_3 | w_2))$$

This helps prevent the result from becoming too small to represent.

#### **Unknown Words (Out-of-Vocabulary Words)**

What happens if a word in the test data was not seen during training? This is a common problem called the **out-of-vocabulary (OOV)** issue. To handle this, we introduce a special token **$<\text{UNK}>$** to represent all unknown words.

#### **Zipf's Law**

How "common" are common are common words and how “rare” are rare words? A corpus will have a number of common words. We see them often enough to know (almost) anything about them. Regardless of how large our corpus is, there will be a lot of rare words and unknown words.

$$\text{word frequency} \propto \frac{1}{\text{word rank}^\alpha}$$

#### **Heap's Law**

What is the relation between corpus size and vocabulary size?

A corpus of N tokens has a vocabulary size:

$$|V| \propto N^\beta$$

where $0 < \beta < 1$

**Both Zipf's Law and Heap's Law are empirical.**

---

### **7. Evaluation of Language Models**

There are two main methods of evaluating language models:

#### **1. Extrinsic Evaluation**

This method evaluates the performance of a language model based on its impact on a **downstream task/application** like **machine translation**, **speech recognition**, or **text classification**. The model’s success is determined by how much it improves the performance of these tasks. It is also the most reliable evaluation, but equally time-consuming.

#### **Word Error Rate (WER)**

Originally developed for speech recognition, it predicts how much the predicted sequence of words differs from the actual sequence of words in the correct transcript:

$$\text{WER} = \dfrac{\text{Insertions}+\text{Deletions}+\text{Substitutions}}{\text{Actual words in transcript}}$$

For example,

Insertions: "had little lamb" $\to$ "had **a** little lamb"

Deletions: "go **to** home" $\to$ "go home"

Substitutions: "the **star** night" $\to$ "the **starry** night"

#### **2. Intrinsic Evaluation: Perplexity**

Intrinsic Evaluation is much quicker in the development cycle, but it may not guarantee extrinsic improvement.

Both intrinsic evaluation and extrinsic evaluation require an evaluation metric that allow us to compare the performance of different models. It is not always obvious how to design the evaluation metric.

Perplexity is a common metric used to evaluate language models on their own. It measures how **surprised** the model is by the test data. A lower perplexity means the model assigns higher probabilities to the test data, indicating a better fit.

Perplexity (PPL) of a sequence of words $w_1, w_2, \dots, w_n$ is defined as:

$$\text{Perplexity}(w_1, w_2, \dots, w_n) = P(w_1, w_2, \dots, w_n)^{-\frac{1}{n}}$$
$$\text{Perplexity}(w_1, w_2, \dots, w_n) = \sqrt[n]{\dfrac{1}{P(w_1, w_2, \dots, w_n)}} = \sqrt[n]{\prod_{i=1}^{n} \dfrac{1}{P(w_i|w_1, w_2, \dots, w_{i-1})}}$$

Or, equivalently, in log space, to prevent underflow:

$$\text{Perplexity}(w_1, w_2, \dots, w_n) = P(w_1, w_2, \dots, w_n)^{-\frac{1}{n}} = \sqrt[n]{\prod_{i=1}^{n} \dfrac{1}{P(w_i|w_1, w_2, \dots, w_{i-1})}}$$

$$\therefore \text{Perplexity}(w_1, w_2, \dots, w_n) = exp(-\dfrac{1}{n}\sum_{i=1}^n \log P(w_i | w_1, \dots, w_{i-1}))$$

A model with lower perplexity assigns higher probabilities to the correct sequences, meaning it better **fits** the language.

    PPL of Unigram > PPL of bigram > PPL of trigram

#### **3. Entropy**

The Entropy/Shannon Entropy describes the level of "information" or "uncertainty".

The Entropy of a discrete random variable $X$ is defined as:

$$H(X)=-\sum_{x}p(x)\log p(x)$$

Here, we use base 2 for $\log$, which provides **unit of bits**.

**Example:**

Suppose we flip a fair coin. The entropy of the outcome is:

$$H(X)=-\sum_{x}p(x)\log p(x)$$
$$=-\sum_{i=1}^2\dfrac{1}{2}\log_2 \dfrac{1}{2}$$
$$\therefore H(X)=1$$

<br>

Suppose we flip an unfair coin, where $P(heads)=0.9$ and $P(tails)=0.1$. The entropy of the outcome is:

$$H(X)=-\sum_{x}p(x)\log p(x)$$
$$=-0.9 \log_2 0.9 - 0.1 \log_2 0.1$$
$$\therefore H(X) = 0.469 < 1$$

<br>

Suppose we flip an unfair coin, where $P(heads)=1$ and $P(tails)=0$. The entropy of the outcome is:

$$H(X)=-\sum_{x}p(x)\log p(x)$$
$$=-1 \log_2 1$$
$$\therefore H(X) = 0$$

<br>

Suppose we roll a 6-sided dice. The entropy of the outcome is:

$$H(X)=-\sum_{x}p(x)\log p(x)$$
$$=-\sum_{i=1}^6\dfrac{1}{6}\log_2 \dfrac{1}{6}$$
$$\therefore H(X)=2.585$$

**Uniform  probability yields maximum uncertainty and therefore, maximum entropy.**

Entropy over a sequence $W=\{w_1, \dots ,w_n\}$ from a language $L$:

$$H(w_1, \dots ,w_n)=-\sum_{w \in L}p(w_1, \dots ,w_n) \log p(w_1, \dots ,w_n)$$

This will depend on how long the sequence is. To have a more meaningful measure, we get the average, also called **entropy rate**:

$$\dfrac{1}{n} H(w_1, \dots ,w_n) = - \dfrac{1}{n} \sum_{W \in L} p(w_1, \dots ,w_n) \log p(w_1, \dots ,w_n)$$

**Define entropy of the language $L$:**

$$H(L) = - \dfrac{1}{n} \lim_{n \to \infty} \sum_{W \in L} p(w_1, \dots ,w_n) \log p(w_1, \dots ,w_n)$$

This can be simplified by the Shannon-McMillan-Breiman Theorem to:

$$H(L) = - \dfrac{1}{n} \lim_{n \to \infty} \log p(w_1, \dots ,w_n)$$

But, we don't know the true probability distribution of $p$.

#### **4. Cross Entropy**

In practice, we don’t know the true probability distribution $p$ for language $L$, only an estimated distribution from a language model.

**Define cross-entropy as:**

$$H(p, \hat p) = - \sum_{x} p(x) \log \hat p(x)$$

According to Gibb’s inequality, entropy is less than or equal to its cross-entropy, i.e.,

$$- \sum_{x} p(x) \log p(x) ≤ -\sum_{x} p(x) \log \hat p(x)$$

*This means that if we have two language models, the more accurate model will have a lower cross-entropy.*

Following the Shannon-McMillan-Breiman Theorem,

$$H(p, \hat p) = - \dfrac{1}{n} \lim_{n \to \infty} \log \hat p(w_1, \dots, w_n)$$

For a language model, lower $H(p, \hat p)$ is better.

We recall that: $Perplexity = 2^{cross-entropy}$.

$$\implies \text{Perplexity}(w_1, w_2, \dots, w_n) = P(w_1, \dots, w_n)^{-\frac{1}{n}}$$

$$\implies 2^{cross-entropy} = 2^{- \frac{1}{n} \log_2 P(w_1, \dots, w_n)} = P(w_1, \dots, w_n)^{-\frac{1}{n}}$$

---

### **Summary - N-Grams**

- **N-gram language models** are a foundational concept in NLP, where we use probability estimates to predict the likelihood of a sequence of words.
- **Unigram, Bigram, and Trigram models** represent different ways of approximating these probabilities based on preceding words.
- **Smoothing techniques** like **Add-One** ensure that our models don’t fail due to unseen word sequences.
- **Perplexity** is the key intrinsic evaluation metric that measures how well the model predicts sequences in a language.

<br>

## IV - Text Classification - Naive Bayes

### What is Text Classification?

**Text classification** is the task of assigning predefined labels to a given text. It is one of the most fundamental tasks in **Natural Language Processing (NLP)**, used in applications like spam detection, sentiment analysis, topic categorization, and more.

### Types of Text Classification

1. **Binary Classification**: Involves two possible classes. For example, sentiment analysis classifies a review as either positive or negative.
2. **Multiclass Classification**: Involves more than two possible classes. For example, categorizing news articles into genres such as politics, sports, or entertainment.

In either case, the goal is to assign a class label to the input text. The classification model, or **classifier**, maps an input text $x \in X$ (e.g., a movie review) to a pre-defined set of class labels $y \in Y$ (e.g., positive/negative/neutral or category).

### Example of Real-world Applications

- **Sentiment Analysis**: Determining whether a product review is positive, negative, or neutral.
- **Spam Detection**: Classifying an email as either spam or not spam.
- **Topic Categorization**: Grouping news articles into topics such as sports, technology, politics, etc.

---

### Naive Bayes Classifier for Text Classification

One of the simplest and most effective models for text classification is the **Naive Bayes classifier**. It is called "naive" because it assumes that all features (in this case, words) are conditionally independent given the class of the document. This assumption makes the model easy to compute, though in practice, words in a sentence are often dependent on each other.

### Key Idea

The Naive Bayes classifier works by calculating the probability that a document belongs to each class and then selecting the class with the highest probability.

### The Naive Bayes Formula

Given a document $x = (w_1, w_2, \dots, w_n)$, consisting of words $w_1, w_2, \dots, w_n$, Naive Bayes calculates the **probability** that the document belongs to class $y$. The formula for classifying $x$ is:

$$y^* = \arg\max_{y} P(Y = y) \prod_{i=1}^{n} P(w_i | y)$$

Where:

- $y^*$ is the predicted class (e.g., positive or negative).
- $P(Y = y)$ is the **prior probability** of class $y$, which is the likelihood of encountering this class before seeing the document.
- $P(w_i | y)$ is the **class-conditional probability** of word $w_i$ appearing in class $y$, which represents how likely the word "fantastic" is in a positive review versus a negative one.
- The product $\prod_{i=1}^{n} P(w_i | y)$ multiplies the individual word probabilities to calculate the overall probability of the document given the class.

---

### Breaking Down the Formula

### 1. **Prior Probability $P(Y = y)$**

This is the probability of encountering class $y$ without any context from the document. For example, if 60% of the reviews in the training data are positive, the prior probability of a positive review is 0.6. We calculate this from the training data as:

$$P(Y = y) = \frac {\text{\# of documents with class y}}{\text{total \# of documents}}$$

### 2. **Class-Conditional Probability $P(w_i | y)$**

This is the probability of seeing a particular word $w_i$ given that the document belongs to class $y$. For instance, $P(\text{"great"} | +)$ is the probability of the word "great" occurring in a positive review. We estimate this by counting how often the word appears in documents of class $y$, divided by the total number of words in that class:

$$P(w_i | y) = \frac{\text{count}(w_i, y)}{\text{total words in class } y}$$

### 3. **Multiplying Word Probabilities**

In Naive Bayes, we assume that the occurrence of each word is independent of the others, given the class. This allows us to compute the probability of the document by multiplying the probabilities of each word:

$$P(w_1, w_2, \dots, w_n | y) = P(w_1 | y) \times P(w_2 | y) \times \dots \times P(w_n | y)$$

Finally, the class $y^*$ that maximizes this probability is selected as the predicted label for the document.

---

### Example: Sentiment Analysis

Let’s go through a detailed example of how Naive Bayes is applied to **sentiment analysis**.

#### Training Data

Imagine you have the following training data with positive (+) and negative (-) movie reviews:

- Positive reviews: "The movie was fantastic and great", "I love this film"
- Negative reviews: "The film was terrible", "I hate it"

From this data, you create two big texts:

- **Big positive text**: Contains all positive reviews ("fantastic," "great," "love," "film").
- **Big negative text**: Contains all negative reviews ("terrible," "hate," "film").

Now, we can compute the probabilities needed for Naive Bayes.

#### Step 1: Prior Probability

The prior probability represents how often positive and negative reviews occur:

$$P(+) = \frac{2}{4} = 0.5, \quad P(-) = \frac{2}{4} = 0.5$$

#### Step 2: Class-Conditional Probability

Now, let’s compute the probability of each word appearing in positive and negative reviews. For example, for the word "fantastic":

$$P(\text{"fantastic"} | +) = \frac{1}{\text{total words in positive reviews}}, \quad P(\text{"fantastic"} | -) = \frac{0}{\text{total words in negative reviews}}$$

We do this for each word in both classes.

#### Step 3: Classifying a New Review

Suppose the new review is: **"fantastic film"**. To classify it, we calculate the probabilities for both the positive and negative classes by multiplying the prior probabilities by the likelihoods of the words "fantastic" and "film":

For the **positive class**:

$$P(+) \times P(\text{"fantastic"} | +) \times P(\text{"film"} | +)$$

For the **negative class**:

$$P(-) \times P(\text{"fantastic"} | -) \times P(\text{"film"} | -)$$

The class with the ***higher*** probability is chosen as the predicted label.

---

### Dealing with Unseen Words and Zero Probabilities

### The Problem

What if a word appears in the test data but wasn’t seen in the training data? For instance, if the word "predictable" wasn’t in the training data, its probability would be zero, and multiplying by zero would make the entire product zero, which would make the classifier useless for that document.

### Solution: Laplace Smoothing

To solve this, we use **Laplace Smoothing**, also called **Add-1 Smoothing**. This technique adds a small value (usually 1) to the count of every word, ensuring that no probability is zero.

The formula becomes:

$$P(w_i | y) = \frac{\text{count}(w_i, y) + 1}{\text{total words in class } y + |V|}$$

Where $V$ is the size of the vocabulary (i.e., the total number of unique words in the training data).

This way, even if a word didn’t appear in the training data for a given class, it will still have a small, non-zero probability.

---

### Practical Considerations

### 1. **Bag-of-Words Representation**

In Naive Bayes, we typically represent text as a **bag-of-words**. This means we ignore the order of the words and only focus on their presence or frequency. For example, the sentence "I love this movie" is treated the same as "movie love I this."

### 2. **Stop Words**

Certain very frequent words, like "the," "is," and "and," are called **stop words**. These words don’t carry much meaning and are often removed from the text before classification. Removing stop words can improve the performance of a Naive Bayes classifier by reducing noise.

### 3. **Unknown Words**

If a word in the test data wasn’t seen in the training data, it is treated as an **unknown word**. One common approach is to create a special $<\text{UNK}>$ token that represents all unknown words. Alternatively, we can simply ignore these words.

---

### Why Use Naive Bayes?

- Naive Bayes is simple but effective for many text classification tasks.
- Despite the "naive" assumption of independence between words, it often performs surprisingly well. This is because it captures enough useful patterns from the data, even if the assumption is not completely accurate.

### Strengths of Naive Bayes

- **Efficiency**: It’s computationally efficient and can handle large datasets.
- **Simple to Implement**: Easy to understand and implement.
- **Baseline Model**: Naive Bayes often serves as a good baseline model in text classification tasks.

### Applications

- **Spam Detection**: Classifying emails as spam or not spam.
- **Sentiment Analysis**: Analyzing customer reviews and tweets for sentiment.
- **Topic Categorization**: Automatically labeling documents by their topics (e.g., classifying news articles into sports, politics, entertainment, etc.).

---

### Summary - Naive Bayes Classification

1. **Text classification** is about assigning a label (e.g., positive/negative) to a given text.
2. The **Naive Bayes classifier** calculates the probability of each class based on the words in the document and assigns the class with the highest probability.
3. **Laplace smoothing** prevents zero probabilities by adding 1 to every word count.
4. **Bag-of-words representation** and the treatment of **stop words** and **unknown words** are important considerations for implementing Naive Bayes.

Naive Bayes is a simple, yet powerful tool for a variety of NLP tasks. It provides a strong foundation for more advanced models and is often used as a baseline to compare more complex methods.

<br>

## V - Logistic Regression

### Introduction to Logistic Regression

**Logistic Regression** is a type of machine learning algorithm used for binary classification problems, where the goal is to assign one of two possible labels (e.g., positive/negative). Unlike linear regression, which predicts a continuous value, logistic regression predicts the **probability** that a given input belongs to a particular class.

#### Why Logistic Regression?

In text classification (and other classification tasks), we often deal with binary decisions. Logistic regression is well-suited for this task because it models the relationship between the input features and the probability of the binary outcomes. It is a **discriminative model**, meaning it directly estimates the probability of the class labels given the features.

---

#### Decision Boundary

The **decision boundary** in logistic regression is determined by the sigmoid function. It is the threshold that separates the two classes. If the predicted probability is greater than 0.5, the model classifies the input as belonging to the positive class (e.g., positive sentiment). Otherwise, it is classified as the negative class.

The decision boundary is linear in the input space, which means logistic regression fits a straight line (or a hyperplane in higher dimensions) to separate the two classes.

---

### Training a Logistic Regression Model

#### Likelihood and Maximum Likelihood Estimation (MLE)

Logistic regression is trained using **Maximum Likelihood Estimation (MLE)**, which involves finding the weights $w_1, w_2, \dots, w_n$ that maximize the likelihood of the observed data.

The **likelihood function** measures how likely it is that the model's parameters would produce the observed data. In logistic regression, the likelihood is defined based on the probabilities assigned to the true labels of the training data.

Let’s say we have a set of training data $\{(x_1, y_1), (x_2, y_2), \dots, (x_n, y_n)\}$, where each $x_i$ is a feature vector and $y_i$ is the corresponding label (0 or 1). The **log-likelihood** for logistic regression is given by:

$$L(w) = \sum_{i=1}^{n} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$$

Where $p_i$ is the predicted probability for instance $x_i$ given by the sigmoid function.

---

### Cross-Validation and Evaluating Classifiers

#### Cross-Validation

**Cross-validation** is a technique used to evaluate the performance of a machine learning model. The goal is to ensure that the model generalizes well to unseen data. One common method is **k-fold cross-validation**.

**K-Fold Cross-Validation**: The data is split into $k$ equal-sized **folds**. The model is trained $k$ times, each time using a different fold as the test set and the remaining $k-1$ folds as the training set. The model’s performance is averaged over the $k$ trials to get a more reliable estimate of its generalization performance.

- **Why use cross-validation?**: It ensures that all data points are used for both training and testing, which provides a more robust estimate of the model’s performance, especially on small datasets.

#### Training, Development, and Test Sets

In machine learning experiments, the data is usually divided into three sets:

1. **Training Set**: Used to train the model.
2. **Development Set** (or validation set): Used to tune hyperparameters and evaluate the model during development.
3. **Test Set**: Used for final evaluation. The test set should remain **unseen** during model development to provide an unbiased evaluation.

### Evaluation Metrics

Several metrics are used to evaluate the performance of classifiers, particularly binary classifiers like logistic regression.

#### 1. **Accuracy**

Accuracy is the most basic evaluation metric and is defined as the percentage of instances correctly classified by the model:

$$\text{Accuracy} = \frac{\text{\# of correct predictions}}{\text{total \# of predictions}}$$

While accuracy is easy to understand, it can be misleading in situations where the data is imbalanced. For example, if 90% of the data belongs to class A and only 10% to class B, a classifier that always predicts A will have 90% accuracy, but it’s not useful for identifying class B.

#### 2. **Precision and Recall**

To better evaluate the performance in imbalanced datasets, we use **precision** and **recall**:

- **Precision**: Measures the proportion of positive predictions that are actually correct. It answers the question: "Of all the examples the model labeled as positive, how many are actually positive?"

$$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$$

- **Recall** (also known as sensitivity): Measures the proportion of actual positives that were correctly identified. It answers the question: "Of all the positive examples in the dataset, how many did the model correctly identify?"

$$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$

Precision and Recall are often combined into a single metric called the **F1 score**, which is the harmonic mean of precision and recall:

$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

#### 3. **Confusion Matrix**

A confusion matrix provides a detailed breakdown of the classifier’s performance by showing the counts of **true positives (TP)**, **true negatives (TN)**, **false positives (FP)**, and **false negatives (FN)**.

For binary classification, the matrix looks like this:

|                 | Predicted Positive | Predicted Negative |
|-----------------|--------------------|--------------------|
| **Actual Positive** | True Positive (TP)    | False Negative (FN)   |
| **Actual Negative** | False Positive (FP)   | True Negative (TN)    |

- **True Positives (TP)**: Correctly predicted positive instances.
- **True Negatives (TN)**: Correctly predicted negative instances.
- **False Positives (FP)**: Incorrectly predicted positive instances (also known as Type I error).
- **False Negatives (FN)**: Incorrectly predicted negative instances (also known as Type II error).

From the confusion matrix, we can calculate accuracy, precision, recall, and other metrics.

#### 4. **Macro-averaging vs. Micro-averaging**

When working with multiclass classification problems, precision and recall can be averaged across classes in two ways:

- **Macro-averaging**: Compute precision and recall for each class separately, then take the average. This treats all classes equally, regardless of their size. *In short, this is the average over $n$ classes.*
  - **Example:** Given a confusion matrix that presents how many items that have labels "Comedy", "Action", and "Romance":
    $$\text{Macro-averaged Recall} = \dfrac{1}{3} \times (\text{Recall}_{action} + \text{Recall}_{comedy} + \text{Recall}_{romance})$$
    $$\text{Macro-averaged Precision} = \dfrac{1}{3} \times (\text{Precision}_{action} + \text{Precision}_{comedy} + \text{Precision}_{romance})$$
  
- **Micro-averaging**: Compute global counts of TP, FP, and FN across all classes, then compute precision and recall from these totals. This gives more weight to larger classes. *In short, this is the average over all examples.*
  - **Example:** Given a confusion matrix that presents how many items that have labels "Comedy", "Action", and "Romance":
    $$\text{Micro-averaged Recall} = \dfrac{TP_{action} + TP_{comedy} + TP_{romance}}{TP_{action} + TP_{comedy} + TP_{romance} + FN_{action} + FN_{comedy} + FN_{romance}}$$
    $$\text{Micro-averaged Precision} = \dfrac{TP_{action} + TP_{comedy} + TP_{romance}}{TP_{action} + TP_{comedy} + TP_{romance} + FP_{action} + FP_{comedy} + FP_{romance}}$$

- Macro-averaging is useful if all classes are equally important, especially for an imbalanced dataset.

- Micro-averaging is useful when you want to account for the total number of misclassifications in the dataset. For multi-class classification, $\text{micro-averaged precision} = \text{micro averaged recall} = \text{accuracy}$

---

### Word Error Rate (WER)

Originally developed for speech recognition, **Word Error Rate (WER)** is used to evaluate the performance of systems that generate sequences of words (e.g., machine translation, speech-to-text). WER measures how much the predicted sequence differs from the actual sequence by counting **insertions**, **deletions**, and **substitutions**.

WER is calculated as:

$$\text{WER} = \frac{\text{Insertions} + \text{Deletions} + \text{Substitutions}}{\text{Number of words in the actual sequence}}$$

**Lower WER values indicate better performance.**

---

### Probabilistic Classifiers

- A probabilistic classifier return the most likely class $y$ for input $x$,

$$y^* = \argmax_y P(Y=y|X=x)$$

- Naive Bayes uses Bayes Rule:

$$y^* = \argmax_y P(y|x) = \argmax_y P(x|y)P(y)$$

- Naive Bayes models the **joint distribution** of the class and the data:
  
$$P(x|y)P(y) = P(x,y)$$

- Joint models are also called **generative models**.

- **Discriminative** (also called **conditional**) models try to model $P(y|x)$ directly.

- For a classification task, given data point $x = (x_1, x_2, \dots, x_n)$, we want a model to predict label $y$:
  - Generative Model (Naive Bayes): $P(x|y)$
  - Discriminative Model (Logistic Regression): $P(y|x)$

- Generative models, like Naive Bayes, model the joint probability distribution $P(X,Y)$ over the input features $X$ and the output labels $Y$.
  - They can generate new data points by sampling from this joint distribution. Once the joint probability is learned, we can use Bayes' Theorem to calculate the conditional probability $P(X|Y)$, which gives us the class probabilities.
  - Generative models aim to understand how the data was generated by modeling both the input features and the output labels.
  - This allows them to handle missing data better in some cases and can model more complex relationships in the data.

- Discriminative models, like logistic regression, model the conditional probability $P(Y∣X)$ directly.
  - Instead of trying to model the data-generating process, they focus on learning the boundary between different classes.
  - They use the training data to directly estimate the probability of a label given the input features without needing to model how the data was generated.
  - Discriminative models often outperform generative models when we have enough training data because they don't have to make as many assumptions about the structure of the data.

---

#### The Sigmoid Function

At the heart of logistic regression is the **sigmoid function**, also known as the **logistic function**. The sigmoid function maps any real-valued number to a value between 0 and 1, which can be interpreted as a probability.

The sigmoid function is given by:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Where:

- $z$ is a linear combination of the input features.
- $e$ is the base of the natural logarithm.

For logistic regression, $z$ is typically represented as:

$$ z= w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b$$

Where:

- $x_1, x_2, \dots, x_n$ are the input features (e.g., word frequencies in a document).
- $w_1, w_2, \dots, w_n$ are the weights associated with each feature.
- $b$ is the bias term (similar to the intercept in linear regression).

The output of the sigmoid function is a probability between 0 and 1. If the probability is greater than 0.5, we assign the positive class; otherwise, we assign the negative class.

For **binary** classfication, logistic regression uses the **sigmoid function**:

$$P(y=1|x) = \sigma (\theta_0 + \sum_{i=1}^n \theta_i \cdot x_i)$$

Parameters to learn: $\theta$

---

### Multinomial Logistic Regression

While logistic regression is primarily used for binary classification, there is an extension of logistic regression called **multinomial logistic regression** for multiclass classification problems (i.e., when there are more than two classes, such as in POS tagging or semantic categories).

The goal is to model the probability of each class, given the input. Instead of using the sigmoid function, which is suitable for binary classification, we use the **softmax** function to ensure that the predicted probabilities for all classes sum to 1.

---

### Softmax Function

The softmax function generalizes the sigmoid function to handle multi-class classification problems. The output of the softmax function is a probability distribution over all possible classes.

Given a set of input features $X$, the softmax function assigns a probability to each class.

The softmax function turns any vector of reals $z = (z_1, \dots, z_K)$ into a discrete probability distribution, $p = (p_1, \dots, p_K)$, where $0≤p_i≤1$ and $\sum_{i=1}^K p_i = 1$

$$p_i = \text{softmax}(z)_i = \dfrac{e^{z_i}}{\sum_{k=1}^K e^{z_k}}$$

**How It Works:**

Each class $k$ is assigned a score $z_k$, which is computed as a linear combination of the input features.

The softmax function converts these scores into probabilities, ensuring that the probabilities sum to 1. The class with the highest score will have the highest probability.

Logistic regression applies the softmax to a linear combination of the input features.

In multinomial logistic regression, we use the softmax function to compute the probabilities for all classes, and then the class with the highest probability is chosen as the predicted label.

Example:
Suppose we are classifying news articles into three categories: sports, politics, and entertainment. Given a feature vector $X$, the softmax function will compute the probabilities of each class. The class with the highest probability is chosen as the predicted category for the article.

---

### Logistic Regression vs. Naive Bayes

It’s helpful to compare **Logistic Regression** to **Naive Bayes**, especially since both are common for text classification tasks:

- **Naive Bayes** is a generative model, meaning it learns the joint probability distribution $P(X, Y)$ and uses Bayes' theorem to calculate the posterior probabilities $P(Y | X)$. Naive Bayes makes strong independence assumptions between features.
  
- **Logistic Regression** is a discriminative model, which directly models the probability $P(Y | X)$ without making assumptions about the relationships between features. It tends to perform better when the features are not independent.

### When to Use Logistic Regression

- Logistic regression is often preferred when you have a lot of features that interact with each other (since it doesn’t assume independence).
- It is also more robust to noise and overfitting compared to Naive Bayes.

---

### Gradient Descent

To maximize the likelihood and find the optimal weights, logistic regression typically uses **gradient descent**.

The gradient of the loss function with respect to the parameters:

$$\dfrac{\partial L_{train}(\theta)}{\partial \theta}$$

It indicates the direction of the steepest increase in $L_{train}(\theta)$.

Move in the opposite direction to decrease $L_{train}(\theta)$:

$$\theta \leftarrow \theta - \eta \dfrac {\partial L_{train}(\theta)}{\partial \theta}$$

$\eta$ is called the **learning rate**.

Gradient descent is an optimization algorithm that iteratively updates the weights in the direction of the gradient (the direction that increases the likelihood).

---

### Summary - Logistic Regression

1. **Logistic Regression** is a popular machine learning model for binary classification. It uses the **sigmoid function** to predict probabilities and is trained using **maximum likelihood estimation**.
2. Logistic regression is optimized using **gradient descent** to find the weights that maximize the likelihood of the data.
3. **Cross-validation** ensures that the model generalizes well to unseen data by evaluating it on multiple folds of the data.
4. Classifiers are evaluated using metrics like **accuracy**, **precision**, **recall**, **F1 score**, and **confusion matrices**. These metrics provide a more nuanced view of a model’s performance, especially in imbalanced datasets.
5. **Word Error Rate (WER)** is used for sequence prediction tasks, such as speech recognition, to measure how closely the predicted output matches the true sequence.

<br>

## VI - Part-of-Speech Tagging

POS Tagging is the process of assigning parts of speech, such as nouns, verbs, adjectives, and adverbs, to each word in a sentence. Each word is assigned a label that identifies its syntactic category.

**Key POS Tags:**

- Noun: Common nouns (e.g., book, water) and proper nouns (e.g., Cincinnati, Mary).
- Verb: Actions (e.g., eat, run, sing).
- Adjective: Descriptive words (e.g., great, small).
- Adverb: Modifies verbs (e.g., quickly, slowly).

<br>

**Classes of Words:**

- Nouns: Count nouns have plural forms (e.g., books), while mass nouns do not (e.g., water).
- Verbs: Can have different tenses (e.g., eat, ate, eaten).
- Adjectives: Can be used attributively (a great book) or predicatively (the book is boring).
- Pronouns: Words like I, you, she.
- Adverbs: Modify verbs, adjectives, or other adverbs (e.g., very, slowly).
- Prepositions: Express relations between words (e.g., in, on).
- Conjunctions: Connect clauses or sentences (e.g., and, but).

<br>

**Open vs. Closed Classes:**

- Open classes (e.g., nouns, verbs) continuously accept new words.
- Closed classes (e.g., prepositions, pronouns) rarely accept new words.

<br>

**Fine-Grained Classes:**

- For example, nouns can be singular (NN), plural (NNS), proper singular (NNP), or proper plural (NNPS).
- Verbs can have forms like base (VB), past tense (VBD), gerund (VBG), and past participle (VBN).

---

<br>

### Creating a POS Tagger

A Part-of-Speech (POS) Tagger is used to assign part-of-speech tags (e.g., noun, verb, adjective) to words in a sentence.

The task of building a POS tagger can vary depending on whether the language is well-studied or new.

**For a New Language:**

- Step 0: Define a POS Tagset
  - We need to decide which parts of speech will be tagged in the language (e.g., noun, verb, adjective).
- Step 1: Annotate a Corpus
  - A human annotator tags a large body of text with the defined POS tags to create a tagged corpus. This tagged corpus is then used to train the POS tagger.

<br>

**For Well-Studied Languages:**

- We can obtain a POS-tagged corpus directly from previously annotated datasets (e.g., the Penn Treebank for English).
- We can use this corpus to build the POS tagging model through supervised learning.

<br>

Example of a Tagset: Penn Treebank Tagset: A common tagset used for POS tagging in English which includes tags like NN for singular noun, VB for base verb, JJ for adjective, etc.

**Inter-Annotator Agreement (IAA)** and the **Kappa Statistic** are metrics used in the evaluation of human annotations in tasks like text annotation, labeling, and other NLP tasks where multiple annotators are involved. These metrics assess how much agreement there is between different annotators and help determine the consistency and reliability of the annotations.

---

<br>

### Inter-Annotator Agreement (IAA)

**Inter-Annotator Agreement (IAA)** refers to the degree to which different human annotators (labelers) agree on their annotations or labels for the same set of data. It measures the consistency of different people when annotating data.

It's crucial in NLP tasks because human annotations serve as the "ground truth" for training and evaluating machine learning models. If annotators frequently disagree, it suggests that the task might be ambiguous or that annotation guidelines need improvement.

<br>

**Why is IAA Important?**

- Ensures that annotations are consistent and reliable.
- High IAA is needed to trust the labeled data as a basis for training models.
  - High IAA is important to ensure that the task is well-defined and that the annotations have high integrity.
  - To achieve high IAA, we must:
    - precisely define the annotation task, which usually requires
    detailed annotation guidelines that include examples and discuss
    how to handle boundary cases.
    - train the annotators, iteratively, while refining the guidelines.
    - measure IAA using an appropriate statistical measure.
- Low IAA may indicate unclear annotation guidelines or subjective labels, leading to poor model performance.

---

<br>

### The Kappa Statistic

The Kappa statistic (Cohen's Kappa), $\kappa$, is a statistical measure used to quantify Inter-Annotator Agreement while accounting for the possibility of agreement occurring by chance.

Unlike simple agreement percentages, which only measure how often annotators agree, the Kappa statistic corrects for the fact that some agreement is expected to occur randomly.

**Cohen’s Kappa Formula -**
The formula for Cohen’s Kappa is given by:

$$\kappa = \dfrac{P(\text{agree}) - P(\text{expected})}{1 - P(\text{expected})}$$

where,

- $P(agree)$ = proportion of times the annotators agree
- $P(expected)$ = proportion of times the annotators are expected to agree by chance

<br>

#### Kappa for two annotators

$$P(expected) = \sum_{c \in C} P(c|A_1) \times P(c|A_2)$$

where

- $C$ is the set of all possible classes (e.g., POS tagset)
- $A_1$ represents annotator 1
- $A_2$ represents annotator 2

<br>

### Example of Kappa Statistic Calculation

Let’s say we have two annotators labeling a dataset with three categories: A, B, and C. Their annotations for 100 items are summarized in the following table:

|              | Annotator 1 A | Annotator 1 B | Annotator 1 C |
|--------------|----------------|---------------|---------------|
| Annotator 2 A | 40             | 10            | 5             |
| Annotator 2 B | 10             | 20            | 10            |
| Annotator 2 C | 5              | 10            | 10            |

#### Step 1: Calculate Observed Agreement $P(agree)$

The total number of agreements (when both annotators chose the same category) is:

- 40 for category A (A-A),
- 20 for category B (B-B),
- 10 for category C (C-C).

Thus, the observed agreement is:

$$P(agree) = \frac{\text{\# of agreements}}{\text{total \# of items}} = \frac{40 + 20 + 10}{100} = 0.70$$

#### Step 2: Calculate Expected Agreement $P(expected)$

Suppose both annotators assign the following proportions to each category:

- **A**: 50%,
- **B**: 30%,
- **C**: 20%.

The expected agreement is calculated by multiplying the probabilities that both annotators would assign the same category purely by chance:

$$P(expected) = (0.5 \times 0.5) + (0.3 \times 0.3) + (0.2 \times 0.2) = 0.25 + 0.09 + 0.04 = 0.38$$

#### Step 3: Calculate Kappa Statistic

Now, using the formula for Cohen’s Kappa:

$$\kappa = \dfrac{P(\text{agree}) - P(\text{expected})}{1 - P(\text{expected})} = \frac{0.70 - 0.38}{1 - 0.38} = \frac{0.32}{0.62} = 0.52$$

Thus, the Kappa statistic is $\kappa = 0.52$, indicating **moderate agreement** between the two annotators.

---

<br>

### Rule-Based POS Tagging

In **Rule-based POS Tagging**, a set of predefined rules is used to assign tags to words in a sentence. These rules may consider the word itself or its context within the sentence.

Rule-based taggers rely on a dictionary and morphological analysis to provide possible POS tags for a word, and/or rules can be generated from training data.

<br>

**How it Works:**

- Dictionary and Morphological Analysis: A dictionary lookup is performed to suggest possible POS tags for each word.
- Disambiguation Rules: Manually developed rules are used to disambiguate between multiple possible POS tags. For example:
  - Rule Example 1: If the preceding word is an article (ART), disambiguate between a noun and a verb by choosing a noun.
  - Rule Example 2: If the preceding word is an adverb (ADV), the word is tagged as a verb.

**Limitations:**

- Manually developed rules can perform well, but they are labor-intensive and not as flexible or adaptable as statistical models.
- Statistical POS taggers, discussed later, generally outperform rule-based systems when annotated data is available.

---

<br>

### Statistical POS Tagging

Statistical POS Tagging identifies the most likely sequence of POS tags for the words in a sentence by applying probabilistic models. Rather than relying on manually defined rules, statistical taggers learn patterns from annotated data.

What is the most likely POS tag sequence $t = (t_1, \dots, t_n)$ for the given sentence $w = (w_1, \dots, w_n)$?

$$t^* = \argmax_t P(t|w)$$

**What Helps in Deciding POS Tags?**

- The Word Itself: Some words are highly correlated with specific POS tags. For example, "soccer" is almost always a noun.
- Tags of Surrounding Words: The context in which the word appears is useful for disambiguation. For example, two determiners rarely appear consecutively in English.

---

<br>

### POS Tagging with Generative Models

Generative models aim to model both the tags and the observed words. The goal is to compute the joint probability of the word sequence and tag sequence $P(w,t)$, and then find the tag sequence that maximizes this probability.

$$t^* = \argmax_t P(t|w)$$

This can be decomposed to give:

$$t^* = \argmax_t P(t)P(w|t)$$

The joint probability model is based on the assumption that:

- Each Tag Depends Only on the Previous Tag: This is the Markov assumption, often leading to bigram or trigram models for tags.
- Words Are Independent Given Tags: This simplifies the model so that $P(w∣t)$ is calculated based on individual words and tags.

---

<br>

### Hidden Markov Model (HMM)

The **Hidden Markov Model (HMM)** is a widely used probabilistic model for POS tagging.

HMMs are particularly useful for sequence labeling tasks where we need to infer hidden states (e.g., POS tags) from observed data (e.g., words in a sentence). In other words, a **Hidden Markov Model (HMM)** is used to find the most likely sequence of hidden events given a sequence of observable events.

The input tokens are the observed events.

The class labels (states) are the hidden events.

It is called hidden because at test time, we only see the words (emissions). The tags (states) are hidden variables.

<br>

#### Elements of an HMM

- a set of states (here, tags) - The hidden variables, which in POS tagging correspond to the part-of-speech tags.
- an output alphabet (here, words) - The observed variables, which in this case are the words in the sentence.
- initial state (here, beginning of sentence $\phi$)
- state transition probabilities (here, $P(t_i|t_{i-1})$) - These represent the likelihood of transitioning from one tag (state) to another.
- symbol emission probabilities (here, $P(w_i|t_i)$) - These represent the probability of emitting a word given a particular POS tag.

<br>

#### How HMMs Work

- Observed Events: The words in the sentence are the observed events.
- Hidden Events: The POS tags are hidden, meaning we don’t directly observe them. The task is to infer the most probable sequence of hidden states (tags) given the observed sequence (words).

<br>

#### HMM vs. Previous Models

- **N-gram Model:** Models sequences but doesn’t handle hidden variables.
- **Naive Bayes**: Has hidden variables but doesn’t model sequential dependencies.
- **HMM:** Combines both hidden variables and sequential dependencies, making it ideal for sequence tasks like POS tagging.

#### Example

Example:
Suppose we want to calculate the probability of a tagged sentence like "The/DET soccer/NOUN team/NOUN won/VERB":

$$P(w,t)=P(DET∣ϕ) \cdot P("the"∣DET) × P(NOUN∣DET) \cdot P("soccer"∣NOUN) × P(NOUN∣NOUN) \cdot P("team"∣NOUN) × P(VERB∣NOUN) \cdot P("won"∣VERB)$$

where the transition probabilities between tags and emission probabilities for words are estimated from training data.

---

<br>

## VII - Viterbi

### Formalizing the Tagging Problem

Given an untagged sentence $w$, the task is to find the **best tag sequence** $t$ for the sentence. This can be formulated as:

$$t^* = \arg \max_t P(t | w)$$

where:

- $P(t)$ is the **state transition probability** (i.e., the probability of moving from one tag to another).
- $P(w|t)$ is the **emission probability** (i.e., the probability of emitting a word given a tag).

<br>

In the context of an HMM, the problem can be further simplified to:

$$t^* = \arg \max_t \prod_{i=1}^{n} P(t_i | t_{i-1}) P(w_i | t_i)$$

This represents the joint probability of a tag sequence and the observed word sequence.

---
<br>

### Hidden Markov Model (HMM) Overview

An HMM is used to model the relationship between a **sequence of observed events** (like words in a sentence) and a corresponding **sequence of hidden states** (like POS tags). The HMM assumes that:

1. The sequence of hidden states follows a **Markov process**, meaning each state depends only on the previous one.
2. Each hidden state emits an observation (word) with a certain probability.

<br>

**Components of HMM**:

- **States**: Correspond to tags (e.g., noun, verb).
- **Observations**: Correspond to words in a sentence.
- **State Transition Probabilities** $P(t_i | t_{i-1})$: Likelihood of transitioning from one state to the next.
- **Emission Probabilities** $P(w_i | t_i)$: Likelihood of emitting a word given a tag.

---

<br>

### The Viterbi Algorithm

The **Viterbi Algorithm** is an efficient way to solve the decoding problem of finding the most probable sequence of tags given a sentence. It avoids the inefficiency of enumerating all possible tag sequences by using **dynamic programming**.

#### **How the Viterbi Algorithm Works**

1. **Initialization**:
   - For each possible tag $t_1$, compute:

    $$v(t_1, 1) = P(t_1 | \phi) P(w_1 | t_1)$$

   - Here, $\phi$ represents the start of the sequence.

2. **Recursion**:
   - For each word $w_i$ at position $i$, and for each possible tag $t_i$, compute:

    $$v(t_i, i) = \max_{t_{i-1}} \left[ v(t_{i-1}, i-1) \times P(t_i | t_{i-1}) \times P(w_i | t_i) \right]$$

   - The algorithm keeps track of the **backpointers** to remember which tag led to the maximum score for each word.

3. **Termination**:
   - After processing all words, the final score is:

    $$t^* = \arg \max_{t_n} v(t_n, n)$$

   - Backtrack through the stored backpointers to retrieve the sequence of tags.

#### **Time Complexity**

The time complexity of the Viterbi Algorithm is $O(C^2L)$, where:

- $C$ is the number of possible tags.
- $L$ is the length of the sentence.

<br>

This algorithm is efficient because we only need to know the best sequence leading to the previous word thanks to the Markov assumption.

---

<br>

### Example of Viterbi Algorithm

Consider a sentence with four words and four possible part-of-speech tags: **Noun (N)**, **Verb (V)**, **Preposition (P)**, and **Article (A)**. Suppose we have the following transition and emission probabilities for a simple HMM:

- **Transition Probabilities**:
  - $P(N | A) = 0.5$
  - $P(V | N) = 0.6$
  - $P(P | V) = 0.4$
  - $P(A | P) = 0.7$

- **Emission Probabilities**:
  - $P(the | A) = 0.8$
  - $P(cat | N) = 0.6$
  - $P(jumps | V) = 0.7$
  - $P(over | P) = 0.5$

For the sentence "The cat jumps over," the Viterbi algorithm will:

1. Start by computing the initial probabilities for the word "the" and all possible tags.
2. For each subsequent word, compute the most likely tag sequence leading up to that word.
3. Backtrack to find the sequence of tags with the highest probability.

---

<br>

### Greedy Algorithm vs. Viterbi

- **Greedy Algorithm**: At each step, it selects the tag with the highest local probability. However, it may result in suboptimal sequences because it doesn’t consider future words.
- **Viterbi Algorithm**: Takes the future into account by maintaining multiple possible sequences and ensuring the global maximum is selected.

---

<br>

### Why Enumeration Won’t Work

If there are $C$ possible tags and a sentence has $L$ words, then there are $C^L$ possible tag sequences. For even moderately sized sentences and tagsets, this is computationally infeasible, as the number of possible sequences grows exponentially. The Viterbi Algorithm is much more efficient, reducing the problem to a polynomial-time solution.

---

<br>

### Applications of the Viterbi Algorithm

- **Part-of-Speech Tagging**: Assigning the most probable sequence of POS tags to a sentence.
- **Speech Recognition**: Decoding the most likely sequence of phonemes or words given acoustic signals.
- **Named Entity Recognition (NER)**: Identifying entities like names, dates, and organizations in text by assigning the most probable sequence of labels to the words.

---

<br>

### Probabilistic Tag Assignment

In **Probabilistic Tag Assignment**, instead of assigning the **most likely** tag sequence as done in the Viterbi algorithm, we calculate the probability of each possible tag for every word. This gives a **distribution** over the possible tags for each word, rather than selecting just one tag.

#### Key Points

1. **Posterior Probability** $P(t | w)$: For each word $w_i$, we compute the probability of each possible tag $t_i$ given the entire sequence of words in the sentence. This is different from the Viterbi algorithm, which finds the single most probable sequence of tags.

2. **Application**: This method is useful when we want to handle ambiguity or uncertainty in the tagging process. For instance, if a word could reasonably be tagged as both a noun and a verb, we can maintain both possibilities with associated probabilities.

3. **How It's Done**: The probabilistic tag assignment can be calculated using algorithms like the **Forward-Backward algorithm**, which calculates the total probability of a tag occurring at each position in the sentence by considering all possible sequences of tags.

---

<br>

### **Forward and Backward Probabilities**

The **Forward-Backward Algorithm** is essential for calculating the probabilities of each tag at each position in a sentence. It is used in Hidden Markov Models (HMMs) to compute the likelihood of a tag at a given position by summing over all possible tag sequences that lead to and follow that position.

#### Forward Probability ($\alpha$)

The **forward probability** $\alpha_t(s)$ is the probability of being in state $s$ after seeing the first $t$ observed events.

Intuitively, it is the sum of the probabilities over all states sequences that could lead to state $s$ for observations/words $o_1, \dots, o_t$:

$$\alpha_t(s) = P(o_1, \dots, o_t, state(o_t)=s)$$

We sequentially process the input, sweeping over each possible state for $o_t$. For each state $s$, we sum over all possible prior states of (the forward probability of the prior state) *(the transition between the states)* (the emission probability of $o_t$):

$$\alpha_t(s)=\sum_{s'=1}^N \alpha_{t-1}(s')a_{s',s}b_s(o_t)$$

This is nearly identical to the Viterbi algorithm, but sums over the probabilities instead of taking the max.

To compute the final probability of each tag, we normalize the forward probabilities in each column of the matrix so the probabilities sum to 1.

The probability of a word $o_t$ having state $s$ would be computed as
follows:

$$P(state(o_t) = s) = \frac{\alpha_t(s)}{\sum_{s'=1}^n \alpha_t(s')}$$

<br>

#### Backward Probability ($\beta$)

The **backward probability** $\beta_t(s)$ is the probability of seeing the observations from time $t+1$ to the end, given that we are in state $s$ at time $t$.

Intuitively, it is the sum of the probabilities over all state sequences that could lead to state $s$ for observations/words $o_{t+1}, \dots, o_T$:

$$\beta_t(s) = P(o_{t+1}, \dots, o_T|state(o_t) = s)$$

Initialization from the end of sequence! Similar to beginning of sentence $\phi$, we can define end of sentence $\Omega$ and initialize $\beta_T(s) = a_{s,\Omega}$.

The algorithm for computing backward probabilities is analogous to algorithm for forward probabilities, except we sweep from right to left:

$$\beta_t(s) = \sum_{s'=1}^N \beta_{t+1}(s')a_{s,s'}b_s(o_{t+1})$$

---

<br>

### Combining Forward and Backward Probabilities

The most accurate estimates will come from using both the forward and backward probabilities.

Intuitively, this approach takes into account both the context prior to a word and the context following it.

Compute all the forward and backward probabilities as described earlier, then normalize the nodes in each column so they sum to 1:

$$P(state(o_t) = s) = \frac{\alpha_t(s)\beta_t(s)}{\sum_{s'=1}^N \alpha_t(s')\beta_t(s')}$$

---

<br>

### **Viterbi vs. Forward-Backward Algorithm**

1. **Viterbi Algorithm**: Finds the **most probable sequence** of tags. It’s useful when we only care about the single best tag sequence.

2. **Forward-Backward Algorithm**: Computes the **posterior probabilities** for each tag at each position, allowing for more nuanced assignments and handling uncertainty in tag assignments. This method is essential when we want to know how likely each tag is, rather than just the most likely sequence.

---

<br>

### **Example of Forward and Backward Probabilities**

Consider a short sentence: "The cat jumps." We have the following possible tags: **Noun (N)**, **Verb (V)**, **Determiner (D)**.

- **Word 1 ("The")**: Can be tagged as **Determiner (D)**.
- **Word 2 ("cat")**: Can be tagged as **Noun (N)**.
- **Word 3 ("jumps")**: Can be tagged as **Verb (V)**.

**Forward Probabilities**:

- For $t_1 = D$ (first word):
  \[
  \alpha(D, 1) = P(D | \phi) P(\text{"The"} | D)
  \]
- For $t_2 = N$ (second word):
  \[
  \alpha(N, 2) = \alpha(D, 1) P(N | D) P(\text{"cat"} | N)
  \]
- For $t_3 = V$ (third word):
  \[
  \alpha(V, 3) = \alpha(N, 2) P(V | N) P(\text{"jumps"} | V)
  \]

**Backward Probabilities**:

- For $t_3 = V$ (third word):
  \[
  \beta(V, 3) = 1
  \]
- For $t_2 = N$ (second word):
  \[
  \beta(N, 2) = \beta(V, 3) P(V | N) P(\text{"jumps"} | V)
  \]
- For $t_1 = D$ (first word):
  \[
  \beta(D, 1) = \beta(N, 2) P(N | D) P(\text{"cat"} | N)
  \]

Finally, combining forward and backward probabilities gives the probability of each tag at each position.

---

<br>

### Summary - Viterbi Algorithm

- **Viterbi Algorithm** is a dynamic programming approach to finding the most probable sequence of hidden states in a model like HMM.
- It efficiently computes the best tag sequence by breaking the problem into smaller subproblems and storing intermediate results.
- It avoids the inefficiencies of enumeration or greedy algorithms by ensuring global optimality.
- **Probabilistic Tag Assignment** assigns a probability distribution over possible tags for each word, not just the single best tag sequence.
- **Forward and Backward Probabilities**: The Forward-Backward algorithm computes the probabilities for each tag at each position by considering both previous and following words in the sentence. This gives a more complete picture of tag likelihoods.
- **Viterbi Algorithm** finds the single most probable sequence, while **Forward-Backward** computes probabilities for each tag individually, allowing for probabilistic tag assignments.

---

<br>

## VIII - Sequence Labelling

Sequence labeling refers to the process of assigning labels to each element in a sequence.

### 1. Shallow Parsing

**Shallow Parsing** (also called **partial parsing** or **syntactic chunking**) is the process of identifying and labeling basic syntactic constituents in a sentence, such as **noun phrases (NPs)**, **verb phrases (VPs)**, and **prepositional phrases (PPs)**. Unlike full parsing, which generates a complete parse tree for a sentence, shallow parsing focuses on identifying only local syntactic structures.

#### Key Features

- **Flat Representation**: Shallow parsers typically produce a **flat syntactic representation** of the sentence without recursion. The output is non-recursive syntactic chunks.
- **NP, VP, PP Identification**: They aim to identify phrases like noun phrases (NP), verb phrases (VP), and prepositional phrases (PP).
- **Speed and Robustness**: Shallow parsers are faster and more robust compared to full parsers, making them useful for ungrammatical input (e.g., social media text or speech transcripts).
  
#### Example of Shallow Parsing

Consider the sentence: "The election in the U.S. will occur in November."

- Shallow parsing might produce the following chunks:
  
  - [NP: The election] [PP: in the U.S.] [VP: will occur] [PP: in November]

The parser segments the sentence into its major syntactic components but does not build a full parse tree that captures the deep structure of the sentence.

#### Rule-Based vs. Classifier-Based Parsing

- **Rule-Based Parsing**: A simple rule-based shallow parser might be used to identify noun phrases by defining rules for which words can appear together as an NP.
- **BIO Tagging**: A classifier can be trained using **BIO (Begin-Inside-Outside) labeling**, where each word is tagged as either the beginning of a chunk (B), inside a chunk (I), or outside a chunk (O).

---

<br>

### 2. Named Entity Recognition (NER)

**Named Entity Recognition (NER)** is a key task in sequence labeling, where the objective is to identify **named entities** in text, such as people, organizations, locations, dates, and measures. NER systems can also recognize specific types of entities like **URLs**, **email addresses**, and **measurements**.

#### Common Named Entities

- **People**: e.g., *Elvis Presley*.
- **Organizations**: e.g., *IBM*.
- **Locations**: e.g., *United States*.
- **Dates & Times**: e.g., *November 9, 1997*.
- **Measures**: e.g., *65 mph*, *$1.4 billion*.

#### Challenges in NER

1. **New Names**: Proper names are constantly evolving, and no dictionary can contain all existing proper names.
2. **Mixed Case Texts**: Identifying proper names in texts can be difficult when the text is in all uppercase or lowercase (e.g., headlines, spoken transcripts).
3. **Ambiguity**: Some names can refer to multiple types of entities (e.g., *Jordan* can be a country or a person).

---

<br>

### 3. Rule-Based vs. Machine Learning NER

#### Rule-Based NER

- **Advantages**: Performs well in specialized domains where rules are well-defined.
- **Disadvantages**: Expensive to build and domain-specific, making it less adaptable.

#### Machine Learning-Based NER

- **Advantages**: Can be easily adapted for new domains and can automatically learn patterns from a labeled training corpus.
- **Disadvantages**: Requires a large, annotated dataset for training.

---

<br>

### 4. Common Types of Rules for NER

Some common types of rules used in rule-based NER systems include:

- **List Matching**: Matching words against predefined lists (e.g., person names, organizations, locations).
- **Pattern Matching**: Surface structure patterns such as:
  - Dates: *Month/Day/Year* format.
  - Email addresses: *<xxxxx@xxx.xxx>*.
  - Phone numbers: *(###) ###-####*.
- **Contextual Patterns**: For instance, locations may appear in phrases like "moved to X" or "located in X."

---

<br>

### 5. Machine Learning Models for Sequence Labeling

Several machine learning models are used for sequence labeling tasks like **NER**, **POS tagging**, and **chunking**. These models are designed to assign labels to sequences based on the dependencies between neighboring items (words) in the sequence.

#### Key Models

1. **Hidden Markov Model (HMM)**: A generative model that computes the probability of a sequence of labels given the observed words and transitions between labels.
2. **Maximum Entropy Markov Model (MEMM)**: A discriminative model that uses logistic regression to predict the label at each position in the sequence, based on the previous label and the current word.
3. **Conditional Random Fields (CRF)**: Another discriminative model similar to MEMM, but trained globally to maximize the probability of the entire sequence rather than individual states.

---

<br>

### 6. BIO Tagging for NER

When NER is viewed as a classification or sequence tagging task, every word is labeled based on whether it is part of a named entity, with **BIO tagging**:

- **B**: The word is the beginning of a named entity.
- **I**: The word is inside a named entity.
- **O**: The word is outside of a named entity.

#### Example - BIO Tagging

For the sentence "John Smith gave Mary a book about Alaska," BIO tagging might produce:

    John/B-PER Smith/I-PER gave/O Mary/B-PER a/O book/O about/O Alaska/B-LOC

---

<br>

### 7. Hidden Markov Models (HMM) for NER

**HMMs** can be applied to NER in much the same way as for POS tagging. Given a training corpus labeled with named entities (NEs), an HMM can learn:

- **Emission probabilities**: The likelihood of a word given a named entity label - $P(word_i|tag_i)$.
- **Transition probabilities**: The likelihood of transitioning from one named entity label to another - $P(tag_i|tag_{i-1})$. The tags are the NE labels.

The **Viterbi algorithm** is then used to find the most probable sequence of named entity labels for a given sentence. A limitation of this approach is that it cannot use arbitrary feature.

---

<br>

### 8. Maximum Entropy Markov Models (MEMM)

**MEMMs** are discriminative models used for sequence labeling tasks like NER. MEMMs directly compute the probability of a sequence of labels given a sequence of words by applying logistic regression at each position in the sequence.

#### MEMM Formula

Given a sentence $w^{(1)}, \dots, w^{(n)}$, the probability of a tag sequence $t^{(1)}, \dots, t^{(n)}$ that maximizes $P(t^{(1)}, \dots, t^{(n)}|w^{(1)}, \dots, w^{(n)})$ directly, is computed as:

$$P(t^{(1)}, \dots, t^{(n)}|w^{(1)}, \dots, w^{(n)}) = \prod_{i=1}^{n} P(t^{(i)} | t^{(i-1)}, w^{(i)})$$

Where $P(t^{(i)} | t^{(i-1)}, w^{(i)})$ is the probability of the current tag based on the previous tag and the current word.

MEMMs use a logistic regression (“Maximum Entropy”) classifier for each $P(t^{(i)} | w^{(i)}, t^{(i-1)})$.

Recall multinomial logistic regression:

$$P(y^{(j)}|x) = \text{softmax}(\theta_0^{(j)} + \sum_{i=1}^n \theta_i^{(j)} \cdot x_i)$$

$$P(t^{(i)} | t^{(i-1)}, w^{(i)}) = \frac{e^{\sum_j \theta_{jk} f_j(t^{(i-1)}, w^{(i)})}}{\sum_l e^{\sum_j \theta_{jl} f_j(t^{(i-1)}, w^{(i)})}}$$

where $t^{(i-1)}$ is the label of the $i^{th}$ word, vs. $t_i$ which is the $i^{th}$ label in the label list.

This requires the definition of a feature function $f(t^{(i-1)}, w^{(i)})$ that returns an $n$-dimensional feature vector.

Training weights $\lambda_{jk}$ for each feature $j$ is used to predict the label $t_k$.

---

<br>

### 9. Conditional Random Fields (CRF)

**CRFs** are also discriminative models, but they improve upon MEMMs by modeling the entire sequence of labels globally, rather than locally optimizing for each individual label. This helps CRFs avoid the **label bias problem** that affects MEMMs.

#### Advantages of CRFs

- **Global Optimization**: CRFs maximize the likelihood of the entire sequence, ensuring better overall performance.
- **Handling Arbitrary Features**: CRFs can incorporate a wide variety of features for each word, such as neighboring words, capitalization, and more.

---

<br>

### 10. Summary - HMM vs. MEMM vs. CRF

- **HMM**: A generative model that focuses on the joint probability of sequences of words and labels.
- **MEMM**: A discriminative model that predicts the probability of the current label based on the previous label and current word, but it may suffer from label bias.
- **CRF**: A discriminative model that avoids label bias by optimizing the probability of the entire label sequence rather than individual transitions.

---

<br>

### Example NER System: MENERGI

The **MENERGI system** is an example of a **Maximum Entropy** approach to Named Entity Recognition. It uses a combination of local and global features:

- **Local Features**: Based on the current word, neighboring words, and properties like capitalization.
- **Global Features**: Extracted from instances of the same token occurring elsewhere in the document.

MENERGI recognizes 7 types of named entities, based on the MUC-6/7 NER task definition:

- Person
- Organization
- Location
- Date
- Time
- Money
- Percent

Each class is sub-divided into 4 sub-classes:

- Begin/Continue/End
- Unique

Since there are 4 sub-classes for each of the 7 NE classes and one outside tag (not a NE), there are a total of $(7*4)+1=29 \space \text{classes}$.

---

<br>

## IX - Lexical Semantics

The area of **lexical semantics** deals with understanding the meanings of words and their relationships.

### 1. Introduction to Lexical Semantics

Lexical semantics is the branch of linguistics concerned with how words convey meaning. In NLP, understanding word meanings helps in many tasks, such as **machine translation**, **question answering**, and **information retrieval**. We need to understand how to represent the meanings of individual words and their relationships to other words.

#### **How to Get the Meaning of Words?**

There are two main approaches to understanding word meaning in NLP:

1. **Dictionary-Based Approach**:
   - This approach uses **dictionaries**, **lexicons**, or **ontologies** to understand word meanings. These resources contain predefined word definitions that can be consulted to obtain a word’s meaning.
   - Example: **WordNet** is an example of a lexicographic resource used in NLP.

2. **Distributional Approach**:
   - In this approach, word meanings are inferred from their **distribution in large text corpora**. This approach is based on the **distributional hypothesis**, which suggests that words that occur in similar contexts have similar meanings.
   - Example: A word’s meaning can be derived from the contexts in which it frequently appears. If "dog" and "cat" often appear in similar sentences, we might infer that they are semantically related.

---

<br>

### 2. Word Senses

In lexical semantics, the concept of **word senses** is crucial. A word may have multiple possible meanings (senses), and distinguishing between these senses is important for NLP systems.

#### **Important Concepts**

- **Word Form**: The actual inflected word that appears in the text, such as "reads" or "reading."
  
- **Lemma**: The base form of a word, which represents all of its different inflected forms. For example, the lemma for "reads" and "reading" is "read."

- **Word Sense**: A **discrete representation** of a specific meaning of a word. For example, the word "bank" can refer to both a **financial institution** and the **side of a river**. Each meaning is a different sense of the word "bank."

---

<br>

### 3. Homonymy and Polysemy

#### Homonymy

Homonymy occurs when two words share the **same spelling** or **pronunciation** but have **unrelated meanings**.

- Example:

  - **plane** (as in airplane)
  - **plane** (as in a flat surface)
  
In this case, both meanings are distinct, and the word "plane" represents different concepts.

#### Polysemy

Polysemy occurs when a word has **different related meanings**.

- Example:
  
  - **bank** (a financial institution)
  - **bank** (the building where financial services are offered)
  
In this case, both meanings are related to finance, making "bank" a polysemous word.

---

<br>

### 4. Metonymy

Metonymy is a figure of speech where a word is used to refer to something **related to** its primary meaning. This is a form of **systematic polysemy**, where the relationships between word senses are predictable.

- **Examples of Metonymy**:
  - **Container vs. Content**: "Sip the glass" (meaning sip the drink inside the glass).
  - **Producer vs. Product**: "Listen to Michael Jackson" (meaning listen to music produced by Michael Jackson).
  - **Organization vs. Building**: "Went to Walmart" (could mean visiting the physical store or interacting with the corporation).
  - **Capital City vs. Government**: "Moscow announced" (refers to the Russian government, not the city itself).

These systematic relationships allow NLP systems to interpret word meanings based on context.

---

<br>

### 5. Synonyms and Antonyms

- **Synonyms**: Words that have the **same meaning** in certain contexts.
  - Example: "big" and "large" are synonymous in many contexts, like "How big/large is that plane?"
  
- **Antonyms**: Words that have **opposite meanings**.
  - Example: "hot" vs. "cold," or "big" vs. "small."

It’s important to note that **synonymy and antonymy** are relationships between **word senses**, not between the words themselves. For instance, "big" and "large" are synonymous in many contexts, but not all (e.g., “big sister” vs. “large sister” doesn’t work).

---

<br>

### 6. Hypernymy and Hyponymy

These are **semantic relationships** that describe a hierarchical relationship between concepts.

- **Hyponym**: A more **specific** term that falls under a broader category.
  - Example: "Dog" is a hyponym of "animal."
  
- **Hypernym**: A **broader** term that encompasses more specific concepts.
  - Example: "Vehicle" is a hypernym of "car."

The relationship is often referred to as the **IS-A relationship**. For example, "dog IS-A animal" and "car IS-A vehicle."

---

<br>

### 7. Meronymy and Holonymy

These relationships deal with **part-whole** structures in meaning.

- **Meronym**: A term that denotes a **part** of something.
  - Example: "Keycap" is a meronym of "keyboard."
  
- **Holonym**: A term that refers to the **whole** of which something is a part.
  - Example: "Keyboard" is a holonym of "keycap."

This is sometimes called a **HAS-A relationship**, where the larger entity contains the smaller part (e.g., a keyboard has a keycap).

---

<br>

### 8. WordNet

**WordNet** is a large lexical database of English, where words are grouped into sets of **synonyms** (called **synsets**), each representing a single/distinct concept. Each synset is linked to others through semantic relations like **hypernyms**, **hyponyms**, and **antonyms**.

- WordNet covers:
  - **118,000 nouns**
  - **12,000 verbs**
  - **22,000 adjectives**
  - **4,500 adverbs**
  - **81,000 noun synsets**
  - **13,000 verb synsets**
  - **19,000 adjective synsets**
  - **3,500 adverb synsets**

#### Word Senses and Synsets

- Each word is associated with one or more senses, represented in **synsets**.
- The synset represents a distinct concept.
- Each synset is accompanied by a **gloss**, which is a brief definition or description of the concept.

- **Example of a Synset**:
  - **plane.n.01**: an aircraft that has a fixed wing and is powered by propellers or jets.
    - "n" means it's a noun.
    - "01" means it is the second sense of the word form “plane”.
  - It belongs to the ***Synset(“airplane.n.01”)***, which includes includes "plane.n.01", "airplane.n.01", and "aeroplane.n.01".

<br>

WordNet synsets are connected by various **relations**:

- **IS-A relation**: hypernyms and hyponyms, which form a ***transitive*** hierarchy. For example, an armchair is a kind of chair, and a chair is a kind of furniture, so an armchair is a kind of furniture.
- **HAS-A relation**: part-whole relationships (meronyms and holonyms).
- **Antonym relation**: connects words that have opposite meanings.

---

<br>

### 9. Word Similarity

Word similarity measures how closely two words are related in meaning. There are two key concepts in this area:

1. **Synonymy**: A **binary relation** where two words are either synonymous or not. For instance, "big" and "large" are synonyms, but "big" and "small" are not.

2. **Similarity**: A **looser metric** where two words can have different degrees of similarity based on shared features.
   - **Example**: "sofa" and "table" are more similar than "sofa" and "car" because both sofa and table are furniture, whereas a car is not.

**Edit distance** (a metric for how similar words are based on their spelling) is not always a reliable indicator of word similarity, as words with similar spelling (e.g., "sofa" and "soda") can have very different meanings.

---

<br>

### 10. Path-Based Similarity

Path-based similarity calculates the **distance** between word senses in a semantic hierarchy like **WordNet**. The shorter the path between two senses, the more similar they are. It is just the distance between synsets.

- **Formula**: The **path length** is the number of **edges** (connections) between two senses:
  
  $$\text{pathlen}(s_1, s_2) = 1 + \text{number of edges in the shortest path in the hypernym graph between sense nodes} \space s_1 \space \text{and} \space s_2$$

- The **similarity** is the inverse of the path length:
  
  $$\text{simpath}(s_1, s_2) = \frac{1}{\text{pathlen}(s_1, s_2)}$$

- The **word similarity** is give by:

$$\text{wordsim}(w_1, w_2) = \max_{s_1 \in senses(w_1), \space s_2 \in senses(w_2)} \text{simpath}(s_1, s_2)$$

- **Example**:
  - The similarity between "nickel" and "coin" is higher (shorter path) than the similarity between "nickel" and "money."

While this method works well, it has limitations, such as treating different hierarchical paths as equivalent, even when the semantic distance might not be uniform across paths.

---

<br>

### 11. Dictionary-Based Similarity

Another approach to word similarity is to use **dictionary definitions** (glosses) to compare words.

- **Gloss Overlap**: The more words two glosses share, the more similar the words are.
  
- **Example**:
  - The gloss

 for "bank" (financial institution) may share words with the gloss for "vault" (a secure storage facility), indicating a higher similarity.

This method can be extended by incorporating **hypernyms** and **hyponyms** to capture more nuanced similarities.

---
>
<br>

### 12. Word Sense Disambiguation (WSD)

**Word Sense Disambiguation (WSD)** is the task of determining the correct sense of a **polysemous word** (a word with multiple meanings) in a specific context.

WSD systems typically assume the presence of a sense inventory (set of all possible senses) for each polysemous word using a lexical resource such as WordNet.

Using the most common sense is a reasonable baseline approach. The first sense listed in WordNet is usually considered to be the most common sense.

For instance, in the sentence "I went to the bank," the system needs to determine whether "bank" refers to a **financial institution** or the **side of a river**.

---

<br>

### 13. Lesk Algorithm

One of the oldest knowledge-based algorithms for WSD is the **Lesk algorithm**, which uses **gloss overlap** to disambiguate word senses.

It uses a dictionary, such as *WordNet*. It disambiguates word by measuring the overlap between the words in its context and the words in each sense definition. The sense with the highest overlap wins.

#### Simplified Lesk Algorithm

This method counts the number of overlapping non-stopwords between the context of the word and the glosses of its possible senses. The sense with the highest overlap is selected.

#### Supervised Approaches

- **Annotated Corpora**: A supervised WSD system can be trained using labeled data (e.g., the **SemCor** corpus, where each word is annotated with its WordNet sense).
- **Naive Bayes Classifier**: This probabilistic model can be trained to predict word senses based on the prior probabilities of senses and the likelihood of words occurring with specific senses.

---

<br>

### 14. Bootstrapping Algorithm for Word Sense Disambiguation

The **bootstrapping algorithm** for WSD was pioneered by **Yarowsky (1995)**. This method leverages **weak supervision**, meaning it does not require fully annotated training data. Instead, it begins with a small set of **seed examples** (initial training data) and gradually expands the set by labeling additional examples based on the model’s predictions. This iterative process makes it a powerful method for WSD, especially when labeled data is scarce.

#### Key Ideas Behind Bootstrapping

The Yarowsky algorithm is based on two key linguistic observations:

1. **One Sense per Collocation**: Nearby words (collocations) provide strong and consistent clues to a word's sense. That is, a word is likely to have the same meaning when it appears in the same context. For instance, "bass" followed by "guitar" is almost always the musical sense of "bass," whereas "bass" near "river" or "fishing" will refer to the fish.

2. **One Sense per Discourse**: The sense of a word is typically consistent throughout a document. Once a particular meaning of a word is chosen in a discourse, that meaning is usually maintained in subsequent sentences. For example, if a newspaper article discusses "bass" as a fish, it will likely continue to refer to the fish and not switch to the musical instrument.

<br>

#### Steps of the Bootstrapping Algorithm

The bootstrapping algorithm proceeds through the following steps:

1. **Collect the Context Words**:
   - For each instance of the target word (e.g., "bass"), collect the **context words** in a predefined **window** around it (e.g., the 5 words before and after "bass"). These context words provide clues about which sense of the word is being used.
   - For example, if the target word is "bass," the context words could include "guitar," "river," "sound," "fishing," etc.

2. **Seed Labeling**:
   - For each word sense, manually label a **small set of seed examples**. These seeds act as **positive training examples** for the algorithm. The remaining untagged instances are referred to as the **residual**.
   - Example:
     - **Seed for "bass" as a fish**: words like "fish," "river," "water."
     - **Seed for "bass" as a musical instrument**: words like "guitar," "jazz," "sound."

3. **Train an Initial Classifier**:
   - Use the **seed examples** to train a basic supervised classifier (such as **Naive Bayes** or another classifier). The classifier is trained to differentiate between the senses based on the context words in the examples.

4. **Classify Unlabeled Instances**:
   - Apply the classifier to the **residual instances** (those that haven't been labeled yet) and assign a sense to them.
   - The classifier’s task is to predict which sense of the word (e.g., "bass" as a fish or "bass" as a musical instrument) applies to each unlabeled instance.
   - For example, if the classifier sees "bass" with context words like "guitar" or "sound," it would predict the "musical instrument" sense.

5. **Threshold-Based Labeling**:
   - For each instance classified by the model, calculate a **confidence score**. Instances with a score above a predefined **threshold** are considered **confident** predictions, and their sense labels are added to the training set as new **positive examples**.
   - The **most confidently labeled instances** are used to **expand the training set** for the next iteration.

6. **One Sense per Discourse Heuristic**:
   - Optionally, after each iteration (or at the end of the process), apply the **one sense per discourse** heuristic. This means if the target word appears multiple times in the same document, it is likely to have the same sense throughout the document. The algorithm can use this rule to automatically label other occurrences of the word in the document with the same sense.

7. **Repeat the Process**:
   - If new examples are added to the training set, go back to **Step 3** and retrain the classifier with the expanded data. Repeat the process until no more instances are confidently labeled or until the classifier reaches a performance threshold.

---

<br>

#### **Example of Bootstrapping for Word Sense Disambiguation**

Let’s take the word **"plant"** as an example, which has two major senses:

- **Plant-A**: A living organism (tree, flower).
- **Plant-B**: An industrial facility (factory).

#### **Step-by-Step Walkthrough:**

1. **Collect Context Words**:
   - For instances of "plant," collect context words like "flower," "life," "factory," "manufacturing," etc.

2. **Seed Labeling**:
   - Seeds for **Plant-A (living organism)** might include context words like "life," "green," "grow."
   - Seeds for **Plant-B (factory)** might include words like "factory," "building," "manufacturing."

3. **Train Initial Classifier**:
   - Train the classifier using the labeled seed examples. The classifier learns to associate words like "life" with **Plant-A** and words like "factory" with **Plant-B**.

4. **Classify Unlabeled Instances**:
   - Apply the classifier to the remaining unlabeled examples and predict whether each instance refers to **Plant-A** or **Plant-B** based on the surrounding words.

5. **Threshold-Based Labeling**:
   - For example, if the classifier sees "plant" in the context of "manufacturing," it might confidently label it as **Plant-B**. If the confidence score is high, the instance is added to the training set for **Plant-B**.

6. **Apply One-Sense-Per-Discourse**:
   - If "plant" is labeled as **Plant-B** in one part of a document, it is assumed to have the same sense throughout the document, so other instances of "plant" in the same document are also labeled **Plant-B**.

7. **Repeat**:
   - The process is repeated, and the training set grows as more confidently labeled examples are added, improving the classifier’s accuracy with each iteration.

---

### **Pros and Cons of the Bootstrapping Approach**

#### **Advantages**

1. **No Annotated Data Needed**:
   - Bootstrapping requires only a small amount of **initial seed data**, which can be manually labeled. After that, the algorithm can learn from raw text data without requiring a fully annotated corpus.

2. **One-Sense-Per-Discourse Heuristic**:
   - The one-sense-per-discourse heuristic is particularly useful because it allows the algorithm to **propagate labels** through a document, minimizing misclassifications that might occur in individual sentences.

#### **Disadvantages**

1. **Coverage**:
   - Some contexts may lack sufficient clues to confidently assign a sense to the word, which could limit the **coverage** of the method.

2. **Difficulty with Similar Senses**:
   - When multiple senses belong to the **same general domain**, distinguishing between them can be difficult. For instance, differentiating between "bass" as a **musical instrument** and **bass** as a **type of vocal range** can be challenging because both belong to the music domain.

---

<br>

### Summary - Lexical Semantics

- **Lexical semantics** helps us understand how words convey meaning and how different meanings are related.
- Words can have multiple **senses** (homonymy and polysemy), and these senses can be related through semantic relations like **hypernymy**, **hyponymy**, **meronymy**, and **holonymy**.
- Tools like **WordNet** help organize word senses into a semantic hierarchy, which can be used to compute **word similarity**.
- **Word Sense Disambiguation (WSD)** is the task of choosing the correct word sense in context, and this can be done through both **knowledge-based** (e.g., Lesk) and **supervised** approaches (e.g., Naive Bayes).
- The bootstrapping algorithm for WSD is an iterative process that begins with **seed examples** and gradually expands the training set by labeling new instances.
  - The algorithm relies on two key ideas: **one sense per collocation** (the context around a word helps determine its meaning) and **one sense per discourse** (a word’s sense is usually consistent throughout a document).
  - This method is particularly effective in scenarios where labeled data is limited, making it a powerful tool for **semi-supervised learning** in NLP.

---

<br>

## X - Distributional Representations

**Distributional representations** of words is a crucial concept in **lexical semantics** and NLP. The central idea is that the meaning of a word can be inferred from the **contexts** in which it appears.

---

<br>

### 1. The Distributional Hypothesis

The **distributional hypothesis** is a fundamental principle in NLP, first articulated by **Zellig Harris (1954)** and later popularized by **J.R. Firth (1957)**. It suggests that:

> “**You shall know a word by the company it keeps!**”

This means that words that occur in similar contexts tend to have similar meanings. The idea is that the context of a word can reveal its semantic properties.

#### **Intuition Behind Distributional Semantics**

Even without knowing the exact meaning of a word, you can infer its general meaning by observing the words around it. For example, consider the following sentence with the made-up word **"shuke"**:

- "The **shuke** weighs 10,000 kg."
- "The **shuke** strengthens national security."
- "DOD plans to procure nearly 2,500 **shukes**."
- "The **shukes** can store up to six missiles internally."

From this context, you can infer that "shuke" likely refers to some kind of **military plane**. This illustrates the power of the distributional hypothesis: the context provides strong clues about the word’s meaning.

---

<br>

### 2. Distributional Similarity

According to the distributional hypothesis, two words are **distributionally similar** if they appear in **similar contexts**. In NLP, we represent words as **vectors** in a high-dimensional space, where each dimension corresponds to a different context. the vector captures how strongly the word is associated with each context. The similarity between two words is then measured as the **similarity between their vectors**.

Words are represented as vectors of the form:

$$w = (w_1, \dots, w_n) \in \mathbb{R}^n$$

in an **$n$-dimensional space**, where:

- Each **dimension** corresponds to a particular **context** $c_i$.
- The **value** of each dimension, $w_i$, shows how strongly the word is associated with that context, $c_i$.

The similarity between words u and v is computed as the similarity of their vectors $u$ and $v$.

For example, if we represent the word "plane" in a context vector space, its vector could include dimensions for contexts like "flies," "airport," "pilot," etc., with values reflecting how strongly each context is associated with the word.

---

<br>

### 3. Distance and Similarity Metrics

When representing words as vectors, we need a way to **measure similarity** between them. Several metrics can be used for this purpose:

#### 1. Manhattan Distance

The **Manhattan distance** (also known as **L1 distance** or **taxicab distance**) is the sum of the absolute differences between corresponding elements of two vectors:

$$\text{ManhattanDistance}(u, v) = \sum_{i=1}^n |u_i - v_i|$$

- **Example**:
  - $u = (1,2,3,4)$
  - $v = (2,3,4,5)$
  - ManhattanDistance = $|1 - 2| + |2 - 3| + |3 - 4| + |4 - 5| = 4$

Manhattan distance is sensitive to **outliers** and can grow as the vector length increases.

<br>

#### 2. Euclidean Distance

The **Euclidean distance** (also known as **L2 distance**) is the square root of the sum of the squared differences between corresponding elements of two vectors:

$$\text{EuclideanDistance}(v, u) = \sqrt{\sum_{i=1}^n (v_i - u_i)^2}$$

- **Example**:
  - $u = (1,2,3,4)$
  - $v = (2,3,4,5)$
  - EuclideanDistance = $\sqrt{1^2 + 1^2 + 1^2 + 1^2} = 2$

Like Manhattan distance, Euclidean distance is sensitive to outliers.

<br>

#### 3. Jaccard Similarity

The **Jaccard similarity** metric measures the amount of **overlap** between the features of two vectors:

$$\text{Jaccard}(v, u) = \frac{\sum_{i=1}^n \min(v_i, u_i)}{\sum_{i=1}^n \max(v_i, u_i)}$$

- **Example**:
  - $u = (1,2,3,4)$
  - $v = (2,3,4,5)$
  - JaccardSimilarity = $\frac{1 + 2 + 3 + 4}{2 + 3 + 4 + 5} = \frac{10}{14}$

Jaccard similarity ignores the **length of vectors**, but it is still sensitive to outliers.

<br>

#### 4. Cosine Similarity

**Cosine similarity** is one of the most widely used metrics in NLP for measuring similarity between vectors. It calculates the cosine of the angle between two vectors, which is essentially the **dot product normalized** by the magnitudes of the vectors:

$$\text{Cosine}(v, u) = \frac{\sum_{i=1}^n v_i \cdot u_i}{\sqrt{\sum_{i=1}^n v_i^2} \cdot \sqrt{\sum_{i=1}^n u_i^2}}$$

- **Example**:
  - $u = (1,2,3,4)$
  - $v = (2,3,4,5)$
  - CosineSimilarity = $\frac{1\cdot2 + 2\cdot3 + 3\cdot4 + 4\cdot5}{\sqrt{1^2 + 2^2 + 3^2 + 4^2} \cdot \sqrt{2^2 + 3^2 + 4^2 + 5^2}} = \frac{40}{\sqrt{30} \cdot \sqrt{54}} = 0.915$

<br>

Cosine similarity is **robust**, meaning it is less affected by vector length and outliers. Values range from:

- **1** when the vectors $v$ and $u$ are identical (point in the same direction).
- **0** when the vectors $v$ and $u$ are **orthogonal** (unrelated).
- **-1** when the vectors $v$ and $u$ point in **opposite directions**.

---

<br>

### 4. Term-Document Matrix

A **term-document matrix** (TDM) is a two-dimensional table where:

- Each **row** represents a **term** (word).
- Each **column** represents a **document**.
- Each **cell** contains the **frequency count** of how often the term, $t$ appears in the document, $d$, denoted by $TF(t, d)$, which is also called **term frequency**.

#### **Example of a Term-Document Matrix**

|         | Doc1 | Doc2 | Doc3 | Doc4 |
|---------|------|------|------|------|
| battle  |   1  |   1  |   8  |  15  |
| soldier |   2  |   2  |  12  |  36  |
| fool    |  37  |  58  |   1  |   5  |
| clown   |   6  | 117  |   0  |   0  |

The TDM can be used to measure similarity between documents (based on their term vectors) or between words (based on their document vectors).

---

<br>

### 5. TF-IDF (Term Frequency-Inverse Document Frequency)

**TF-IDF** is a widely used **term-weighting scheme** in **information retrieval (IR)**. It helps identify which words are more important in a document by balancing two factors:

1. **Term Frequency (TF)**: The number of times a term appears in a document:

   $$\text{TF}(t,d) = \text{\# of occurrences of term } t \text{ in document } d$$

2. **Document Frequency (IDF)**:

    $$\text{DF}(t) = \text{\# of documents that contain the term} \space t$$

3. **Inverse Document Frequency (IDF)**: A measure of how common or rare a term is across all documents:

   $$\text{IDF}(t) = \log \left( \frac{N}{\text{DF}(t)} \right)$$

   where $N$ is the total number of documents, and $\text{DF}(t)$ is the number of documents that contain term $t$.

#### TF-IDF Formula

$$\text{TF-IDF}(t,d) = \text{TF}(t,d) \cdot \text{IDF}(t)$$

The TF-IDF value for terms that occur in all documents is:

$$\log (\frac{N}{\text{DF}(t)}) = \log 1 = 0$$

TF-IDF assigns higher weights to terms that are frequent in a document but rare across the collection, thus helping in **highlighting relevant terms**.

#### Example - TF-IDF

- If the word "soldier" appears frequently in a document but rarely in others, its TF-IDF score will be high for that document.
- Common words like "a" or "the" will have low TF-IDF scores because they appear in almost every document.

---

<br>

### 6. Word-Word Co-occurrence Matrix

A **word-word matrix** is similar to a term-document matrix but focuses on word co-occurrences within a given context.

- Each **cell** represents how frequently one word co-occurs with another word.
- **Context** can be defined in various ways:
  - Words within a **fixed window** (e.g.,$±5$ words).
  - Words within the **same sentence**.
  - Words that are grammatically related (e.g., subject-object pairs).

#### Example of a Word-Word Matrix

|          | sugar | lemon | data  |
|----------|-------|-------|-------|
| lemon    |   1   |   1   |   0   |
| pineapple|   0   |  10   |   0   |
| computer |   0   |   0   |  18   |
| information| 0   |   0   |  20   |

This matrix helps measure **word similarity** based on shared contexts.

---

<br>

### 7. Pointwise Mutual Information (PMI)

**PMI** measures the **association** between two words comparing the observed joint distribution of two terms with the probability that they should occur together if they are independent::

$$\text{PMI}(w, c) = \log \frac{P(w, c)}{P(w)P(c)}$$

- $P(w, c)$: Joint probability of words $w$ and $c$ co-occurring.
- $P(w)$ and $P(c)$: Marginal probabilities of words $w$ and $c$, respectively.

PMI gives higher scores to word pairs that **co-occur more frequently than expected by chance**. PMI can also be negative (co-occurring less often than expected by chance).

#### **Example**

- **PMI("barked", "dog")**: If "barked" and "dog" co-occur frequently, their PMI score will be high.
- **PMI("the", "dog")**: Since "the" occurs with almost any word, its PMI score with "dog" will be lower.

<br>

#### Positive PMI (PPMI)

To avoid negative PMI scores (which occur when words co-occur less often than expected), we use **Positive PMI**:

$$\text{PPMI}(w, c) = \max(0, \log \frac{P(w, c)}{P(w)P(c)})$$

---

<br>

### Pointwise Mutual Information (PMI) Example

Let's compute PMI for the words **"barked"** and **"dog"**. Assume the following probabilities:

- $P(\text{barked}) = 0.02$
- $P(\text{dog}) = 0.03$
- $P(\text{barked and dog}) = 0.02$

Using the PMI formula:

\[
\text{PMI(barked, dog)} = \log_2 \left( \frac{0.02}{0.02 \cdot 0.03} \right) = \log_2 \left( \frac{0.02}{0.0006} \right) = \log_2 (33.33) = 5.06
\]

This shows that "barked" and "dog" co-occur more frequently than we would expect by chance.

---

<br>

### 8. Sparse vs. Dense Vectors

#### **Sparse Vectors**

- **Long and sparse** vectors like **TF-IDF** or **PMI** vectors are often used to represent words, where each dimension corresponds to a different context or document.
- Sparse vectors contain many **zero values**, which can make them computationally expensive to store and manipulate.
- *Long because the dimension equals the # of documents (TF-IDF) or vocabulary size (PMI).*
- *Sparse because most elements are zero.*

#### **Dense Vectors**

- **Dense vectors** (e.g., word embeddings) are typically **shorter** (length 50-1000) and contain **mostly non-zero values**.
- Dense vectors are more efficient and often better at capturing **semantic relationships** like **synonymy** (e.g., "car" and "automobile").
- *Short because their length is typically between 50-1000.*
- *Dense because most of the elements are non-zero.*

Dense vectors can be learned through **count-based methods** (like **Singular Value Decomposition (SVD)**) or **prediction-based methods** (like **word embeddings** trained via neural networks).

---

<br>

### **Summary - Distributional Representations**

- **Distributional representations** of words are based on the **contexts** in which they occur.
- Words are represented as **vectors** in high-dimensional space, and their similarity is measured using metrics like **cosine similarity**, **Manhattan distance**, or **PMI**.
- Techniques like **TF-IDF** and **PMI** are used to assign weights to word-context pairs, while **word embeddings** offer a dense, efficient representation of words that can capture deep semantic relationships.

---

<br>

## XI - Word Embeddings

**Word embeddings** are dense vector representations of words in continuous vector space. Word embeddings have revolutionized NLP by capturing the **semantic similarity** between words. Unlike traditional methods such as **TF-IDF** or **PMI**, which produce **sparse** vectors, word embeddings are **dense**, lower-dimensional vectors, making them more efficient and effective for many machine learning tasks.

---

<br>

### 1. Sparse vs. Dense Vectors

Before we dive into word embeddings, let's first clarify the difference between **sparse** and **dense** vectors:

#### Sparse Vectors

- **High-dimensional**: Traditional vector representations, such as **TF-IDF** or **PMI** vectors, can have dimensions equal to the size of the **vocabulary** or **document corpus**. This results in vectors that are often extremely large.
- **Sparse**: Most of the elements in these vectors are **zero**, which makes them inefficient to store and process.
  
#### Dense Vectors

- **Low-dimensional**: Word embeddings are typically **300-dimensional**, regardless of the size of the corpus.
- **Dense**: Most elements in the vector have **non-zero** values, making them much more efficient and useful for tasks like machine learning.

#### **Why Use Dense Vectors?**

- **Easier for machine learning**: Dense vectors contain fewer parameters (weights), which makes it easier to train models like neural networks.
- **Capture synonymy better**: In **sparse vectors**, synonyms like "car" and "automobile" may be represented as different dimensions. However, in dense vectors, words with similar meanings tend to be located **close to each other** in vector space.

---

<br>

### 2. How to Obtain Dense Vectors?

There are two major approaches for obtaining dense vectors:

#### **1. Count-based Methods**

- **Singular Value Decomposition (SVD)**: One common method to obtain dense vectors from a **term-document matrix** is to apply **SVD**. This decomposes the matrix into **lower-rank** matrices, producing dense vectors for each word.

#### **2. Prediction-based Methods**

- These methods involve training a **binary classifier** to predict whether one word is likely to appear in the **context** of another word. The parameters learned by the classifier are used as word embeddings.
- This method can be trained on large amounts of **raw text** in a **self-supervised manner**, making it scalable for modern NLP tasks.

---

<br>

### 3. Word2Vec

The most popular prediction-based method for learning word embeddings is **Word2Vec**, introduced by **Mikolov et al.** in 2013. There are two main variants of Word2Vec:

#### **1. Skip-gram Model**

- The **skip-gram model** trains a classifier on a binary task: given a **target word**, predict whether a **context word** is likely to appear near it.
  
##### **How Skip-gram Works**

- **Target word**: The word for which we are predicting contexts.
- **Context words**: The words that appear in a **window** around the target word in a sentence.
  
For example, in the sentence:

- "The lemon, a [tablespoon of apricot jam, a] pinch..."

Here, "apricot" is the **target word**, and words like "tablespoon," "of," "jam" are the **context words**.

#### Procedure for Skip-Gram Model

- Treat the target word and a neighboring context word as
positive examples.
- Randomly sample other words in the lexicon to get negative
examples.
- Use logistic regression to train a classifier to distinguish those
two cases.
- Use the learned weights as the embeddings.

The goal of the skip-gram model is to learn embeddings that maximize the probability of **positive examples** (real context words like "tablespoon" and "jam") and minimize the probability of **negative examples** (randomly sampled non-context words like "spaceship" or "NLP").

#### **2. Continuous Bag of Words (CBOW)**

- The **CBOW model** is the inverse of skip-gram. Instead of predicting context words from a target word, it predicts the **target word** from its surrounding **context**.
- In CBOW, you aggregate all the words in the context window and try to predict the center word.

---

<br>

### 4. Training Word Embeddings

#### **1. Skip-gram Training Data**

- Consider a **±2 window** for the word "apricot" in the sentence:
  
  "…lemon, a [tablespoon of apricot jam, a] pinch…"

- **Positive examples** (real context words) could be:
  - ("apricot", "tablespoon")
  - ("apricot", "of")
  - ("apricot", "jam")

- **Negative examples** (non-context words) are randomly sampled from the vocabulary, based on their frequency.

#### **2. Negative Sampling**

- To train the skip-gram model efficiently, we use **negative sampling**. Instead of predicting the entire vocabulary (which is computationally expensive), we randomly sample a few words that do **not** co-occur with the target word.
- These **negative samples** are used to train the classifier to distinguish between real contexts and random words.

#### **3. Skip-gram Classifier**

The **skip-gram classifier** is trained using **logistic regression** to predict whether a context word belongs to the context of the target word. The probability is modeled using the **sigmoid function**:

$$P( + | w, c ) = \sigma( w \cdot c ) = \frac{1}{1 + e^{-w \cdot c}}$$

and

$$P( - | w, c) = 1 - P( + | w, c)$$

where:

- $w$ is the vector for the **target word**.
- $c$ is the vector for the **context word**.

The goal is to find a model that **maximizes** the similarity of the target
word with the actual context words, and **minimize** the similarity of
the target with the $k$ negative sampled non-neighbor words.

---

<br>

### 5. Learning via Gradient Descent

To train the model, we use **stochastic gradient descent** (SGD) to update the parameters:

- **Weights ($W$)**: The vectors representing the target words.
- **Context vectors ($C$)**: The vectors representing context words.

During training, the model performs the following updates:

1. **Maximizing** the similarity between target words and real context words (positive samples).
2. **Minimizing** the similarity between target words and negative samples.

The objective function can be expressed as:

$$L = \sum_{(w, c)} [\log P( + | w, c ) + \sum_{i=1}^k \log P( - | w, c_{neg_i} )]$$

Where:

- $(w, c)$ are the target and context word pairs.
- $c_{neg_i}$ are the negative samples.

At each step, the gradient of the **loss function** is computed, and the parameters (word vectors) are updated accordingly.

---

<br>

### 6. Embedding Matrices

After training, the skip-gram model produces two matrices of embeddings:

- **Target embeddings matrix ($W$)**: This matrix contains vectors representing each word as the **target** word.
- **Context embeddings matrix ($C$)**: This matrix contains vectors representing each word as a **context** word.

Typically, the final word embedding for a word is obtained by **adding** its vector from the **target** matrix and the **context** matrix, representing the $i^{th}$ word as the vector $W_i + C_i$.

<br>

#### Summary

- For a vocabulary of size V: Start with V random vectors (typically 300-dimensional) as initial embeddings.
- Train a logistic regression classifier to distinguish words that co-occur in the corpus from those that don’t.
  - Pairs of words that co-occur are positive examples
  - Pairs of words that don't co-occur are negative examples
- During training, target and context vectors of positive examples will become similar, and those of negative examples will become dissimilar.
- Keep two embedding matrices $W$ and $C$, where each word in the vocabulary is mapped to a 300D vector.

---

<br>

### 7. Evaluating Word Embeddings

There are two primary ways to evaluate the quality of word embeddings:

#### 1. Extrinsic Evaluation

- **Downstream tasks**: You can evaluate the embeddings by using them as features for other (downstream) tasks, such as **text classification** or **clustering**.
  - For instance, embeddings can replace hand-engineered features in tasks like **sentiment analysis** or **question answering**.

#### 2. Intrinsic Evaluation

- **Word similarity**: Compare the similarity between word pairs using the learned embeddings and compare them to human judgments. Datasets like **WordSim-353** and **SimLex-999** provide word pairs along with human-rated similarity scores.
  
- **Word analogy**: The word analogy task tests the embeddings' ability to capture **semantic relationships** between words. A famous example is:
  - **"man is to woman as king is to ___"**.
  
  The model should infer that the answer is "queen" by solving:

  $$v_{\text{man}} - v_{\text{woman}} \approx v_{\text{king}} - v_{\text{queen}}$$

Other examples include:

- **"Paris is to France as Rome is to Italy."**
- **"Run is to running as swim is to swimming."**

---

<br>

### 8. Word Embeddings and Historical Change

Word embeddings can also be used to study how word meanings evolve over time. By training embeddings on **older** vs. **modern texts**, we can observe how the meaning of a word has changed.

For example, the word "gay" once meant "happy," but its modern usage has shifted to refer to **sexual orientation**. By comparing embeddings learned from older texts to those from modern corpora, we can visualize these shifts in meaning.

---

<br>

### 9. Visualizing Word Embeddings

Word embeddings are typically **300-dimensional** vectors, which are hard to visualize directly. However, we can project them into **2D space** using techniques like **PCA (Principal Component Analysis)** or **t-SNE** to visualize the relationships between words.

For example, a 2D plot of embeddings might show clusters of related words (e.g., fruits like "apple," "banana," and "orange" clustering together).

---

<br>

### 10. Resources for Pretrained Word Embeddings

There are several popular resources for **pre-trained word embeddings**, which can be used directly in NLP applications:

#### **1. Word2Vec**

- **Source**: [Google's Word2Vec](https://code.google.com/archive/p/word2vec/)
- **Details**: 300-dimensional vectors trained on 100 billion tokens from the **Google News** dataset.

#### **2. GloVe (Global Vectors for Word Representation)**

- **Source**: [Stanford's GloVe](https://nlp.stanford.edu/projects/glove/)
- **Details**: 300-dimensional vectors trained on various large corpora, such as **Wikipedia**, **Common Crawl**, and **Twitter** (up to 840 billion tokens).

---

<br>

### Summary - Word Embeddings

- **Word embeddings** are dense vector representations of words that capture semantic similarity based on the distribution of words in large corpora.
- The **skip-gram model** in **Word2Vec** learns embeddings by predicting context words from target words, optimizing for similar words to be closer in vector space.
- Word embeddings can be evaluated through **intrinsic** (e.g., word similarity, analogy tasks) or **extrinsic** (e.g., downstream NLP tasks) methods.
- Pre-trained embeddings like **Word2Vec** and **GloVe** are widely used in NLP and can be fine-tuned for specific tasks.

---

<br>

## XII - Neural Networks for NLP

Neural networks have revolutionized many areas of NLP over the past decade due to their ability to model complex patterns in data, including language.

---

<br>

### 1. Neural Networks: Overview

A **neural network** is composed of multiple layers of **neurons** or **units**, each of which performs simple computations and passes the result to the next layer. The power of neural networks comes from their ability to learn non-linear representations of input data through **hidden layers**.

#### **Basic Structure**

- **Input Layer**: Receives the raw data.
- **Hidden Layers**: Perform non-linear transformations of the data.
- **Output Layer**: Produces the final prediction.

Each **neural unit** in a layer computes a **weighted sum** of its inputs, adds a **bias**, and applies a **non-linear activation function**.

$$z = w \cdot x + b = (\sum_{i} w_ix_i) + b$$

where:

- $w$ are the weights,
- $x$ are the input values,
- $b$ is the bias term.

<br>

Instead of generating $z$ directly, another non-linear function $f$ is often applied to $z$ before output, which is called the **activation function**:

$$y = f(z)$$

This is what allows the network to model non-linear patterns in data.

---

<br>

### 2. Non-Linear Activation Functions

Neural networks rely on **non-linear activation functions** to introduce complexity into the model. Common activation functions include:

- **Sigmoid**: Often used for binary classification tasks.
  
  $$\sigma(z) = \frac{1}{1 + e^{-z}}$$

    which for a neural unit becomes:

    $$\sigma(z) = \frac{1}{1 + e^{-(w \cdot x + b)}}$$

    **Derivative:**

    $$f'(z) = f(z) \times (1-f(z))$$

- **Tanh**: Maps input values to the range [-1, 1], useful for normalized data.
  
  $$\text{tanh}(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

    **Derivative:**

    $$f'(z) = 1-f(z)^2$$

- **ReLU (Rectified Linear Unit)**: The most commonly used activation function in modern deep learning. It has the form:
  
  $$\text{ReLU}(z) = \max(0, z)$$

  **Derivative:**

    $$
    f'(z) =
    \begin{cases}
    1 & \text{if } z > 0 \\
    0 & \text{if } z < 0
    \end{cases}$$
  
  ReLU is computationally efficient and helps prevent the **vanishing gradient problem**.

---

<br>

### 3. Perceptrons

A **Perceptron** is the simplest type of neural unit. It outputs either a 0 or 1, depending on whether the weighted sum of its inputs is greater than a threshold:

$$
y =
\begin{cases}
1 & \text{if } w \cdot x + b > 0 \\
0 & \text{otherwise}
\end{cases}$$

Perceptrons can be used to compute simple logical functions like **AND** and **OR**.

However, perceptrons are **linear classifiers**, meaning that they cannot separate data that is not linearly separable (such as **XOR**). This limitation led to the development of **multi-layer networks**.

Perceptron equation, given $x_1$ and $x_2$, is the equation of a line:

$$w_1x_1 + w_2x_2 + b = 0$$

i.e.,

$$x_2 = - \frac{w_1}{w_2}x_1 - \frac{b}{w_2}$$

This line acts as an imaginary boundary:

- 0 if input is on one side of the line.
- 1 if on the other side.

---

<br>

### 4. XOR Problem

The XOR problem demonstrates the limitations of single-layer perceptrons. XOR is not linearly separable, meaning you cannot draw a straight line that separates the true (1) and false (0) cases.

If we create a neural network with a layer of hidden units and the ReLU activation function, then you can represent XOR.

$$\text{ReLU}: f(z) = \max(0, z)$$

---

<br>

### 5. Multi-Layer Neural Networks

By adding multiple layers of hidden units, a neural network becomes capable of solving problems that a single perceptron cannot (e.g., XOR). These are called **multi-layer perceptrons (MLPs)** or **feedforward neural networks**.

- **Hidden Layers**: Transform the input data non-linearly before passing it to the output layer.
- **Fully-Connected Layers**: Every unit in one layer is connected to every unit in the next layer. This is the typical setup for traditional feedforward neural networks.

The power of multi-layer networks comes from their ability to **learn hierarchical representations** of data.

---

<br>

### 6. Backpropagation and Gradient Descent

The most common method for training neural networks is **backpropagation**, combined with **gradient descent**.

#### Steps:
1. **Forward Propagation**: Input data is passed through the network to compute the output.
2. **Loss Calculation**: The difference between the predicted output and the actual target is computed using a loss function.
3. **Backpropagation**: The error is propagated backward through the network, and the gradients of the weights with respect to the loss are computed.
4. **Weight Update**: Using gradient descent, the weights are updated in the direction that minimizes the loss.

---

<br>

### 7. Activation Functions: Sigmoid, Tanh, and ReLU

- **Sigmoid**: Often used in the output layer for binary classification. However, it suffers from the **vanishing gradient problem** for large input values. [Formula for the Sigmoid Function](#2-non-linear-activation-functions)
  
- **Tanh**: Similar to the sigmoid function but centered around zero, making it preferable for certain tasks. [Formula for the Tanh Function](#2-non-linear-activation-functions)

- **ReLU**: The default activation function for most hidden layers. ReLU is computationally efficient and does not suffer as much from vanishing gradients as the sigmoid or tanh functions. However, ReLU has its own problem: **dead neurons**, where neurons stop learning after receiving negative input. [Formula for the ReLu Activation Function](#2-non-linear-activation-functions)

---

<br>

### 8. Softmax for Multiclass Classification

The **softmax function** is used in the output layer when dealing with multiclass classification tasks. It converts the raw output of the network into a **probability distribution** over multiple classes.

The softmax function can be applied to transform the real-valued output vector into a probability distribution by normalizing the values across the output vector.

For a vector $z$ with $d$ dimensions, the formula is:

$$\text{softmax}_i(z) = \frac{e^{z_i}}{\sum_{j=1}^d e^{z_j}}$$

where $1 ≤ i ≤ d$

This ensures that the output values are positive and sum to 1, making them interpretable as probabilities.

---

<br>

### 9. Feedforward Neural Networks for NLP

Feedforward neural networks, also known as **multi-layer perceptrons (MLPs)**, have been successfully applied to various NLP tasks such as **text classification** and **language modeling**.

- Feedforward neural nets have at least one layer of hidden units and no cycles.
- Each layer is usually fully-connected (FC) to the next layer. We often talk about the weights as a matrix $W$, where the value for the hidden units are:

$$h = f(Wx)$$

<br>

Examples of Feedforward Neural Networks:

- **Text Classification**: Input features (e.g., bag of words or word embeddings) are passed through the network, and the output is a probability distribution over possible classes (e.g., sentiment labels: positive/negative).
  
- **Language Modeling**: Neural language models predict the probability of the next word in a sequence given the previous words. Traditional **n-gram models** are replaced with **neural n-gram models** that use neural networks to estimate these probabilities.

---

<br>

### 10. Word Embeddings as Features

Instead of relying on **hand-crafted features** like in traditional machine learning models, neural networks in NLP can use **word embeddings**. Word embeddings are dense vector representations of words that capture semantic relationships between words.

- **Pre-trained Word Embeddings**: These embeddings can be learned from large corpora and used as features for various tasks, improving model performance by capturing more generalizable features.

---

<br>

### 11. Neural Language Models

Neural language models improve upon traditional n-gram models by using neural networks to estimate the probability of a word given its context.

- **Input Layer**: Consists of the word vectors (e.g., embeddings) for the previous words.
- **Hidden Layer**: Uses a non-linear transformation to learn relationships between the words.
- **Output Layer**: Uses a softmax function to predict the next word.

Neural models can generalize better to unseen words or contexts by using word embeddings that capture semantic similarities between words.

---

<br>

## XIII - Recurrent Neural Networks (RNNs)

**Recurrent Neural Networks (RNNs)** are a powerful type of neural network designed to handle **sequential data**. RNNs are especially useful in tasks where the input or output involves sequences, such as **part-of-speech tagging**, **named entity recognition (NER)**, **language modeling**, and **machine translation**.

---

<br>

### 1. Introduction to Sequence Labeling with Neural Networks

Sequence labeling tasks, such as **part-of-speech tagging** and **NER**, require models to process variable-length inputs and predict corresponding output sequences. Traditional **feedforward neural networks** are not suited to these tasks, as they only handle fixed-size inputs and outputs.

RNNs address this issue by processing **variable-length sequences**. This makes RNNs a crucial tool for many NLP applications where text of varying lengths needs to be modeled.

---

<br>

### 2. Recurrent Neural Networks (RNNs)

**Recurrent Neural Networks (RNNs)** are a class of neural networks that are particularly suited to sequence data. Unlike feedforward networks, RNNs can handle inputs of variable length by **recurring** over time steps. The key idea behind RNNs is that the **hidden state** at time step $t$ is influenced not only by the current input but also by the hidden state at the previous time step $t-1$.

#### **How RNNs Work:**
- **Recurrence Relation**: At each time step $t$, the hidden state $h^{(t)}$ is computed as a function of both the current input $x^{(t)}$ and the hidden state from the previous time step $h^{(t-1)}$:
  
  $$h(t) = g(Uh^{(t-1)} + Wx^{(t)})$$

  Where:
  - $U$ is the weight matrix applied to the previous hidden state.
  - $W$ is the weight matrix applied to the current input.
  - $g$ is a non-linear activation function (typically **tanh** or **ReLU**).

---

<br>

### 3. Illustration of Recurrent Process

Each time step in an RNN corresponds to a small feedforward network. The hidden state at each time step $t$ takes two inputs:
1. The hidden state from the previous time step $h^{(t-1)}$.
2. The current input $x^{(t)}$.

This means that RNNs can **maintain memory** of previous inputs, making them ideal for tasks where context matters, such as language modeling or translation.

#### Example:
For a sentence like "The door opened," the RNN processes each word sequentially, updating the hidden state at each time step based on the previous word and the current word.

---

<br>

### 4. RNN vs. Feedforward Neural Networks

- RNNs differ from feedforward neural networks in that they can handle inputs of **any length**, while feedforward networks have a fixed input size.
- RNNs can also capture **dependencies** between different time steps in a sequence, which is crucial for tasks like language modeling, where the prediction of a word depends on previous words in the sentence.

---

<br>

### 5. RNNs for Language Modeling

RNNs are widely used in **language modeling**, where the goal is to predict the next word in a sequence given the previous words. The hidden state at each time step allows the model to retain information about the context, leading to better predictions for the next word.

#### Example:
For the sentence "The cat jumps," the RNN would predict the probability of the next word (e.g., "over") based on the hidden state that encodes information from the previous words.

---

<br>

### 6. Improvements Over Feedforward Neural Networks

Compared to traditional **n-gram language models**, **feedforward neural networks** offer several improvements:
- **No sparsity problem**: RNNs do not suffer from the sparsity issues that plague n-gram models because they can learn distributed representations of words (e.g., **word embeddings**).
- **Ability to handle unseen words**: By using pre-trained embeddings, RNNs can generalize to unseen words.
- **Better generalization**: RNNs can capture long-range dependencies and do not require a fixed-size context window.

<br>

Compared to **feedforward neural networks**, **RNNs** offer several improvements:
- Can take any length of input.
- Capture longer dependence.
- Model size will not increase as the sequence length increases.

However, vanilla RNNs still have limitations, such as the **vanishing gradient problem**.

---

<br>

### 7. Training an RNN Language Model

RNN language models are trained using **self-supervision**, where the model is trained to predict the next word at each time step in a sequence. The training objective is to minimize the **cross-entropy loss** between the predicted word distribution and the true next word.

#### Loss Function:
The loss function for an RNN language model is the **cross-entropy** between the predicted distribution $\hat{y}(t)$ and the actual next word $y(t)$:

$$L_{\text{CE}} = - \sum_{w \in V} y^{(t)}[w] \log \hat{y}^{(t)}[w]$$

where:
- $V$ is the vocabulary.
- $\hat{y}^{(t)}$ is the predicted probability distribution for the next word.

<br>

The total loss over a sequence is the average loss across all time steps:

$$L(\theta) = - \frac{1}{T} \sum_{t=1}^{T} \log \hat{y}^{(t)}[w_{t+1}]$$

---

<br>

### 8. Weight Tying in RNNs

To reduce the number of parameters in an RNN, **weight tying** can be used. In this approach, the weights used in the embedding layer and the output layer are shared. This reduces the overall model size and improves generalization.

---

<br>

### 9. RNNs for Other NLP Tasks

#### Sequence Labeling

Inputs are word embeddings and outputs are tag probabilities from a softmax function.

#### Text Classification

We take the hidden layer for the last token of the text, to pass it to another feedforward network to choose a class.

#### Language Generation

- Text generation, together with image/code/video generation, nowadays often called “generative AI”.
- Using a language model to incrementally generate the next word
based on previous choices is called **autoregressive generation**.

---

<br>

### 10. Stacked and Bidirectional RNNs

#### **Stacked RNNs**:
RNNs can be made **deep** by stacking multiple RNN layers on top of each other, like feedforward nets. In practice, using 2 to 4 stacked layers outperforms using a single RNN layer, as different layers can capture different levels of abstraction.

#### **Bidirectional RNNs**:
**Bidirectional RNNs** (BiRNNs) are a variation of RNNs where two RNNs are run in parallel: one processes the sequence from left to right, and the other from the end to the start. The hidden states from both RNNs are concatenated, allowing the model to use both **past** and **future** context to make predictions.

---

<br>

### 11. RNN Variants: LSTMs and GRUs

#### **Long Short-Term Memory Networks (LSTMs)**:
**LSTMs** are a type of RNN designed to address the **vanishing gradient problem**, which occurs in standard RNNs. LSTMs introduce a **memory cell** and **gates** to control the flow of information:
- **Forget Gate**: Decides which parts of the memory to forget.
- **Input Gate**: Decides which parts of the input to add to the memory.
- **Output Gate**: Decides how much of the memory to use in the hidden state.

By using gates, LSTMs can retain information over long time steps, making them more suitable for tasks that require long-term dependencies.

#### **Gated Recurrent Units (GRUs)**:
**GRUs** are a simplified version of LSTMs that combine the forget and input gates into a single **update gate**. GRUs perform similarly to LSTMs but are computationally more efficient because they have fewer parameters.

---

<br>

### 12. Vanishing and Exploding Gradients

#### **Vanishing Gradients**:
In simple RNNs, the gradient of the loss with respect to the earlier layers becomes very small as it propagates backward through the time steps. This makes it difficult for the network to learn long-term dependencies.

For example, with $g = \tanh$, models suffer from the **vanishing gradient** problem: the gradients are eventually driven to zero as sequence gets longer.

#### **Exploding Gradients**:
In some cases, the gradients become too large during backpropagation, causing the network’s weights to explode, leading to unstable training.

For example, with $g = ReLu$, models suffer from the **exploding gradient** problem: the gradients are becoming too large.

#### **Gradient Clipping**:
One solution to the exploding gradient problem is **gradient clipping**, where the gradient is scaled down if it exceeds a certain threshold.

---

<br>

### 13. LSTMs

LSTMs are designed to preserve information over many time steps, addressing the vanishing gradient problem. By using **gates** to control the flow of information, LSTMs can keep information in memory for long periods, enabling them to capture dependencies that are hundreds of time steps long.

LSTMs contain an additional **memory cell state** that also gets passed through the network and updated at each time step.

**Key idea:** turning multiplication into addition and using “gates” to control how much information to add/erase.

In practice, LSTMs perform well over sequences with around 100 time steps, compared to vanilla RNNs, which struggle with sequences longer than 7 time steps.

---

### 14. How LSTM Solves Vanishing Gradients

- LSTMs make it much easier as compared to RNNs to preserve information over many time steps.
- For example, if the forget gate is set to 1 for a cell dimension and the input gate set to 0, then that cell is preserved indefinitely.
- However, LSTM doesn’t guarantee that there is no vanishing/exploding gradient.

---

<br>

## XIV - Machine Translation and Seq2Seq Models

### 1. Machine Translation (MT)

**Machine Translation (MT)** is the task of automatically translating text from one language (the **source language**) into another (the **target language**). High-quality MT can greatly enhance global communication, but it remains a challenging problem.

#### Why is MT Challenging?
MT is difficult because it requires a deep understanding of the source language and the ability to generate grammatically correct and semantically accurate text in the target language. Some of the key challenges include:
- **Word Order**: Different languages have different syntactic structures.
- **Word Sense Ambiguity**: Many words have multiple meanings depending on the context.
- **Idioms and Phrases**: Phrases like “kick the bucket” are difficult to translate literally.

#### Early Failures in MT:
MT research faced several setbacks in the early stages, primarily due to the complexity of language and limitations in computational models. One notable failure was the **ALPAC Report** (1966), which concluded that human translation was far superior to MT, leading to a decline in MT research for many years.

---

<br>

### 2. The Vauquois Triangle

The **Vauquois Triangle** is a model that illustrates different levels of abstraction used in MT systems. The three main approaches to MT are:

1. **Direct Translation**:
   - Translates the source language directly into the target language without an intermediate representation.
   - Typically involves **word-for-word translation** using a bilingual dictionary.
   - **Drawback**: This approach struggles with word sense disambiguation, idiomatic expressions, and differing word orders between languages.

2. **Transfer-Based Translation**:
   - Uses an intermediate representation at the syntactic or semantic level to transfer meaning from the source language to the target language.
   - This approach relies on language-specific rules to convert between languages.

3. **Interlingua**:
   - Uses a **language-independent meaning representation** (interlingua) as the intermediate stage. The interlingua is used to generate the target language output.
   - **Advantage**: Allows for multilingual MT, where the same interlingua can be used to translate between any pair of languages.

---

<br>

### 3. Rule-Based Machine Translation (RBMT)

**Rule-Based Machine Translation (RBMT)** systems use handcrafted linguistic rules and bilingual dictionaries to perform translation. While these systems were once popular, they suffer from several limitations:
- **Complexity**: RBMT systems require significant human effort to develop and maintain.
- **Domain Specificity**: Rules need to be created for each language pair, making it difficult to scale.
- **Inflexibility**: Rule-based systems cannot handle the variability and ambiguity present in natural language as effectively as statistical or neural approaches.

---

<br>

### 4. Statistical Machine Translation (SMT)

**Statistical Machine Translation (SMT)**, popular in the 1990s and 2000s, treats translation as a probabilistic problem. SMT models rely on statistical methods to translate text based on large amounts of bilingual data (parallel corpora). SMT is built on **Bayes' rule**:

$$P(T | S) = \arg \max_T P(S | T) P(T)$$

Where:
- $P(S | T)$ is the **translation model**, capturing how well a sentence in the source language $S$ can be translated into a target language sentence $T$.
- $P(T)$ is the **language model**, which captures how likely the target sentence $T$ is grammatically and fluently correct.

#### Parallel Corpora:
**Parallel corpora** are essential for training SMT systems. These are collections of source language texts aligned with their corresponding translations in the target language.

Some widely used corpora include:
- **The Hansard Corpus**: Canadian parliamentary proceedings available in both French and English.
- **European Union Corpora**: Contain multiple translations for European languages.

---

<br>

### 5. Evaluation of Machine Translation

Evaluating MT systems is crucial to measure how well they perform. There are two main criteria for evaluation:
1. **Adequacy**: The translation should preserve the meaning of the original sentence.
2. **Fluency**: The translation should be grammatically and syntactically correct in the target language.

#### Human Evaluation:
- **Pros**: Provides the most accurate assessment.
- **Cons**: Expensive, time-consuming, and difficult to scale.

#### Automatic Evaluation:
- **Pros**: Inexpensive and scalable.
- **Cons**: May not fully capture the quality of the translation as well as human evaluators.

One popular automatic metric is **BLEU (Bilingual Evaluation Understudy)**, which compares a machine-generated translation to one or more human reference translations based on **n-gram precision**.

---

<br>

### 6. BLEU Score

The **BLEU score** is calculated by comparing the n-grams (sequences of n words) in the machine-generated translation to those in human reference translations. ***The higher the overlap of n-grams, the higher the BLEU score.*** BLEU also includes a **brevity penalty** to prevent translations that are too short.

$$\text{BLEU} = \text{geometric mean of } p_n \times \text{brevity penalty}$$

Where $p_n$ is the precision for n-grams, typically from $n = 1 \dots 4$, that is up to $n = 4$.

---

<br>

### 7. Neural Machine Translation (NMT)

**Neural Machine Translation (NMT)** has become the dominant approach in recent years due to its superior performance compared to SMT. NMT models are based on neural networks and use **sequence-to-sequence (Seq2Seq)** architectures or **end-to-end neural nets**.

#### Seq2Seq Architecture:
- **Encoder**: Processes the input (source language sentence) and encodes it into a fixed-length vector representation.
- **Decoder**: Takes the encoded representation and generates the corresponding translation in the target language.

#### Advantages of NMT over SMT:
- **End-to-End Training**: NMT systems are trained as a single neural network without requiring separate components like language models or translation models.
- **Better Use of Context**: NMT models can capture long-range dependencies and semantic information in the input sentence.

#### Attention Mechanism:
The attention mechanism was introduced to overcome the limitations of fixed-length encoding in Seq2Seq models. It allows the decoder to **focus on different parts of the input sequence** when generating each word in the translation, improving performance, especially for longer sentences.

---

<br>

### 8. Training Seq2Seq Models

Seq2Seq models are trained to minimize the **negative log-likelihood** of the correct translation. Given a source sentence, the model predicts the next word in the target sentence, and the loss is computed based on the probability of the correct word.

$$L = - \sum_{t=1}^T \log P(y_t | y_{<t}, X)$$

where:
- $y_t$ is the correct target word at time step $t$,
- $X$ is the source sentence.

---

<br>

### 9. Decoding in Seq2Seq Models

Once trained, Seq2Seq models can be used to generate translations. There are two main methods for decoding:
1. **Greedy Decoding**: At each time step, the most probable word is selected. However, this can lead to suboptimal translations because it does not consider future words.

2. **Beam Search**: Keeps track of the top $k$ most probable sequences (beam size $k$) at each time step, exploring multiple translation paths. Beam search usually leads to better translations than greedy decoding.

---

<br>

### 10. Pros and Cons of NMT

#### Pros:
- **Better Contextual Understanding**: NMT models produce more fluent and accurate translations by better capturing the context of the source sentence.
- **End-to-End Training**: Unlike SMT, NMT doesn’t require separate components for different translation tasks.
- **Less Feature Engineering**: NMT models rely on learned representations (word embeddings) rather than handcrafted features.

#### Cons:
- **Lack of Interpretability**: NMT models are less interpretable than rule-based or statistical systems, making it harder to understand how the translation was generated.
- **Difficulty with Rare Words**: NMT models can struggle with rare or unseen words, especially in low-resource languages.

---

<br>

### 11. Beam Search Decoding

Beam search improves upon greedy decoding by maintaining a set of possible translations at each step. It keeps the top $k$ sequences (beam size $k$) and selects the sequence with the highest probability at the end.

#### Example:
For a beam size of 2, the algorithm explores two possible continuations of the translation at each step. This results in better translations because it considers multiple possibilities rather than committing to the highest-probability word at each step.

---

<br>
