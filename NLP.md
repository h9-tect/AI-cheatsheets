# Advanced Natural Language Processing (NLP) Cheatsheet

## Table of Contents

1. [Foundations of NLP](#foundations-of-nlp)
2. [Text Preprocessing](#text-preprocessing)
3. [Feature Extraction and Representation](#feature-extraction-and-representation)
4. [Classical NLP Models](#classical-nlp-models)
5. [Deep Learning for NLP](#deep-learning-for-nlp)
6. [Advanced NLP Architectures](#advanced-nlp-architectures)
7. [NLP Tasks and Techniques](#nlp-tasks-and-techniques)
8. [Evaluation Metrics for NLP](#evaluation-metrics-for-nlp)
9. [Deployment and Scalability](#deployment-and-scalability)
10. [Ethical Considerations in NLP](#ethical-considerations-in-nlp)
11. [Best Practices and Advanced Tips](#best-practices-and-advanced-tips)

## Foundations of NLP

### Linguistic Foundations
- **Morphology**: Study of word formation
  - Stemming algorithms: Porter, Snowball
  - Lemmatization: WordNet lemmatizer, spaCy's lemmatizer
- **Syntax**: Rules for sentence formation
  - Constituency parsing vs. Dependency parsing
  - Universal Dependencies framework
- **Semantics**: Meaning in language
  - WordNet for lexical semantics
  - Frame semantics and FrameNet
- **Pragmatics**: Context-dependent meaning
  - Speech act theory
  - Gricean maxims

### Statistical NLP
- **N-gram models**
  - Smoothing techniques: Laplace, Good-Turing, Kneser-Ney
  - Perplexity as an evaluation metric
- **Hidden Markov Models (HMMs)**
  - Viterbi algorithm for decoding
  - Baum-Welch algorithm for training
- **Maximum Entropy models**
  - Feature engineering for MaxEnt models
  - Comparison with logistic regression

Tip: Implement simple n-gram models from scratch to truly understand their workings before moving to more complex models.

## Text Preprocessing

### Tokenization
- **Rule-based tokenization**
  - Regular expressions for token boundary detection
  - Handling contractions and possessives
- **Statistical tokenization**
  - Maximum Entropy Markov Models for tokenization
  - Unsupervised tokenization with Byte Pair Encoding (BPE)
- **Subword tokenization**
  - WordPiece: Used in BERT
  - SentencePiece: Language-agnostic tokenization
  - Unigram language model tokenization

Tip: Use SentencePiece for multilingual models to handle a variety of languages efficiently.

### Normalization
- **Case folding**: Considerations for proper nouns and acronyms
- **Stemming**:
  - Porter stemmer: Algorithmic approach
  - Snowball stemmer: Improved version of Porter
- **Lemmatization**:
  - WordNet lemmatizer: Uses lexical database
  - Morphological analysis-based lemmatization
- **Handling spelling variations and errors**
  - Edit distance algorithms: Levenshtein, Damerau-Levenshtein
  - Phonetic algorithms: Soundex, Metaphone

Tip: Consider using lemmatization over stemming for tasks where meaning preservation is crucial.

### Noise Removal
- **Regular expressions for text cleaning**
  - Removing HTML tags, URLs, and special characters
- **Handling Unicode and non-ASCII characters**
  - NFKC normalization for Unicode
- **Emoji and emoticon processing**
  - Emoji sentiment analysis
  - Converting emoticons to standard forms

Tip: Create a comprehensive text cleaning pipeline, but be cautious not to remove important information. Always validate your cleaning steps on a sample of your data.

## Feature Extraction and Representation

### Bag of Words (BoW) and TF-IDF
- **BoW implementations**
  - CountVectorizer in scikit-learn
  - Handling out-of-vocabulary words
- **TF-IDF variations**
  - Sublinear TF scaling
  - Okapi BM25 as an alternative to TF-IDF
- **N-grams and skip-grams**
  - Efficient storage of sparse matrices (CSR format)

Tip: Use feature hashing (HashingVectorizer in scikit-learn) for memory-efficient feature extraction on large datasets.

### Word Embeddings
- **Word2Vec**
  - Continuous Bag of Words (CBOW) vs. Skip-gram
  - Negative sampling and hierarchical softmax
- **GloVe (Global Vectors)**
  - Co-occurrence matrix factorization
  - Comparison with Word2Vec
- **FastText**
  - Subword embeddings for handling OOV words
  - Language-specific vs. multilingual embeddings

Tip: Train domain-specific embeddings if you have enough data. They often outperform general-purpose embeddings for domain-specific tasks.

### Contextualized Embeddings
- **ELMo (Embeddings from Language Models)**
  - Bidirectional LSTM architecture
  - Character-level CNN for token representation
- **BERT embeddings**
  - Strategies for extracting embeddings from BERT
  - Fine-tuning vs. feature extraction
- **Sentence-BERT**
  - Siamese and triplet network structures
  - Pooling strategies for sentence embeddings

Tip: For sentence-level tasks, consider using Sentence-BERT embeddings as they're optimized for semantic similarity tasks.

## Classical NLP Models

### Naive Bayes Classifier
- **Variants**: Multinomial, Bernoulli, Gaussian Naive Bayes
- **Handling the zero-frequency problem**
  - Laplace smoothing
  - Lidstone smoothing
- **Feature selection for Naive Bayes**
  - Mutual Information
  - Chi-squared test

Tip: Use Multinomial Naive Bayes for text classification tasks. It often provides a strong baseline with minimal computational cost.

### Support Vector Machines (SVM) for Text Classification
- **Kernel tricks**
  - Linear kernel for high-dimensional sparse data
  - RBF kernel for lower-dimensional dense representations
- **Multi-class classification strategies**
  - One-vs-Rest: Trains N classifiers for N classes
  - One-vs-One: Trains N(N-1)/2 classifiers
- **Handling imbalanced datasets**
  - Adjusting class weights
  - SMOTE for oversampling

Tip: Start with a linear SVM for text classification. It's often sufficient and much faster to train than kernel SVMs for high-dimensional text data.

### Conditional Random Fields (CRF)
- **Feature templates for CRFs**
  - Current word, surrounding words, POS tags, etc.
- **Training algorithms**
  - L-BFGS for batch learning
  - Stochastic Gradient Descent for online learning
- **Structured prediction with CRFs**
  - Viterbi algorithm for inference
  - Constrained conditional likelihood for semi-supervised learning

Tip: Use sklearn-crfsuite for an easy-to-use implementation of CRFs in Python. Combine CRFs with neural networks for state-of-the-art sequence labeling.

## Deep Learning for NLP

### Recurrent Neural Networks (RNNs)
- **LSTM architecture details**
  - Input, forget, and output gates
  - Peephole connections
- **GRU (Gated Recurrent Unit)**
  - Comparison with LSTM: fewer parameters, often similar performance
- **Bidirectional RNNs**
  - Combining forward and backward hidden states
- **Attention mechanisms in RNNs**
  - Bahdanau attention vs. Luong attention
  - Multi-head attention in RNN context

Tip: Use gradient clipping to prevent exploding gradients in RNNs. Consider using GRUs instead of LSTMs for faster training on smaller datasets.

### Convolutional Neural Networks (CNNs) for NLP
- **1D convolutions for text**
  - Kernel sizes and their impact
  - Dilated convolutions for capturing longer-range dependencies
- **Character-level CNNs**
  - Embedding layer for characters
  - Max-pooling strategies
- **CNN-RNN hybrid models**
  - CNN for feature extraction, RNN for sequence modeling

Tip: CNNs can be very effective for text classification tasks, especially when combined with pre-trained word embeddings. They're often faster to train than RNNs.

### Seq2Seq Models
- **Encoder-Decoder architecture**
  - Handling variable-length inputs and outputs
  - Teacher forcing: benefits and drawbacks
- **Attention mechanisms in Seq2Seq**
  - Global vs. local attention
  - Monotonic attention for tasks like speech recognition
- **Beam search decoding**
  - Beam width trade-offs
  - Length normalization in beam search

Tip: Implement scheduled sampling to bridge the gap between training and inference in seq2seq models. This can help mitigate exposure bias.

## Advanced NLP Architectures

### Transformer Architecture
- **Self-attention mechanism**
  - Scaled dot-product attention
  - Multi-head attention: parallel attention heads
- **Positional encodings**
  - Sinusoidal position embeddings
  - Learned position embeddings
- **Layer normalization and residual connections**
  - Pre-norm vs. post-norm configurations
  - Impact on training stability
- **Transformer-XL: segment-level recurrence**
  - Relative positional encodings
  - State reuse for handling long sequences

Tip: When fine-tuning Transformers, try using a lower learning rate for bottom layers and higher for top layers. This "discriminative fine-tuning" can lead to better performance.

### BERT and its Variants
- **Pre-training objectives**
  - Masked Language Model (MLM)
  - Next Sentence Prediction (NSP)
- **WordPiece tokenization**
  - Handling subwords and rare words
- **Fine-tuning strategies**
  - Task-specific heads
  - Gradual unfreezing
- **Variants and improvements**
  - RoBERTa: Robustly optimized BERT approach
  - ALBERT: Parameter reduction techniques
  - DistilBERT: Knowledge distillation for smaller models

Tip: When fine-tuning BERT, start with a small learning rate (e.g., 2e-5) and use a linear learning rate decay. Monitor validation performance for early stopping.

### GPT Series
- **Autoregressive language modeling**
  - Causal self-attention
  - Byte-Pair Encoding for tokenization
- **GPT-2 and GPT-3**
  - Scaling laws in language models
  - Few-shot learning capabilities
- **InstructGPT and ChatGPT**
  - Reinforcement Learning from Human Feedback (RLHF)
  - Constitutional AI principles

Tip: For text generation tasks, experiment with temperature and top-k/top-p sampling to control the trade-off between creativity and coherence.

### T5 (Text-to-Text Transfer Transformer)
- **Unified text-to-text framework**
  - Consistent input-output format for all tasks
- **Span corruption pre-training**
  - Comparison with BERT's masked language modeling
- **Encoder-decoder vs. decoder-only models**
  - Trade-offs in performance and computational efficiency

Tip: When using T5, leverage its ability to frame any NLP task as text-to-text. This allows for creative problem formulations and multi-task learning setups.

## NLP Tasks and Techniques

### Text Classification
- **Multi-class and multi-label classification**
  - One-vs-Rest vs. Softmax for multi-class
  - Binary Relevance vs. Classifier Chains for multi-label
- **Hierarchical classification**
  - Local Classifier per Node (LCN)
  - Global Classifier (GC)
- **Handling class imbalance**
  - Oversampling techniques: SMOTE, ADASYN
  - Class-balanced loss functions

Tip: For highly imbalanced datasets, consider using Focal Loss, which automatically down-weights the loss assigned to well-classified examples.

### Named Entity Recognition (NER)
- **Tagging schemes**
  - IOB, BIOES tagging
  - Nested NER handling
- **Feature engineering for NER**
  - Gazetteers and lexicon features
  - Word shape and orthographic features
- **Neural architectures for NER**
  - BiLSTM-CRF
  - BERT with token classification head

Tip: Incorporate domain-specific gazetteers to improve NER performance, especially for specialized entities.

### Sentiment Analysis
- **Aspect-based sentiment analysis**
  - Joint extraction of aspects and sentiments
  - Attention mechanisms for aspect-sentiment association
- **Handling negation and sarcasm**
  - Dependency parsing for negation scope detection
  - Contextual features for sarcasm detection
- **Cross-lingual sentiment analysis**
  - Translation-based approaches
  - Multilingual embeddings for zero-shot transfer

Tip: Use dependency parsing to capture long-range dependencies and negation scopes in sentiment analysis tasks.

### Machine Translation
- **Neural Machine Translation (NMT)**
  - Attention-based seq2seq models
  - Transformer-based models (e.g., mBART)
- **Multilingual NMT**
  - Language-agnostic encoders
  - Zero-shot and few-shot translation
- **Data augmentation techniques**
  - Back-translation
  - Paraphrasing for data augmentation

Tip: Implement Minimum Bayes Risk (MBR) decoding for improved translation quality, especially for high-stakes applications.

### Question Answering
- **Extractive QA**
  - Span prediction architectures
  - Handling unanswerable questions
- **Generative QA**
  - Seq2seq models for answer generation
  - Copying mechanisms for factual accuracy
- **Open-domain QA**
  - Retriever-reader architectures
  - Dense passage retrieval techniques

Tip: For open-domain QA, use a two-stage approach: first retrieve relevant documents, then extract or generate the answer. This can significantly improve efficiency and accuracy.

### Text Summarization
- **Extractive summarization**
  - Sentence scoring techniques
  - Graph-based methods (e.g., TextRank)
- **Abstractive summarization**
  - Pointer-generator networks
  - Bottom-up attention for long document summarization
- **Evaluation beyond ROUGE**
  - BERTScore for semantic similarity
  - Human evaluation protocols

Tip: Combine extractive and abstractive approaches for more faithful and coherent summaries. Use extractive methods to select salient content, then refine with abstractive techniques.

### Topic Modeling
- **Latent Dirichlet Allocation (LDA)**
  - Gibbs sampling vs. Variational inference
  - Selecting the number of topics
- **Neural topic models**
  - Autoencoder-based approaches
  - Contextualized topic models using BERT

Tip: Use coherence measures (e.g., C_v score) to evaluate topic model quality instead of relying solely on perplexity.

## Evaluation Metrics for NLP

### Classification Metrics
- **Beyond accuracy**
  - Matthews Correlation Coefficient for imbalanced datasets
  - Kappa statistic for inter-rater agreement
- **Threshold-independent metrics**
  - Area Under the ROC Curve (AUC-ROC)
  - Precision-Recall curves and Average Precision

Tip: For imbalanced datasets, prioritize metrics like F1-score, AUC-ROC, or Average Precision over simple accuracy.


### Sequence Labeling Metrics
- **Token-level vs. Span-level evaluation**
  - Exact match vs. partial match criteria
  - BIO tagging consistency checks
- **CoNLL evaluation script for NER**
  - Precision, Recall, and F1-score per entity type
  - Micro vs. Macro averaging
- **Boundary detection metrics**
  - Beginning/Inside/End/Single (BIES) tagging evaluation
  - SemEval ioBES scheme for fine-grained evaluation

Tip: Use span-based F1 score for tasks like NER, as it better reflects the model's ability to identify complete entities rather than just individual tokens.

### Machine Translation Metrics
- **BLEU (Bilingual Evaluation Understudy)**
  - N-gram precision and brevity penalty
  - Smoothing techniques for short sentences
- **METEOR (Metric for Evaluation of Translation with Explicit ORdering)**
  - Incorporation of stemming and synonym matching
  - Parameterized harmonic mean of precision and recall
- **chrF (Character n-gram F-score)**
  - Language-independent metric
  - Correlation with human judgments
- **Human evaluation techniques**
  - Direct Assessment (DA) protocol
  - Multidimensional Quality Metrics (MQM) framework

Tip: Use a combination of automatic metrics and targeted human evaluation. BLEU is widely reported but has limitations; consider using chrF or METEOR alongside it, especially for morphologically rich languages.

### Text Generation Metrics
- **Perplexity**
  - Relationship with cross-entropy loss
  - Domain-specific perplexity evaluation
- **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**
  - ROUGE-N, ROUGE-L, ROUGE-W variants
  - Limitations and considerations for abstractive tasks
- **BERTScore**
  - Token-level matching using contextual embeddings
  - Correlation with human judgments
- **MoverScore**
  - Earth Mover's Distance on contextualized embeddings
  - Handling of synonym and paraphrase evaluation

Tip: For creative text generation tasks, combine reference-based metrics (like ROUGE) with reference-free metrics (like perplexity) and human evaluation for a comprehensive assessment.

## Deployment and Scalability

### Model Compression Techniques
- **Knowledge Distillation for NLP models**
  - Temperature scaling in softmax
  - Distillation objectives: soft targets, intermediate representations
- **Quantization of NLP models**
  - Post-training quantization vs. Quantization-aware training
  - Mixed-precision techniques (e.g., bfloat16)
- **Pruning techniques for Transformers**
  - Magnitude-based pruning
  - Structured pruning: attention head pruning, layer pruning
- **Low-rank factorization**
  - SVD-based methods for weight matrices
  - Tensor decomposition techniques

Tip: Start with knowledge distillation for model compression, as it often provides a good balance between model size reduction and performance retention.

### Efficient Inference
- **ONNX Runtime for NLP models**
  - Graph optimizations for Transformer models
  - Quantization and operator fusion in ONNX
- **TensorRT for optimized inference**
  - INT8 calibration for NLP models
  - Dynamic shape handling for variable-length inputs
- **Caching and batching strategies**
  - KV-cache for autoregressive decoding
  - Dynamic batching for serving multiple requests
- **Sparse Inference**
  - Sparse attention mechanisms
  - Block-sparse operations for efficient computation

Tip: Implement an adaptive batching strategy that dynamically adjusts batch size based on current load to optimize throughput and latency trade-offs.

### Scalable NLP Pipelines
- **Distributed training**
  - Data parallelism vs. Model parallelism
  - Sharded data parallelism for large models
- **Efficient data loading and preprocessing**
  - Online tokenization and dynamic padding
  - Caching strategies for repeated epochs
- **Handling large text datasets**
  - Streaming datasets for out-of-memory training
  - Efficient storage formats (e.g., Apache Parquet for text data)
- **Scaling evaluation and inference**
  - Distributed evaluation strategies
  - Asynchronous pipeline parallelism for inference

Tip: Use techniques like gradient accumulation and gradient checkpointing to train large models on limited hardware. This allows you to effectively increase your batch size without increasing memory usage.

## Ethical Considerations in NLP

### Bias in NLP Models
- **Types of bias**
  - Selection bias in training data
  - Demographic biases in language models
  - Representation bias in word embeddings
- **Bias detection techniques**
  - Word Embedding Association Test (WEAT)
  - Sentence template-based bias probing
- **Bias mitigation strategies**
  - Data augmentation for underrepresented groups
  - Adversarial debiasing techniques
  - Counterfactual data augmentation

Tip: Regularly audit your models for various types of bias, not just during development but also after deployment, as biases can emerge over time with changing data distributions.

### Privacy Concerns
- **Differential privacy in NLP**
  - ε-differential privacy for text data
  - Federated learning with differential privacy guarantees
- **Anonymization techniques for text data**
  - Named entity recognition for identifying personal information
  - K-anonymity and t-closeness for text datasets
- **Secure multi-party computation for NLP**
  - Privacy-preserving sentiment analysis
  - Secure aggregation in federated learning

Tip: Implement a comprehensive data governance framework that includes regular privacy audits and clear policies on data retention and usage.

### Environmental Impact
- **Carbon footprint of large language models**
  - Estimating CO2 emissions from model training
  - Green AI practices and reporting standards
- **Efficient model design**
  - Neural architecture search for efficiency
  - Once-for-all networks: Train one, specialize many
- **Hardware considerations**
  - Energy-efficient GPU selection
  - Optimizing data center cooling for AI workloads

Tip: Consider using carbon-aware scheduling for large training jobs, running them during times when the electricity grid has a higher proportion of renewable energy.

## Best Practices and Advanced Tips

1. **Data Collection and Annotation**
   - Active learning strategies for efficient annotation
     - Uncertainty sampling
     - Diversity-based sampling
   - Inter-annotator agreement metrics
     - Cohen's Kappa for binary tasks
     - Fleiss' Kappa for multi-annotator scenarios
   - Annotation tools and platforms
     - Prodigy for rapid annotation
     - BRAT for complex annotation schemas

Tip: Implement a two-stage annotation process: rapid first pass followed by expert review of uncertain cases to balance speed and quality.

2. **Handling Low-Resource Languages**
   - Cross-lingual transfer learning techniques
     - mBERT and XLM-R for zero-shot transfer
     - Adapters for efficient fine-tuning
   - Data augmentation for low-resource settings
     - Back-translation and paraphrasing
     - Multilingual knowledge distillation
   - Few-shot learning approaches
     - Prototypical networks for NLP tasks
     - Meta-learning for quick adaptation

Tip: Leverage linguistic knowledge to create rule-based systems that can bootstrap your low-resource NLP pipeline before moving to data-driven approaches.

3. **Interpretability in NLP**
   - Attention visualization techniques
     - BertViz for Transformer attention patterns
     - Attention rollout and attention flow methods
   - LIME and SHAP for local interpretability
     - Text-specific LIME implementations
     - Kernel SHAP for consistent explanations
   - Probing tasks for model analysis
     - Edge probing for linguistic knowledge
     - Diagnostic classifiers for hidden representations

Tip: Combine multiple interpretability techniques for a holistic understanding. Attention visualizations can provide insights, but should be complemented with methods like SHAP for more reliable explanations.

4. **Handling Long Documents**
   - Hierarchical attention networks
     - Word-level and sentence-level attention mechanisms
     - Document-level representations for classification
   - Sliding window approaches
     - Overlap-tile strategy for long text processing
     - Aggregation techniques for window-level predictions
   - Efficient Transformer variants
     - Longformer: Sparse attention patterns
     - Big Bird: Global-local attention mechanisms
   - Memory-efficient fine-tuning
     - Gradient checkpointing
     - Mixed-precision training

Tip: For extremely long documents, consider a two-stage approach: use an efficient model to identify relevant sections, then apply a more sophisticated model to these sections for detailed analysis.

5. **Continual Learning in NLP**
   - Techniques to mitigate catastrophic forgetting
     - Elastic Weight Consolidation (EWC)
     - Gradient Episodic Memory (GEM)
   - Dynamic architectures
     - Progressive Neural Networks
     - Dynamically Expandable Networks
   - Replay-based methods
     - Generative replay using language models
     - Experience replay with importance sampling

Tip: Implement a task-specific output layer for each new task while sharing the majority of the network. This allows for task-specific fine-tuning without compromising performance on previous tasks.

6. **Domain Adaptation**
   - Unsupervised domain adaptation
     - Pivots and domain-invariant feature learning
     - Adversarial training for domain adaptation
   - Few-shot domain adaptation
     - Prototypical networks for quick adaptation
     - Meta-learning approaches (e.g., MAML)
   - Continual pre-training strategies
     - Adaptive pre-training: Continued pre-training on domain-specific data
     - Selective fine-tuning of model components

Tip: Create a domain-specific vocabulary and integrate it into your tokenizer. This can significantly improve performance on domain-specific tasks without requiring extensive retraining.

7. **Multimodal NLP**
   - Vision-and-Language models
     - CLIP: Contrastive Language-Image Pre-training
     - VisualBERT: Joint representation learning
   - Multimodal named entity recognition
     - Fusion strategies for text and image features
     - Attention mechanisms for cross-modal alignment
   - Multimodal machine translation
     - Incorporating visual context in translation
     - Multi-task learning for improved generalization

Tip: When dealing with multimodal data, pay special attention to synchronization and alignment between modalities. Misaligned data can significantly degrade model performance.

8. **Robustness and Adversarial NLP**
   - Adversarial training for NLP
     - Virtual adversarial training
     - Adversarial token perturbations
   - Certified robustness techniques
     - Interval bound propagation for Transformers
     - Randomized smoothing for text classification
   - Defending against specific attack types
     - TextFooler: Synonym-based attacks
     - BERT-Attack: Contextualized perturbations

Tip: Regularly evaluate your models against state-of-the-art adversarial attacks. This not only improves robustness but can also uncover potential vulnerabilities in your system.

9. **Efficient Hyperparameter Tuning**
   - Bayesian optimization
     - Gaussian Process-based optimization
     - Tree-structured Parzen Estimators (TPE)
   - Population-based training
     - Evolutionary strategies for joint model and hyperparameter optimization
   - Neural Architecture Search (NAS) for NLP
     - Efficient NAS techniques: ENAS, DARTS
     - Hardware-aware NAS for deployment optimization

Tip: Implement a multi-fidelity optimization approach, using cheaper approximations (e.g., training on a subset of data) in early stages of hyperparameter search before fine-tuning on the full dataset.

10. **Staying Updated and Contributing**
    - Follow top NLP conferences and workshops
      - ACL, EMNLP, NAACL, CoNLL
      - Specialized workshops: WMT, RepL4NLP
    - Engage with the NLP community
      - Participate in shared tasks and competitions
      - Contribute to open-source projects (e.g., Hugging Face Transformers, spaCy)
    - Reproduce and extend recent papers
      - Use platforms like PapersWithCode for implementations
      - Publish reproducibility reports and extensions

Tip: Set up a personal research workflow that includes regular paper reading, implementation of key techniques, and experimentation. Share your findings through blog posts or tech talks to solidify your understanding and contribute to the community.

Remember, NLP is a rapidly evolving field with new techniques and models emerging constantly. While mastering these advanced techniques is important, the ability to quickly adapt to new methods, critically evaluate their strengths and weaknesses, and creatively apply them to solve real-world problems is equally crucial. Always start with a strong baseline and iterate based on empirical results and domain-specific requirements. Happy NLP journey!
