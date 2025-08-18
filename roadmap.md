# 📘 Machine Learning Roadmap (12-Month Plan)

This is a structured week-by-week ML learning plan with checklists.
Goal: Go from fundamentals → advanced ML → research-level understanding.
Track progress by checking off tasks as you complete them ✅.

---

## **Phase 1: Foundations (Weeks 1–8)**

### **Week 1: Python Basics for ML**

- [x] Review Python fundamentals (functions, classes, list comprehensions).
- [x] Learn NumPy basics: arrays, broadcasting, matrix multiplication.
- [ ] Learn Pandas basics: DataFrames, indexing, filtering, groupby.
- [ ] Mini-project: Load the **Iris dataset** → clean, analyze, visualize.

**Resources:**

- Python Data Science Handbook (Ch. 1–3).
- Kaggle Python micro-course.

---

### **Week 2: Data Visualization & EDA**

- [ ] Learn Matplotlib basics (line plots, scatter plots, histograms).
- [ ] Learn Seaborn (pairplots, heatmaps, categorical plots).
- [ ] Practice: Visualize distributions, correlations, and outliers.
- [ ] Mini-project: Titanic dataset → EDA (exploratory data analysis).

**Resources:**

- Seaborn documentation.
- Kaggle Titanic dataset.

---

### **Week 3: Linear Algebra I**

- [ ] Vectors, dot product, matrix multiplication.
- [ ] Norms, orthogonality, projections.
- [ ] Eigenvalues, eigenvectors (intuition).
- [ ] Coding: Implement matrix multiplication & vector projections in NumPy.

**Resources:**

- 3Blue1Brown _Essence of Linear Algebra_ (Ch. 1–5).
- MIT OCW 18.06 (Lectures 1–3).

---

### **Week 4: Linear Algebra II**

- [ ] Matrix rank, null space, column space.
- [ ] Determinants, invertibility.
- [ ] Singular Value Decomposition (SVD).
- [ ] Mini-project: Implement PCA from scratch using eigen decomposition.

**Resources:**

- Strang’s _Introduction to Linear Algebra_.
- Bishop’s PRML (Ch. 12 for PCA).

---

### **Week 5: Probability Basics**

- [ ] Random variables, PMF, PDF, CDF.
- [ ] Expectation, variance, covariance.
- [ ] Bayes theorem, conditional probability.
- [ ] Coding: Simulate coin tosses, dice rolls, conditional probabilities.

**Resources:**

- Harvard Stat 110 (Lectures 1–5).
- _Introduction to Probability_ (Bertsekas & Tsitsiklis).

---

### **Week 6: Statistics**

- [ ] Distributions: Gaussian, Bernoulli, Binomial, Poisson.
- [ ] Law of large numbers, central limit theorem.
- [ ] Hypothesis testing, confidence intervals.
- [ ] Mini-project: Simulate CLT with random samples.

**Resources:**

- _All of Statistics_ (Wasserman).
- Khan Academy Probability & Stats refresher.

---

### **Week 7: Calculus for ML**

- [ ] Derivatives, gradients, partial derivatives.
- [ ] Chain rule, Jacobians, Hessians.
- [ ] Coding: Implement gradient of a quadratic function.
- [ ] Mini-project: Derive gradient descent for linear regression.

**Resources:**

- MIT OCW 18.02 Multivariable Calculus.
- Paul’s Online Math Notes.

---

### **Week 8: Optimization Basics**

- [ ] Convex vs non-convex optimization.
- [ ] Gradient descent, stochastic gradient descent (SGD).
- [ ] Momentum, RMSProp, Adam optimizers.
- [ ] Mini-project: Implement gradient descent from scratch in Python.

**Resources:**

- _Convex Optimization_ (Boyd & Vandenberghe, Ch. 1–3).

---

## **Phase 2: Core ML (Weeks 9–20)**

### **Week 9: Linear Regression**

- [ ] Ordinary least squares (closed form).
- [ ] Gradient descent for linear regression.
- [ ] Coding: Implement linear regression from scratch.
- [ ] Mini-project: Predict housing prices (Boston dataset).

**Resources:**

- CS229 Lecture Notes (Linear Regression).

---

### **Week 10: Logistic Regression**

- [ ] Sigmoid function, decision boundary.
- [ ] Maximum likelihood estimation.
- [ ] Gradient descent for logistic regression.
- [ ] Mini-project: Binary classification (spam detection).

**Resources:**

- CS229 Lecture Notes (Logistic Regression).

---

### **Week 11: Decision Trees & Ensembles**

- [ ] Decision trees (ID3, Gini impurity, entropy).
- [ ] Random Forests, Bagging.
- [ ] Coding: Implement a decision tree from scratch.
- [ ] Mini-project: Classify Titanic dataset with decision trees.

**Resources:**

- Elements of Statistical Learning (Ch. 9).

---

### **Week 12: Support Vector Machines (SVMs)**

- [ ] Margin maximization, hyperplanes.
- [ ] Kernel trick (RBF, polynomial).
- [ ] Mini-project: Implement SVM using scikit-learn on Iris dataset.

**Resources:**

- CS229 Lecture Notes (SVMs).

---

### **Week 13: Clustering (K-means)**

- [ ] K-means algorithm, objective function.
- [ ] Initialization methods (random, k-means++).
- [ ] Mini-project: Cluster handwritten digits (MNIST subset).

**Resources:**

- Bishop’s PRML (Ch. 9).

---

### **Week 14: Dimensionality Reduction (PCA)**

- [ ] PCA derivation (variance maximization).
- [ ] Eigen decomposition vs SVD.
- [ ] Mini-project: Visualize MNIST in 2D using PCA.

**Resources:**

- Elements of Statistical Learning (Ch. 14).

---

### **Week 15: Gaussian Mixture Models (GMMs)**

- [ ] Mixture of Gaussians.
- [ ] Expectation-Maximization (EM) algorithm.
- [ ] Mini-project: Fit GMMs to cluster Iris dataset.

**Resources:**

- Bishop’s PRML (Ch. 9.2).

---

### **Week 16: Model Evaluation**

- [ ] Cross-validation, train/test splits.
- [ ] Bias-variance tradeoff.
- [ ] ROC curves, precision-recall, F1 score.
- [ ] Mini-project: Evaluate multiple classifiers on Titanic dataset.

**Resources:**

- Elements of Statistical Learning (Ch. 7).

---

### **Week 17: Regularization**

- [ ] L1 (Lasso) vs L2 (Ridge) regularization.
- [ ] Elastic Net.
- [ ] Mini-project: Compare Ridge vs Lasso regression on housing dataset.

**Resources:**

- CS229 Lecture Notes (Regularization).

---

### **Week 18: Feature Engineering**

- [ ] Feature scaling (normalization, standardization).
- [ ] One-hot encoding, embeddings.
- [ ] Feature selection methods.
- [ ] Mini-project: Feature engineering on Kaggle Titanic dataset.

---

### **Week 19: Ensemble Methods**

- [ ] Bagging vs Boosting.
- [ ] AdaBoost, Gradient Boosting, XGBoost.
- [ ] Mini-project: Compare Random Forest vs XGBoost on classification task.

**Resources:**

- Elements of Statistical Learning (Ch. 10).

---

### **Week 20: Capstone (Classical ML)**

- [ ] Kaggle competition (Titanic or House Prices).
- [ ] Build full ML pipeline: preprocessing → model → evaluation.
- [ ] Document results in `projects/classical_ml/`.

---

## **Phase 3: Deep Learning (Weeks 21–32)**

### **Week 21: Neural Network Basics I**

- [ ] Perceptron model, activation functions (sigmoid, tanh, ReLU).
- [ ] Forward propagation.
- [ ] Loss functions: MSE, cross-entropy.
- [ ] Coding: Implement a single-layer perceptron from scratch.

**Resources:**

- Goodfellow _Deep Learning_ (Ch. 6).
- Michael Nielsen _Neural Networks and Deep Learning_ (Ch. 1–2).

---

### **Week 22: Neural Network Basics II**

- [ ] Backpropagation algorithm (chain rule).
- [ ] Weight initialization strategies.
- [ ] Coding: Implement a 2-layer NN from scratch (no PyTorch).
- [ ] Mini-project: Classify MNIST digits with your NN.

**Resources:**

- Goodfellow _Deep Learning_ (Ch. 6–7).

---

### **Week 23: Training Neural Networks**

- [ ] Optimizers: SGD, Momentum, RMSProp, Adam.
- [ ] Learning rate schedules.
- [ ] Overfitting: dropout, early stopping.
- [ ] Mini-project: Train NN on MNIST with dropout + Adam.

**Resources:**

- CS231n Lecture Notes (Optimization).

---

### **Week 24: PyTorch Basics**

- [ ] Tensors, autograd, nn.Module.
- [ ] Dataloaders, training loops.
- [ ] Mini-project: Re-implement MNIST NN in PyTorch.

**Resources:**

- PyTorch official tutorials.

---

### **Week 25: Convolutional Neural Networks (CNNs) I**

- [ ] Convolutions, filters, padding, stride.
- [ ] Pooling layers.
- [ ] Architectures: LeNet, AlexNet.
- [ ] Mini-project: Train a CNN on CIFAR-10.

**Resources:**

- CS231n (Lectures 5–7).

---

### **Week 26: CNNs II**

- [ ] Modern CNNs: VGG, ResNet (skip connections).
- [ ] Batch normalization.
- [ ] Mini-project: Implement ResNet on CIFAR-10.

**Resources:**

- ResNet paper (2015).

---

### **Week 27: Recurrent Neural Networks (RNNs)**

- [ ] RNN basics, vanishing gradient problem.
- [ ] LSTMs, GRUs.
- [ ] Mini-project: Train an RNN for text classification (IMDB reviews).

**Resources:**

- CS224n (Lectures 5–7).

---

### **Week 28: Sequence Models**

- [ ] Word embeddings (Word2Vec, GloVe).
- [ ] Seq2Seq models.
- [ ] Mini-project: Build a character-level text generator.

**Resources:**

- CS224n (Lectures 8–10).

---

### **Week 29: Attention Mechanisms**

- [ ] Attention basics (Bahdanau, Luong).
- [ ] Self-attention.
- [ ] Mini-project: Implement attention in a Seq2Seq model.

**Resources:**

- CS224n (Lecture 11).

---

### **Week 30: Transformers I**

- [ ] Read _Attention Is All You Need_.
- [ ] Encoder-decoder architecture.
- [ ] Multi-head attention, positional encoding.
- [ ] Mini-project: Implement a toy Transformer in PyTorch.

---

### **Week 31: Transformers II**

- [ ] BERT, GPT architectures.
- [ ] Transfer learning in NLP.
- [ ] Mini-project: Fine-tune BERT for sentiment analysis.

**Resources:**

- HuggingFace course.

---

### **Week 32: Deep Learning Capstone**

- [ ] Choose a dataset (CIFAR-10, IMDB, or custom).
- [ ] Build a full DL pipeline (data → model → training → evaluation).
- [ ] Document results in `projects/deep_learning/`.

---

## **Phase 4: Advanced ML (Weeks 33–44)**

### **Week 33: Generative Models I (Autoencoders)**

- [ ] Autoencoders, undercomplete vs overcomplete.
- [ ] Variational Autoencoders (VAEs).
- [ ] Mini-project: Implement a VAE on MNIST.

**Resources:**

- Kingma & Welling (VAE paper).

---

### **Week 34: Generative Models II (GANs)**

- [ ] GAN basics: generator, discriminator.
- [ ] Training instability, mode collapse.
- [ ] Mini-project: Implement a simple GAN on MNIST.

**Resources:**

- Goodfellow GAN paper (2014).

---

### **Week 35: Generative Models III (Advanced GANs)**

- [ ] DCGAN, WGAN, CycleGAN.
- [ ] Mini-project: Implement DCGAN on CIFAR-10.

---

### **Week 36: Diffusion Models**

- [ ] Diffusion process, denoising.
- [ ] DDPMs (Denoising Diffusion Probabilistic Models).
- [ ] Mini-project: Implement a toy diffusion model.

**Resources:**

- Ho et al. (DDPM paper, 2020).

---

### **Week 37: Reinforcement Learning I**

- [ ] Markov Decision Processes (MDPs).
- [ ] Value iteration, policy iteration.
- [ ] Mini-project: Solve GridWorld with value iteration.

**Resources:**

- Sutton & Barto (Ch. 3–4).

---

### **Week 38: Reinforcement Learning II**

- [ ] Q-learning, SARSA.
- [ ] Mini-project: Train an agent in OpenAI Gym (CartPole).

**Resources:**

- Sutton & Barto (Ch. 6).

---

### **Week 39: Deep Reinforcement Learning I**

- [ ] Deep Q-Networks (DQN).
- [ ] Experience replay, target networks.
- [ ] Mini-project: Implement DQN for CartPole.

**Resources:**

- DeepMind DQN paper (2015).

---

### **Week 40: Deep Reinforcement Learning II**

- [ ] Policy gradients.
- [ ] Actor-Critic methods.
- [ ] Mini-project: Implement REINFORCE algorithm.

**Resources:**

- Berkeley CS285 lectures.

---

### **Week 41: Probabilistic Models**

- [ ] Bayesian networks, graphical models.
- [ ] Variational inference.
- [ ] Mini-project: Implement a simple Bayesian network.

**Resources:**

- Daphne Koller’s PGM course.

---

### **Week 42: Advanced Optimization**

- [ ] Second-order methods (Newton, quasi-Newton).
- [ ] Adaptive optimizers (AdamW, LAMB).
- [ ] Mini-project: Compare optimizers on CIFAR-10.

---

### **Week 43: Scaling Deep Learning**

- [ ] Distributed training (DataParallel, DDP).
- [ ] Mixed precision training.
- [ ] Mini-project: Train ResNet with mixed precision.

**Resources:**

- PyTorch Lightning docs.

---

### **Week 44: Advanced ML Capstone**

- [ ] Choose an advanced topic (GAN, Transformer, RL).
- [ ] Reproduce a research paper.
- [ ] Document results in `papers/`.

---

## **Phase 5: Research & Contribution (Weeks 45–52)**

### **Week 45: Paper Reading I**

- [ ] Read AlexNet (2012).
- [ ] Summarize in `papers/alexnet/paper_summary.md`.
- [ ] Reproduce results on CIFAR-10.

---

### **Week 46: Paper Reading II**

- [ ] Read ResNet (2015).
- [ ] Summarize in `papers/resnet/paper_summary.md`.
- [ ] Implement ResNet in PyTorch.

---

### **Week 47: Paper Reading III**

- [ ] Read Transformer (2017).
- [ ] Summarize in `papers/transformer/paper_summary.md`.
- [ ] Implement a toy Transformer.

---

### **Week 48: Paper Reading IV**

- [ ] Read Diffusion Models (2020).
- [ ] Summarize in `papers/diffusion/paper_summary.md`.
- [ ] Implement a toy diffusion model.

---

### **Week 49: Open Source Contribution I**

- [ ] Explore PyTorch or HuggingFace issues.
- [ ] Fix a bug or improve docs.
- [ ] Submit your first PR.

---

### **Week 50: Open Source Contribution II**

- [ ] Implement a small feature or example.
- [ ] Submit PR.
- [ ] Document contribution in `contributions.md`.

---

### **Week 51: Portfolio Building**

- [ ] Write 1–2 blog posts (Medium/GitHub Pages).
- [ ] Showcase projects in `projects/`.
- [ ] Polish repo README.

---

### **Week 52: Reflection & Next Steps**

- [ ] Review all notes, projects, papers.
- [ ] Write a final reflection in `weekly_log.md`.
- [ ] Plan Year 2 (specialization: CV, NLP, RL, or research).

---

✅ By the end of this plan, you’ll be able to:

- Derive and implement ML/DL algorithms from scratch.
- Train and fine-tune state-of-the-art models.
- Read and reproduce research papers.
- Contribute to the ML community.
