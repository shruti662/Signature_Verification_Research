"Offline Signature Verification using Few-Shot Learning"
📌 Project Overview

This project focuses on developing a few-shot offline signature verification system capable of accurately verifying handwritten signatures using only a limited number of genuine samples per user. The system is designed to address real-world biometric constraints where large datasets are not always available.

🎯 Objectives
Build a robust signature verification model using minimal training data (5 samples per user)
Combine Siamese Networks and Prototypical Learning for improved accuracy
Ensure scale-invariant and stable similarity comparison
Develop a system suitable for real-world authentication scenarios

🧠 Methodology

The proposed system uses a hybrid deep learning approach:

Siamese Neural Network
Learns similarity between pairs of signatures
Prototypical Learning
Generates a representative embedding (prototype) for each user
Embedding-Based Verification
L2 Normalization
Cosine Similarity for comparison
Threshold Optimization
Determines the best decision boundary for genuine vs forged signatures

⚙️ Tech Stack
Programming Language: Python
Libraries & Frameworks:
TensorFlow / PyTorch
NumPy
OpenCV
Scikit-learn
Matplotlib

📊 Work Done So Far
Data preprocessing and normalization of signature images
Implementation of Siamese network architecture
Integration of prototypical learning for user representation
Feature embedding generation using deep learning
Similarity computation using cosine distance
Initial threshold tuning experiments

📁 Dataset
Offline handwritten signature dataset (genuine + forged samples)
CEDAR Signature Dataset 
A benchmark dataset consisting of genuine and skilled forged signatures from multiple users. It is widely used for baseline evaluation in offline signature verification research.
GPDS-300 Signature Dataset
A subset of the GPDS dataset containing signatures from 300 users. It introduces significant intra-class variability and is used to enhance model generalization and performance across diverse writing styles.
MCYT-100 Signature Dataset 
A structured dataset containing signatures from 100 users, including both genuine and forged samples. It is useful for evaluating consistency and verification accuracy under controlled conditions.
Preprocessed into grayscale and resized format for model input

📈 Future Work
Improve model accuracy with data augmentation
Optimize threshold selection using ROC analysis
Deploy as a web-based authentication system
Integrate real-time verification API

🚀 Applications
Banking authentication systems
Document verification
Identity validation systems
Secure digital transactions


Shruti Gadilkar
Final Year Computer Engineering Student
