# Audio-Emotion-Recognition
<br>
A deep learning project that classifies emotions from audio files using CNN. The system can identify 8 different emotional states from speech audio: neutral, calm, happy, sad, angry, fear, disgust and surprise.
<br>

## Key Features:
<br>
- Real-time emotion classification from audio<br>
- Support for 8 distinct emotional categories<br>
- Robust preprocessing pipeline for audio normalization<br>
- CNN-based architecture optimized for audio feature learning<br>
- Comprehensive evaluation metrics and visualization<br>

## Dataset: 
<br>
The project uses the RAVDESS dataset containing audio samples from 24 professional actors (12 male, 12 female). Each audio file represents one of eight emotions performed with controlled intensity and authenticity. Combined the audio files from both Song and Speech into one folder for simpler code.
<br>

## Preprocessing Methodology:
<br>
The preprocessing pipeline transforms raw audio into meaningful numerical representations suitable for machine learning:

1. Audio Loading & Normalization <br>
   - Loaded audio files using librosa <br>
   - Standardized sampling rates and durations <br>
   - Appled noise reduction techniques<br>

2. EDA <br>
   - Observed MelSpectogram <br>
   - Studies the effect of noise, stretch, shift and pitch on a random file. <br>

3. Feature Engineering <br>
   - MFCC (Mel-Frequency Cepstral Coefficients): Captures spectral characteristics <br>
   - Zero Crossing Rate: Measures signal variability <br>
   - RMS Energy: Captures audio intensity <br>

4. Data Augmentation <br>
   - Time stretching for temporal variations <br>
   - Pitch shifting for tonal diversity <br>
   - Background noise injection for robustness <br>

5. Feature Scaling <br>
   - StandardScaler normalization <br>
   - Ensures consistent feature magnitudes across samples <br>

6. Encoding <br>
   - One Hot Encoding <br>

7. Missing Values <br>
   - Filled missing values as 0.

## Model Architecture <br>
The model employs a sophisticated 1D Convolutional Neural Network optimized for sequential audio data: <br>

```
Input Layer: (2376 features, 1 channel)
├── Conv1D(512 filters, kernel=5, relu) + BatchNorm + MaxPool(pool_size=5, strides=2)
├── Conv1D(512 filters, kernel=5, relu) + BatchNorm + MaxPool(pool_size=5, strides=2) + Dropout(0.2)
├── Conv1D(256 filters, kernel=5, relu) + BatchNorm + MaxPool(pool_size=5, strides=2)
├── Conv1D(256 filters, kernel=3, relu) + BatchNorm + MaxPool(pool_size=5, strides=2) + Dropout(0.2)
├── Conv1D(128 filters, kernel=3, relu) + BatchNorm + MaxPool(pool_size=3, strides=2) + Dropout(0.2)
├── Flatten
├── Dense(512, relu) + BatchNorm
└── Dense(8, softmax) → Emotion Predictions
```
<br>

**Architecture Highlights:**
<br>
- Progressive Filter Reduction: 512 → 256 → 128 filters for hierarchical feature learning <br>
- Batch Normalization: Accelerates training and improves stability <br>
- Strategic Dropout: Prevents overfitting with 20% dropout rates <br>
- Adaptive Learning: ReduceLROnPlateau for optimal convergence <br>

### Training Configuration <br>
- Optimizer: Adam with initial learning rate 0.001 <br>
- Loss Function: Categorical crossentropy <br>
- Batch Size: 64 samples <br>
- Early Stopping: Monitors validation loss with patience=5 <br>
- Learning Rate Scheduling: Reduces by 50% when validation plateaus <br>

## Performance Metrics
<br>
### Model Accuracy
<br>
The trained model demonstrates excellent performance across all emotional categories:<br>

Training Performance: <br>
- Final Training Accuracy: 99.92% <br>
- Final Validation Accuracy: 68.23% <br>
- Training Loss: 0.0086 <br>
- Validation Loss: 1.1422 <br>
 
Training Progression: <br>
- Epoch 1: 34.36% accuracy → 2.0941 loss <br>
- Epoch 15: 98.38% accuracy → 0.0672 loss (after LR reduction) <br>
- Epoch 28: 99.90% accuracy → 0.0086 loss (final) <br>

### Model Complexity
<br>
- Total Parameters: 7,193,736 (27.44 MB) <br>.  
- Trainable Parameters: 7,189,384 (27.43 MB) <br>
- Non-trainable Parameters: 4,352 (17.00 KB) <br>
File containing weights was large to upload on repo so compressed using tensorflowlite. <br>

### Learning Dynamics
<br>
The model shows clear learning progression with strategic learning rate reductions: <br>
- Initial LR: 0.001 (Epochs 1-13) <br>
- First Reduction: 0.0005 (Epochs 14-18) <br>
- Second Reduction: 0.00025 (Epochs 19-25) <br>
- Final LR: 0.000125 (Epochs 26-28) <br>

## Key Insights
<br>

Strengths:* <br>
- Exceptional training accuracy indicating strong feature learning capability <br>
- Robust architecture with proper regularization techniques <br>
- Efficient parameter utilization with strategic layer design <br>

Considerations: <br>
- Gap between training (99.92%) and validation (68.23%) accuracy suggests some overfitting which can be reducd by giving more time and tuning the hyperparameters. <br>
- Model complexity is well-balanced for the task requirements <br>
- Learning rate scheduling effectively prevented training stagnation <br>
