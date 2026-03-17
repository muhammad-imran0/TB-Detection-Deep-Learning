# Machine Vision-Based Tuberculosis Detection Using Deep Learning on Chest X-Ray Images

**Student ID:** 3004795  
**Module Code:** CN7023  
**Module Title:** Artificial Intelligence & Machine Vision  
**Module Leader:** Dr Shaheen Khatoon  
**Submission Date:** May 6, 2026  

---

## 1. Introduction

### 1.1 Objectives

The main goal of this project is to build a deep learning system in MATLAB that can look at a chest X-ray and tell whether the patient has Tuberculosis or not. To do this properly, I built and compared three different CNN models:

- A custom CNN designed from scratch (a relatively shallow network with 4 convolutional blocks)
- Transfer learning using GoogLeNet (an efficient architecture with inception modules)
- Transfer learning using ResNet-18 (a residual network with skip connections)

By comparing all three, I can see how a simple hand-built network stacks up against pretrained models that have already learned from millions of images.

### 1.2 Identification of Real-World Problem

Tuberculosis (TB) is still one of the deadliest infectious diseases in the world, with roughly 10.6 million new cases and 1.3 million deaths every year (WHO, 2023). The main way doctors screen for TB is by looking at chest X-rays, but this depends on having a trained radiologist available — and in many developing countries, there simply aren't enough.

On top of that, different doctors can look at the same X-ray and disagree on the diagnosis. This inconsistency is a real problem when lives are at stake. An automated system that can quickly flag suspicious X-rays would help prioritise cases, catch more TB cases early, and support healthcare workers in places where specialist radiologists are scarce.

Deep learning-based image classification is a natural fit here because CNNs can learn to spot subtle patterns in medical images that might be missed or interpreted differently by human eyes.

### 1.3 Overview of Report Content

Section 2 explains the approach taken and why these specific architectures were chosen. Section 3 describes the dataset. Section 4 covers how the images were preprocessed and encoded for the networks. Section 5 details the three network architectures and their training setups. Section 6 presents the results (accuracy, curves, confusion matrices, classification metrics). Section 7 critically analyses what worked, what didn't, and what could be improved. Section 8 wraps up with conclusions.

---

## 2. Creative and Innovative Approaches

### 2.1 Innovative Design Approach

This project takes a three-model comparative approach, which goes beyond just using a single network. By training three architecturally different models on the same data with the same splits, I get a fair comparison that reveals the strengths and weaknesses of each approach.

**Experiment 1 — Custom CNN (Shallow Network):**
- Designed from scratch following the pattern learned in the CN7023 Week 6 practical
- 4 convolutional blocks with progressively more filters (16 → 32 → 64 → 128)
- Includes Batch Normalisation for stable training and Dropout (50%) to fight overfitting
- This shows understanding of how to build a CNN architecture from the ground up

**Experiment 2 — GoogLeNet Transfer Learning:**
- Uses GoogLeNet pretrained on ImageNet (1.2M images, 1000 classes)
- GoogLeNet introduced the "inception module" which runs multiple filter sizes (1x1, 3x3, 5x5) in parallel and combines them, instead of just stacking convolutions one after another
- Only 5 million parameters thanks to global average pooling instead of large fully connected layers
- Won the ImageNet competition in 2014
- I removed the last 3 layers (loss3-classifier, prob, output) and added my own 2-class output
- Follows the transfer learning approach from the CN7023 Week 7 practical

**Experiment 3 — ResNet-18 Transfer Learning:**
- Uses ResNet-18 pretrained on ImageNet
- ResNet introduced residual (skip) connections that let gradients flow directly through the network, solving the "vanishing gradient" problem that limited how deep networks could go
- 18 layers and only 11 million parameters
- Won the ImageNet competition in 2015 and is widely used in medical imaging research
- Same transfer learning approach: remove the final classification layers, plug in my own

### 2.2 Description of Methods and Strategies

**Data Preprocessing Pipeline:**
1. Images loaded via `imageDatastore` with labels from folder names
2. Resized to 224×224 using `augmentedImageDatastore`
3. Grayscale X-rays converted to 3-channel RGB (pretrained models expect colour input)
4. Training set balanced via oversampling (TB images duplicated to match Normal count)
5. Training images augmented with random rotation (±15°), translation (±10px), horizontal flip, and scale (0.9–1.1)

**Training Strategy:**
- Custom CNN: Adam optimiser with learning rate 0.001 and piecewise decay (halved every 5 epochs)
- Transfer learning models: SGDM optimiser with learning rate 0.0001 (much lower to protect pretrained weights)
- New classification layers given 10× higher learning rate to catch up
- Early stopping with patience of 5 to prevent overfitting
- 70/15/15 train-validation-test split using `splitEachLabel`

### 2.3 Justification for Chosen Approach

1. **Handling class imbalance**: The dataset has 3,500 Normal but only 700 TB images, so I oversampled the TB training images to balance the classes. Without this, the model could get ~83% accuracy by always predicting "Normal" while completely missing every TB case.

2. **Transfer learning makes sense here**: Medical datasets are relatively small. Both GoogLeNet and ResNet-18 have already learned useful features (edges, textures, shapes) from millions of images, and those features transfer well to X-ray analysis.

3. **Three architecturally different models**: The custom CNN tests what a simple network can do. GoogLeNet represents the "inception module" approach with parallel filter branches. ResNet-18 represents the "skip connections" breakthrough. Comparing all three gives genuine insight into which approach works best for TB detection.

4. **MATLAB ecosystem**: The Deep Learning Toolbox provides everything needed end-to-end — data loading, augmentation, training, evaluation — all in one environment, consistent with what was taught in CN7023.

---

## 3. Dataset Description

**Dataset Name:** Tuberculosis (TB) Chest X-ray Database  
**Source:** Kaggle  
**Link:** https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset  
**Image Dimensions:** 512 × 512 pixels (resized to 224 × 224 for model input)  
**Number of Classes:** 2 (Binary Classification)

| Class | Number of Images |
|-------|-----------------|
| Normal | 3,500 |
| Tuberculosis | 700 |
| **Total** | **4,200** |

This dataset is **imbalanced** — there are 5× more Normal images than TB images. This is actually realistic because in clinical settings most chest X-rays come back normal, but it creates a challenge for training because a model could achieve high accuracy by simply predicting "Normal" every time. I addressed this through oversampling (see Section 4).

The dataset was compiled from multiple hospital sources including Shenzhen No.3 People's Hospital (China) and Montgomery County's TB screening programme (USA).

**[INSERT SCREENSHOT: Class distribution bar chart from MATLAB output]**

**[INSERT SCREENSHOT: Sample chest X-ray images (5 Normal, 5 TB) from MATLAB output]**

---

## 4. Data Encoding and Preprocessing

### 4.1 Image Encoding

1. **Image Datastore**: `imageDatastore` loads all 4,200 images and assigns labels from folder names ('Normal' and 'Tuberculosis').

2. **Resizing**: All images resized from 512×512 to 224×224 using `augmentedImageDatastore`. This matches GoogLeNet and ResNet-18's expected input size, reduces memory usage, and still keeps enough resolution to see diagnostic features.

3. **Colour Conversion**: X-rays are grayscale but the pretrained networks expect 3-channel RGB input. The `'ColorPreprocessing', 'gray2rgb'` option handles this by replicating the single channel across all three.

4. **Normalisation**: The `imageInputLayer` in MATLAB automatically normalises pixel values. Batch Normalisation layers within the networks handle per-layer normalisation.

5. **Label Encoding**: Categorical labels are created automatically from folder names, which the `classificationLayer` uses directly during training and evaluation.

### 4.2 Oversampling to Handle Class Imbalance

After splitting, the training set had 2,450 Normal images but only 490 TB images. To prevent the model from being biased towards the majority class, I duplicated the TB training images until both classes had 2,450 samples each. This brought the total training set to 4,900 images.

| Stage | Normal | TB | Total |
|-------|--------|-----|-------|
| Before oversampling | 2,450 | 490 | 2,940 |
| After oversampling | 2,450 | 2,450 | 4,900 |

### 4.3 Data Augmentation

Applied only to training images to artificially increase diversity:

| Augmentation | Parameter | Value | Why |
|-------------|-----------|-------|-----|
| Rotation | `RandRotation` | [-15°, 15°] | Patients aren't always positioned identically |
| X Shift | `RandXTranslation` | [-10, 10] px | Simulates lateral positioning differences |
| Y Shift | `RandYTranslation` | [-10, 10] px | Simulates vertical positioning differences |
| Horizontal Flip | `RandXReflection` | true | X-rays can be mirrored depending on the machine |
| Scale | `RandScale` | [0.9, 1.1] | Simulates distance variations from the X-ray source |

Validation and test sets are not augmented — they need to stay consistent for fair evaluation.

### 4.4 Data Split

| Subset | % | Normal | TB | Total |
|--------|---|--------|-----|-------|
| Training | 70% | 2,450 | 490 (→ 2,450 after oversampling) | 2,940 (→ 4,900) |
| Validation | 15% | 525 | 105 | 630 |
| Test | 15% | 525 | 105 | 630 |

---

## 5. Network Architecture and Training

### 5.1 Custom CNN Architecture (Experiment 1)

A shallow network with 4 convolutional blocks, designed following the Week 6 lab pattern:

| Layer Block | Layers | Output Size |
|-------------|--------|-------------|
| Input | `imageInputLayer([224 224 3])` | 224×224×3 |
| Block 1 | Conv(3,16) → BN → ReLU → MaxPool(2) | 112×112×16 |
| Block 2 | Conv(3,32) → BN → ReLU → MaxPool(2) | 56×56×32 |
| Block 3 | Conv(3,64) → BN → ReLU → MaxPool(2) | 28×28×64 |
| Block 4 | Conv(3,128) → BN → ReLU → MaxPool(2) | 14×14×128 |
| FC Head | FC(256) → ReLU → Dropout(0.5) → FC(2) → Softmax → Classification | 2 classes |

The filter count doubles each block (16→32→64→128), letting earlier layers detect simple edges while deeper layers recognise more complex structures. Dropout randomly switches off half the neurons during training, forcing the network to spread its knowledge across more connections and reducing overfitting.

### 5.2 GoogLeNet Transfer Learning (Experiment 2)

GoogLeNet won the 2014 ImageNet competition by introducing the "inception module" — instead of choosing one filter size per layer, it runs 1×1, 3×3, and 5×5 convolutions in parallel and concatenates the results. This lets the network capture features at multiple scales simultaneously.

Despite going 22 layers deep, GoogLeNet has only about 5 million parameters because it uses global average pooling instead of the large fully connected layers that made VGG16 so heavy (140M parameters). This makes GoogLeNet much faster to train, which was important since I was running on CPU.

**Modifications for TB detection:**
- Removed original `loss3-classifier` (1000 classes), `prob`, and `output` layers
- Added: `fullyConnectedLayer(2)` → `softmaxLayer` → `classificationLayer`
- Connected new layers to `pool5-drop_7x7_s1` (the dropout layer after global average pooling)
- New layers get `WeightLearnRateFactor = 10` so they learn faster than the frozen pretrained layers

### 5.3 ResNet-18 Transfer Learning (Experiment 3)

ResNet introduced skip (residual) connections where the input to a block gets added directly to the output, letting gradients bypass layers that might otherwise cause them to vanish. This breakthrough meant networks could go much deeper without the performance degradation that plagued earlier architectures.

ResNet-18 has 18 layers and only 11M parameters. Despite being larger than GoogLeNet, it uses a completely different architectural philosophy (residual learning vs inception modules), making it a valuable comparison.

**Modifications for TB detection:**
- Removed `fc1000` (1000 classes), `prob`, and `ClassificationLayer_predictions`
- Added: `fullyConnectedLayer(2)` → `softmaxLayer` → `classificationLayer`
- Connected new layers to the `pool5` (global average pooling) output
- New layers get `WeightLearnRateFactor = 10`

### 5.4 Training Configuration

| Parameter | Custom CNN | GoogLeNet | ResNet-18 |
|-----------|-----------|-----------|-----------|
| Optimiser | Adam | SGDM | SGDM |
| Initial Learning Rate | 0.001 | 0.0001 | 0.0001 |
| LR Schedule | Piecewise (×0.5 every 5 epochs) | Piecewise (×0.5 every 5 epochs) | Piecewise (×0.5 every 5 epochs) |
| Max Epochs | 20 | 15 | 15 |
| Mini-Batch Size | 32 | 32 | 32 |
| Validation Patience | 5 | 5 | 5 |
| Loss Function | Cross-Entropy | Cross-Entropy | Cross-Entropy |

I used Adam for the custom CNN because it adapts the learning rate automatically during training, which usually gives better results when training from scratch. The transfer learning models use SGDM with a much lower learning rate (0.0001 vs 0.001) because their pretrained weights are already good — large updates would destroy the useful features they learned from ImageNet.

---

## 6. Results Obtained

### 6.1 Test Set Accuracy

All three models were evaluated on the held-out test set (630 images that were never seen during training):

| Model | Validation Accuracy | Test Accuracy |
|-------|-------------------|-------------|
| Custom CNN | 95.87% | 95.40% |
| GoogLeNet | 98.89% | 100.00% |
| ResNet-18 | 98.25% | 99.52% |

GoogLeNet achieved the highest test accuracy at 100.00%, followed by ResNet-18 at 99.52% and the Custom CNN at 95.40%. All three models exceeded the 90% accuracy target. GoogLeNet classified all 630 test images correctly, demonstrating the strength of inception modules combined with transfer learning for this task.

### 6.2 Accuracy Curves

Three separate accuracy curve figures were generated, each showing:
- **Training accuracy** (blue) — performance on training data per epoch
- **Validation accuracy** (red) — performance on unseen validation data per epoch
- **Test accuracy** (green line) — final performance on the held-out test set

These curves help visualise how quickly each model learns and whether overfitting occurs (training accuracy climbing while validation accuracy stalls or drops).

**[INSERT SCREENSHOT: Custom CNN accuracy curve from MATLAB output]**

**[INSERT SCREENSHOT: GoogLeNet accuracy curve from MATLAB output]**

**[INSERT SCREENSHOT: ResNet-18 accuracy curve from MATLAB output]**

### 6.3 Loss Curves

Loss curves for all three models were plotted side by side. They show training loss and validation loss per epoch, which helps spot overfitting — if training loss keeps decreasing but validation loss starts increasing, the model is memorising rather than learning.

**[INSERT SCREENSHOT: Loss curves (3 subplots) from MATLAB output]**

### 6.4 Confusion Matrices

Confusion matrices were generated for all three models using `confusionmat` and `confusionchart`. Each matrix shows:
- Diagonal cells: correct predictions (True Positives and True Negatives)
- Off-diagonal cells: misclassifications
- Row summaries: recall (sensitivity) per class
- Column summaries: precision per class

**[INSERT SCREENSHOT: Custom CNN confusion matrix from MATLAB output]**

**[INSERT SCREENSHOT: GoogLeNet confusion matrix from MATLAB output]**

**[INSERT SCREENSHOT: ResNet-18 confusion matrix from MATLAB output]**

### 6.5 ROC Curves

ROC (Receiver Operating Characteristic) curves were plotted for all three models. These show the trade-off between the True Positive Rate (catching actual TB cases) and the False Positive Rate (falsely flagging Normal X-rays as TB) at different classification thresholds. The AUC (Area Under Curve) value summarises model performance — a value closer to 1.0 means the model is better at separating TB from Normal.

ROC analysis is especially important in medical imaging because missing a TB case (false negative) has far worse consequences than a false alarm (false positive).

| Model | AUC |
|-------|-----|
| Custom CNN | 0.9926 |
| GoogLeNet | 1.0000 |
| ResNet-18 | 0.9998 |

All three models achieved AUC values above 0.99, with GoogLeNet reaching a perfect 1.0000, confirming excellent class separation across all thresholds.

**[INSERT SCREENSHOT: ROC curves (3 subplots) from MATLAB output]**

**[INSERT SCREENSHOT: ROC comparison (all 3 on one plot) from MATLAB output]**

### 6.6 Classification Metrics

Per-class and overall Precision, Recall, and F1-Score were computed for each model:

- **Precision** = TP / (TP + FP) — of all images flagged as TB, how many actually had TB?
- **Recall** = TP / (TP + FN) — of all actual TB patients, how many did the model catch?
- **F1-Score** = 2 × (Precision × Recall) / (Precision + Recall) — harmonic mean balancing both

**Custom CNN — Per-class metrics:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Normal | 0.9901 | 0.9543 | 0.9719 |
| Tuberculosis | 0.8065 | 0.9524 | 0.8734 |

Overall: Precision 0.9540, Recall 0.9540, F1 0.9540

**GoogLeNet — Per-class metrics:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Normal | 1.0000 | 1.0000 | 1.0000 |
| Tuberculosis | 1.0000 | 1.0000 | 1.0000 |

Overall: Precision 1.0000, Recall 1.0000, F1 1.0000

**ResNet-18 — Per-class metrics:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Normal | 0.9981 | 0.9962 | 0.9971 |
| Tuberculosis | 0.9811 | 0.9905 | 0.9858 |

Overall: Precision 0.9952, Recall 0.9952, F1 0.9952

### 6.7 Model Comparison

A grouped bar chart comparing all four metrics (Accuracy, Precision, Recall, F1-Score) across the three models was generated, with all models displayed side by side for easy comparison. The y-axis is zoomed in (93–101%) to highlight the differences between the high-performing models.

**[INSERT SCREENSHOT: Model comparison grouped bar chart from MATLAB output]**

### 6.8 Sample Predictions

Eight random test images were displayed with their true and predicted labels using the best model (GoogLeNet). Correct predictions are shown in green, incorrect in red.

**[INSERT SCREENSHOT: Sample test predictions (GoogLeNet) from MATLAB output]**

---

## 7. Critical Analysis of Results

### 7.1 How Results Were Achieved

1. **Oversampling to fix class imbalance**: The original dataset had 5× more Normal images than TB. By duplicating TB training images to match the Normal count, I made sure the model couldn't just predict "Normal" for everything. This was critical — without it, a model could reach ~83% accuracy while missing every TB case.

2. **Data augmentation**: Random rotations, translations, flips, and scaling during training exposed the model to realistic variations, preventing it from memorising specific images.

3. **Batch Normalisation + Dropout**: These regularisation techniques worked together to keep training stable and prevent overfitting — particularly important given the relatively small dataset.

4. **Transfer learning advantage**: GoogLeNet and ResNet-18 benefit enormously from features learned on ImageNet. Low-level features like edges and textures are universal across image domains, so they transfer well to medical imaging even though ImageNet contains no X-rays.

5. **GoogLeNet's efficiency**: GoogLeNet achieved the best results (100.00% test accuracy) with only 5M parameters, thanks to its inception modules and global average pooling. The parallel filter branches let it capture features at multiple scales, which is especially useful for X-rays where TB abnormalities can vary in size.

### 7.2 Observations on Individual Models

**Custom CNN (95.40%):** Still performed above 95% for a network built from scratch, showing that even a shallow 4-block CNN can learn to distinguish TB from Normal X-rays when combined with proper preprocessing, augmentation, and oversampling. However, the clear gap between this and the transfer learning models (95.40% vs 99.52–100%) demonstrates why transfer learning is so valuable — pretrained features give a significant head start.

**GoogLeNet (100.00% — Best Model):** Achieved perfect accuracy on the test set, correctly classifying all 630 test images. It is worth noting that 100% on a 630-image test set does not mean the model is perfect on all X-rays — a larger or more diverse test set might reveal some misclassifications. However, this result clearly shows that GoogLeNet's inception modules are well-suited for detecting TB patterns at multiple scales. The AUC of 1.0000 further confirms excellent class separation.

**ResNet-18 (99.52%):** Achieved strong final results, though the validation accuracy during training was noticeably unstable — it fluctuated between 36% and 78% before early stopping kicked in at epoch 2. Despite this, the final evaluation on the full test set gave a stable 99.52%. This instability during training is likely due to the batch normalisation layers in ResNet behaving differently during training (using mini-batch statistics) versus evaluation (using running averages), combined with the small validation set.

### 7.3 Comparison with Published Research

Rahman et al. (2020) — the creators of the TB chest X-ray dataset used in this project — tested nine different CNN architectures on the same data using transfer learning. Their best results were:

| Study | Model | Approach | Accuracy |
|-------|-------|----------|----------|
| Rahman et al. (2020) | ChexNet | Whole X-ray images | 96.47% |
| Rahman et al. (2020) | DenseNet201 | Segmented lung images | 98.60% |
| **This project** | **GoogLeNet** | **Whole X-ray + oversampling** | **100.00%** |
| **This project** | **ResNet-18** | **Whole X-ray + oversampling** | **99.52%** |
| **This project** | **Custom CNN** | **Whole X-ray + oversampling** | **95.40%** |

My GoogLeNet model (100.00%) outperformed their best whole-image result (ChexNet at 96.47%) by a significant margin, and even surpassed their segmented-image result (DenseNet201 at 98.60%) without needing any lung segmentation step. This suggests that combining oversampling to fix class imbalance, data augmentation, and the inception module architecture can match or exceed results that required additional preprocessing like lung segmentation.

It is worth noting that Rahman et al. used a balanced version of the dataset (3,500 per class), while I used the imbalanced version (3,500 Normal, 700 TB) and addressed the imbalance through oversampling. The fact that my results are comparable or better indicates that oversampling was an effective strategy.

### 7.4 How Results Can Be Improved

1. **Deeper ResNet variants**: ResNet-50 or ResNet-101 have more capacity to learn complex patterns. Available in MATLAB via `resnet50` and `resnet101`.

2. **VGG16 or VGG19**: These deeper sequential architectures have 140M+ parameters and might capture more fine-grained features, but would require a GPU for practical training times.

3. **Ensemble methods**: Combining predictions from all three models through majority voting could reduce individual model weaknesses and improve overall robustness.

4. **Grad-CAM visualisation**: MATLAB's `gradCAM` function can highlight which parts of the X-ray the model focuses on, which would increase clinical trust and provide interpretability.

5. **Hyperparameter tuning**: Systematic search using `bayesopt` for learning rate, dropout rate, batch size, and augmentation parameters could squeeze out extra performance.

6. **Higher resolution input**: Training at 299×299 or 512×512 instead of 224×224 would preserve more diagnostic detail, at the cost of longer training times and higher memory usage.

7. **K-fold cross-validation**: Using 5-fold cross-validation instead of a single 70/15/15 split would give more reliable and stable performance estimates.

### 7.5 Factors Affecting Performance

- **Class imbalance**: The 5:1 ratio of Normal to TB images required oversampling to prevent bias. Without it, accuracy numbers would be misleadingly high.
- **Image quality**: X-rays from different hospitals vary in exposure, contrast, and resolution, which can confuse models trained primarily on data from specific sources.
- **Subtle early-stage TB**: Cases with minimal radiographic changes are harder to detect and may account for most misclassifications.
- **Resolution trade-off**: Resizing from 512×512 to 224×224 speeds up training but sacrifices fine detail that could be diagnostically relevant.
- **CPU training constraints**: Running on MATLAB Online (CPU only) meant that very large models like VGG16 were impractical, which is why I chose GoogLeNet as a more efficient alternative.

---

## 8. Conclusion

This project tackled the real-world healthcare problem of automated TB detection from chest X-rays. Three different deep learning approaches were implemented in MATLAB and compared head-to-head on the same dataset:

1. A custom CNN built from scratch demonstrated that even a relatively simple network can learn to distinguish Normal from TB X-rays, achieving 95.40% test accuracy with proper preprocessing and training.

2. GoogLeNet transfer learning achieved the best results at 100.00% test accuracy, correctly classifying all 630 test images. Its inception modules — which process images at multiple scales simultaneously — are well-suited for detecting TB patterns in X-rays. Its small parameter count (5M) also made it fast to train on CPU.

3. ResNet-18 transfer learning achieved 99.52% test accuracy, confirming that skip connections enable effective feature learning. However, its training was less stable than GoogLeNet's, suggesting that the inception architecture may be a better fit for this particular task.

**Key Takeaways:**

1. Transfer learning consistently outperforms training from scratch when the dataset is on the smaller side — the pretrained features provide a massive head start.

2. Architecture matters: GoogLeNet's parallel inception modules outperformed both the custom CNN and ResNet-18, achieving the highest accuracy with the fewest parameters.

3. Handling class imbalance through oversampling was essential — the original 5:1 ratio would have produced misleading results without correction.

4. Proper preprocessing (resizing, colour conversion, augmentation) and regularisation (Batch Normalisation, Dropout, early stopping) are essential for getting good results in medical imaging.

5. While the results are promising for a decision-support tool, clinical deployment would require further validation on larger multi-centre datasets, regulatory approval, and integration with existing diagnostic workflows.

---

## References

1. Rahman, T., Khandakar, A., Kadir, M.A., et al. (2020). "Reliable Tuberculosis Detection using Chest X-ray with Deep Learning, Segmentation and Visualization." *IEEE Access*, 8, 191586-191601.

2. Goswami, A., et al. (2023). "Chest X-Ray Based Detection of Tuberculosis Using Deep Learning." *Cureus*, 15(7).

3. Hashmi, M. F., et al. (2020). "Transfer Learning with Deep Convolutional Neural Network (CNN) for Pneumonia Detection Using Chest X-ray." *Applied Sciences*, 10(9), 3233.

4. Szegedy, C., et al. (2015). "Going Deeper with Convolutions." *Proceedings of CVPR*. (GoogLeNet / Inception)

5. He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition." *Proceedings of CVPR*, pp. 770-778.

6. Simonyan, K., & Zisserman, A. (2015). "Very Deep Convolutional Networks for Large-Scale Image Recognition." *Proceedings of ICLR*. (VGG)

7. Canziani, A., Paszke, A., & Culurciello, E. (2017). "An Analysis of Deep Neural Network Models for Practical Applications." *arXiv:1605.07678*.

8. World Health Organization. (2023). "Global Tuberculosis Report 2023." Geneva: WHO.

9. Kaggle. "Tuberculosis (TB) Chest X-ray Database." Available at: https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset

10. MathWorks. "Deep Learning Toolbox Documentation." Available at: https://uk.mathworks.com/help/deeplearning/
