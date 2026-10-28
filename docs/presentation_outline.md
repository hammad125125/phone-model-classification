# Semester Project Presentation Slide Deck Outline

**Presentation Title:** Mobile Phone Model Classification Using Deep Learning and Transfer Learning  
**Target Duration:** 10–12 Minutes + 3 Minutes Q&A  

---

### Slide 1: Title & Team Introduction
- Project Title: Mobile Phone Model Classification Using Deep Learning and Transfer Learning
- Student Names, Student IDs, Course Title, Instructor / Advisor Name
- University Department & Date

### Slide 2: Motivation & Problem Formulation
- Why mobile phone classification? (Automated e-commerce trade-ins, recycling facilities, inventory intake).
- The Challenge: Modern smartphones look nearly identical from the front (all-screen glass slabs); rear camera modules provide the primary visual discriminant.

### Slide 3: Objectives & Technical Scope
- Configurable multi-class brand and model classification.
- Transfer learning with MobileNetV2 pretrained on ImageNet.
- Modular, reproducible engineering: training, evaluation, automated testing, and web deployment.

### Slide 4: MobileNetV2 Architecture & Inverted Residuals
- Diagram of standard convolution vs Depthwise Separable Convolution.
- Computational savings: ~89% fewer multiply-accumulate operations.
- Inverted residual bottlenecks: Expand &rarr; Depthwise Filter &rarr; Linear Projection.

### Slide 5: Custom Classification Head Design
- Global Average Pooling (condensing spatial tensor to 1280 features).
- Batch Normalization & Dropout (0.3) for regularization.
- Dense intermediate layer (128 units, ReLU) and Dropout (0.2).
- Softmax output layer generating normalized categorical probabilities.

### Slide 6: Dataset Pipeline & Leakage Safeguards
- Data acquisition & licensing compliance (Kaggle CDLA / Stanford academic benchmarks).
- Zero-leakage verification between Train, Validation, and Test partitions.
- Online training data augmentation (flips, rotations, zooms, contrast shifts).

### Slide 7: Two-Phase Training Strategy
- Phase 1: Feature Extraction (Backbone frozen, Adam lr = 1e-3, 15 epochs).
- Phase 2: Upper-layer Fine-Tuning (Unfreezing layers &ge; 100, Adam lr = 1e-5, 10 epochs).
- Dynamic callbacks: `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`.

### Slide 8: Quantitative Evaluation & Test Set Results
- Test Set Accuracy, Macro Precision, Macro Recall, Macro F1-Score.
- Training loss & accuracy convergence graphs across epochs.

### Slide 9: Confusion Matrix & Error Analysis
- Visual confusion matrix heatmap.
- Analysis of primary inter-class confusions (e.g. models sharing similar camera island shapes).
- Impact of lighting and viewing angles.

### Slide 10: Full-Stack Web Application Architecture
- Flask REST API (`/api/predict`, `/api/model-status`, `/api/health`).
- Responsive modern UI (HTML5, CSS3, vanilla JS).
- In-memory model persistence preventing redundant disk reload bottlenecks.

### Slide 11: Live Demonstration / Workflow Walkthrough
- Step-by-step walkthrough: Uploading an image &rarr; Preprocessing &rarr; Inference &rarr; Top-3 probability output with confidence bars.
- Transparent status handling (untrained vs trained state).

### Slide 12: Conclusion, Limitations & Future Work
- Key takeaways & competencies demonstrated.
- Limitations: Opaque phone cases, extreme lighting glare, minor sub-model differences.
- Future work: TFLite mobile compilation, multi-angle camera fusion.
- Q&A Invitation.

