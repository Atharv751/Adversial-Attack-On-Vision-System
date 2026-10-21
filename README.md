# Adversarial Attacks on Vision Systems

> **Analyzing the vulnerability of CNN-based image classification models to adversarial perturbations and evaluating their robustness against common attacks.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## About the Project

Deep learning models used in computer vision can achieve high accuracy on normal images but may behave incorrectly when presented with carefully crafted adversarial examples.

This project studies this problem using a **Convolutional Neural Network (CNN)** trained on the **CIFAR-10 dataset**.

The project focuses on three main areas:

- Generating adversarial examples using **FGSM**
- Evaluating stronger iterative attacks using **PGD**
- Improving model robustness using **Adversarial Training**

The objective is to understand how small changes to an input image can affect a model's prediction and how adversarial training can improve its resistance to such attacks.

---

## Objectives

The main objectives of this project are:

1. Train a CNN for image classification using the CIFAR-10 dataset.
2. Establish the model's performance on clean, unmodified images.
3. Generate adversarial examples using the **Fast Gradient Sign Method (FGSM)**.
4. Generate stronger adversarial examples using **Projected Gradient Descent (PGD)**.
5. Analyze the effect of different perturbation strengths on model accuracy.
6. Compare the effectiveness of FGSM and PGD attacks.
7. Apply adversarial training as a defense mechanism.
8. Compare the robustness of the standard and adversarially trained models.

---

## Dataset

The project uses the **CIFAR-10** dataset.

CIFAR-10 contains 60,000 color images divided into 10 classes:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

Each image has a resolution of **32 × 32 pixels** with three color channels.

---

## Methodology

The overall workflow of the project is:

```text
CIFAR-10 Dataset
       │
       ▼
   CNN Training
       │
       ▼
Clean Image Evaluation
       │
       ▼
Adversarial Attack Generation
       │
   ┌───┴────┐
   ▼        ▼
 FGSM      PGD
   │        │
   └───┬────┘
       ▼
Robustness Evaluation
       │
       ▼
Adversarial Training
       │
       ▼
Final Robustness Comparison
```

---

## Adversarial Attacks

### 1. FGSM

**Fast Gradient Sign Method (FGSM)** is a single-step adversarial attack.

It uses the gradient of the loss with respect to the input image to determine the direction in which the input should be modified.

The adversarial image can be represented as:

```text
x_adv = x + ε · sign(∇x L(θ, x, y))
```

where:

- `x` = original image
- `x_adv` = adversarial image
- `ε` = perturbation strength
- `L` = model loss
- `θ` = model parameters
- `y` = correct label

FGSM is computationally efficient because it generates the perturbation in a single gradient step.

---

### 2. PGD

**Projected Gradient Descent (PGD)** is an iterative adversarial attack.

Instead of making one update like FGSM, PGD repeatedly updates the image while keeping the perturbation within a predefined limit.

```text
x(t+1) = ΠB(x,ε)[xt + α · sign(∇x L)]
```

PGD is generally considered a stronger attack because it has multiple optimization steps.

---

## Adversarial Training

To investigate a possible defense, the project also uses **adversarial training**.

Instead of training the CNN only on clean images, adversarial examples are incorporated into the training process.

```text
Clean Training Images
        +
Adversarial Examples
        │
        ▼
   CNN Training
        │
        ▼
Robust Model
```

The resulting model can then be evaluated against both clean and adversarial images.

---

## Experiments

The project evaluates the model under different conditions.

### Clean Image Evaluation

The baseline CNN is evaluated using the original CIFAR-10 test images.

### FGSM Evaluation

The model is tested against FGSM-generated adversarial examples using different perturbation strengths.

### PGD Evaluation

The model is evaluated against iterative PGD attacks with different configurations and numbers of attack steps.

### Robustness Comparison

The results of the attacks are compared to understand:

- How model accuracy changes as perturbation increases
- Which attack is more effective
- How model confidence changes
- Which CIFAR-10 classes are more affected
- How adversarial training changes robustness

---

## Results

The project generates quantitative and visual results for analyzing adversarial robustness.

The main comparisons include:

| Evaluation | Standard CNN | Adversarially Trained CNN |
|---|---:|---:|
| Clean Images | Measured | Measured |
| FGSM Attack | Measured | Measured |
| PGD Attack | Measured | Measured |

> **Note:** The values in the table should be filled with the results obtained after running the experiments rather than using predefined values.

This allows the final results to reflect the actual experimental setup and model performance.

---

## Generated Visualizations

The project produces several visualizations for analyzing the attacks:

| Output | Description |
|---|---|
| `fgsm_examples.png` | Comparison of original and FGSM adversarial images |
| `fgsm_accuracy_curve.png` | Model accuracy across different FGSM perturbation strengths |
| `pgd_examples.png` | Comparison of original, FGSM and PGD examples |
| `fgsm_vs_pgd.png` | Comparison of FGSM and PGD performance |
| `robustness_comparison.png` | Overall robustness comparison |
| `per_class_robustness.png` | Class-wise robustness analysis |
| `confidence_histogram.png` | Change in model confidence under attack |
| `training_comparison.png` | Comparison of standard and adversarial training |

---

## Project Structure

```text
adversarial-cnn/
│
├── utils.py
│   └── CNN architecture, data loaders and shared utilities
│
├── step1_train_cnn.py
│   └── Train the baseline CNN on CIFAR-10
│
├── step2_fgsm_attack.py
│   └── Generate FGSM adversarial examples
│
├── step3_pgd_attack.py
│   └── Generate PGD adversarial examples
│
├── step4_robustness_eval.py
│   └── Evaluate model robustness
│
├── step5_adversarial_training.py
│   └── Train the model using adversarial examples
│
├── Adversarial_Attack_CIFAR10.ipynb
│   └── Complete Google Colab workflow
│
├── requirements.txt
│
└── README.md
```

---

## Technologies Used

- **Python**
- **PyTorch**
- **Torchvision**
- **NumPy**
- **Matplotlib**
- **Google Colab / Jupyter Notebook**

A GPU is recommended for faster training and adversarial attack generation.

---

## Expected Learning Outcomes

Through this project, the following concepts are explored:

- Convolutional Neural Networks
- Image classification
- Gradient-based optimization
- Adversarial examples
- FGSM attacks
- PGD attacks
- Model robustness
- Adversarial training
- Robustness evaluation
- Visualization and analysis of model predictions

---

## References

1. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). *Explaining and Harnessing Adversarial Examples*. ICLR.

2. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). *Towards Deep Learning Models Resistant to Adversarial Attacks*. ICLR.

3. Szegedy, C., et al. (2014). *Intriguing Properties of Neural Networks*. ICLR.

---

## License

This project is released under the **MIT License**.

---

## Project Summary

**Adversarial Attacks on Vision Systems** demonstrates how deep learning-based vision models can be affected by carefully crafted input perturbations and investigates adversarial training as a method for improving model robustness.

The project provides a practical study of the relationship between **model accuracy, adversarial perturbations, attack strength, and robustness**.# Adversial-Attack-On-Vision-System
