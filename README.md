# Brain-Tumor-Classification-and-Detection

Introduction

This project was developed as part of IE 7615: Neural Network and Deep Learning to build and optimize models for brain tumor classification. The primary objective is to accurately classify MRI images into four categories: glioma, meningioma, no-tumor, and pituitary tumor. This research aims to enhance early diagnosis and improve medical decision-making through deep learning techniques.

Features

🧠 Deep Learning-based Image Classification: Utilizes CNN and transfer learning models to classify brain tumors.

📊 Image Augmentation: Enhances model generalization by applying augmentation techniques.

🔧 Hyperparameter Tuning: Optimizes model performance using advanced tuning techniques.

🔍 Comparative Analysis of Models: Evaluates CNN, ResNet50, VGG19, SVM, and Inception V3.

🚀 Scalability and Deployment Readiness: Ensures adaptability for real-world applications using PyTorch.

Dataset

📂 Source: Publicly available MRI dataset for brain tumor classification.

📏 Size: Includes thousands of labeled MRI images across four tumor categories.

⚙️ Preprocessing Steps:

✅ Image resizing and normalization.

✅ Data augmentation (flipping, rotation, contrast enhancement).

✅ Splitting into training, validation, and test sets.

Technologies Used

🖥️ Programming Languages: Python

📚 Deep Learning Frameworks: PyTorch, TensorFlow

🔢 Machine Learning Libraries: Scikit-Learn

🏗️ Model Architectures: CNN, ResNet50, VGG19, Inception V3, Support Vector Machines (SVM)

🖼️ Image Processing: OpenCV, PIL

📊 Visualization: Matplotlib, Seaborn

Installation

Clone the repository:

git clone https://github.com/yourusername/brain-tumor-classification.git
cd brain-tumor-classification

Install dependencies:

pip install -r requirements.txt

Usage

Run the model training script:

python train.py --model VGG19 --epochs 50 --batch_size 32

Perform inference on a sample MRI image:

python predict.py --image sample_mri.jpg --model VGG19

Results & Insights

✅ CNN Model: Achieved 99% training accuracy and 95% validation accuracy.

✅ Image Augmentation: Improved generalization, resulting in 97% training accuracy and 95% test accuracy.

✅ Transfer Learning:

VGG19: 99% training accuracy, 98% validation accuracy.

ResNet50: 95% training accuracy, 94% validation accuracy.

🏆 Best Model: VGG19 with 94.12% detection accuracy after hyperparameter tuning.

Future Work & Improvements

🔬 Integration with Medical Diagnosis Pipelines.

📊 Testing on Larger and More Diverse Datasets.

🧐 Implementation of Explainable AI (XAI) Techniques for better model interpretability.

🌐 Deployment as a Web or Mobile Application for real-world accessibility.

Contributions & Contact

💡 Contributions are welcome! Feel free to fork the repository and submit pull requests.

For any queries, contact Aditya Velapurkar at adityavelapurkar1002@gmail.com.
