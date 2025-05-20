Scene Classification with EfficientNet: A Computer Engineering Student’s Deep Learning Journey

Rabia Yazıcı

“In the beginning, everything was very complicated… But now when I look back, every line of code, every error message, every success brought me to this point today.”

Hello! I am a computer engineering student. In this blog, I would like to share with you the scene classification project I developed as part of our deep learning course. This project tested not only my technical skills, but also my problem-solving approach, my patience, and the power of my passion for learning. If you are also interested in artificial intelligence and image classification projects, this journey may inspire you.

Project Goal

Our goal was to develop a model that automatically classifies images into six different scene categories. Our categories were:

🏢 Buildings🌲 Forest❄️ Glacier🏔️ Mountain🌊 Sea🛣️ Street

While this task may seem simple at first glance, things get serious when concepts like deep learning models, data processing, and overfitting come into play…

First Meeting with the Dataset

The dataset was quite large. Each class contained thousands of images, and the sizes and color ranges of these images were different from each other. The first thing I had to learn was how to make this data suitable for the model.

Data Preprocessing Steps

Size Standardization: I made all the images 224x224. EfficientNet’s input format required this.

Normalization: I normalized the RGB channels with ImageNet’s mean and standard deviations.

Data Augmentation: I tried to get the model used to diversity with methods such as


Model Selection: EfficientNet Why?

My choice was EfficientNet-B0. Why?

· My computer was not powerful. EfficientNet works with fewer parameters.

· It offered a high accuracy rate.

· Training time was shorter. Deadline was approaching quickly!

EfficientNet is a model that stands out with its balanced depth, width and resolution scaling.

Transfer Learning: Leveraging Ready-Made Knowledge

Transfer learning is taking a model that has been trained on large datasets and adapting it to your own dataset. With this method:

· I froze the first layers

· I made only the last few blocks trainable

· I restructured the output layer


This approach allowed the model to generalize well even with low data.

Fighting Overfitting: A Common Enemy

In the first attempts, the model was very successful on the training data but was failing on the test data. In other words, I was experiencing overfitting.

Precautions I took:

· Dropout: I applied a dropout of 0.

· I trained fewer layers

· I kept the batch size small (16), my computer could not handle the larger one

· I increased the variety of data augmentation

Training Process: Learning from Mistakes

Optimizer: Adam

I used the Adam optimization algorithm.

I tried it with lr=0.01 at first, but the loss skyrocketed. Then I got more stable results with 0.001.

Loss Function: Cross Entropy

The loss function I used:

L = -∑(y_i * log(p_i))

Although it looks complicated at first, it’s actually very intuitive: The model gets penalized for making wrong predictions.

Performance Improvement: When Pushing the Limits

· Memory Issues

I didn’t have a GPU, so I had to use memory efficiently.

My Solutions:

Small batch size

Gradient checkpointing

Train only the last layers

· Speed ​​Optimizations

When approaching the deadline:

Reduced the number of epochs to 3

Increased the dataloader performance

Optimized the training on the CPU

Results: Exceeded My Expectations

My test results were quite satisfactory:

90%+ accuracy in buildings and streets categories

Confusions between mountain and glacier classes (thanks to Confusion Matrix, I saw this)

Forest-Sea confusions were also due to similarity in color tones

Thanks to Confusion Matrix, I saw the areas where the model was struggling and understood the reasons better.

Student Notes

My biggest lessons from this process:

· Making mistakes is part of learning.

· You can learn something at every step.

· Optimizations, rather than the complexity of the model, determine the success of the project.

Source Code and Hugging Face

You can access the source code of the project on GitHub. I took care to add explanations and comment lines in the code. I would love for it to be useful to other students.

GitHub - rabiayazici/SceneClassification
Contribute to rabiayazici/SceneClassification development by creating an account on GitHub.
github.com

You can also access the Hugging Face from the below URL:

https://huggingface.co/spaces/ryazici/NaturalSceneClassification

Final Word: A Success Story

This project was not just an assignment for me. It was a process of learning how to learn, being patient, and achieving big things with small steps.

At first, I said, “I can’t do this job,” but now when I look back, I can say:

“I can. With my mistakes, my effort, my determination… I learned and succeeded.”
