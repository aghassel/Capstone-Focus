# AI-Enhanced Glasses for Individuals with Visual and Auditory Impairments

## Overview

This project aims to develop **AI-Enhanced Glasses**, codenamed **Focus**, designed to support individuals with visual and auditory impairments. The glasses offer real-time assistance for communication, navigation, and daily tasks, promoting greater independence and enhancing the quality of life for users. The design integrates state-of-the-art deep learning models and computer vision technologies to bridge the gap for individuals with sensory impairments.

<div align="center">
  <a href="https://github.com/aghassel/Focus-AI-Glasses/blob/main/images/prototype.png">
    <img src="https://github.com/aghassel/Focus-AI-Glasses/blob/main/images/prototype.png" alt="Prototype" width="500px">
  </a>
</div>

---

## Key Features

<div align="center">
  
| Feature                         | Description                                                   |
|----------------------------------|---------------------------------------------------------------|
| **Speech Translation & Transcription** | Converts speech to text and supports 133 languages.            |
| **Sign Language Interpretation** | Translates ASL gestures into text and speech in real-time.     |
| **Navigation Assistance**        | Detects obstacles and provides alerts for oncoming objects.    |
| **Health Monitoring**            | Tracks heart rate and SpO2 levels in real-time.                |
| **Real-time Q&A using LLMs**     | Allows users to ask questions and get immediate responses.     |
| **Battery Life**                 | Provides up to 5 hours of continuous usage.                    |

</div>

---

## Problem Statement

Globally, **1.1 billion** people experience some form of vision loss, with **285 million** having low vision or blindness. Additionally, **466 million** individuals suffer from hearing loss, while approximately **1 million** in the U.S. use American Sign Language (ASL) as their primary mode of communication.

<div align="center">
  <a href="https://github.com/aghassel/Focus-AI-Glasses/blob/main/images/stats_1.png" style="display:inline-block;">
    <img src="https://github.com/aghassel/Focus-AI-Glasses/blob/main/images/stats_1.png" alt="Stats 1" height="400px">
  </a>
  <a href="https://github.com/aghassel/Focus-AI-Glasses/blob/main/images/stats_2.jpg" style="display:inline-block;">
    <img src="https://github.com/aghassel/Focus-AI-Glasses/blob/main/images/stats_2.jpg" alt="Stats 2" height="400px">
  </a>
</div>


---

## Solution Focus

The glasses are tailored to enhance the daily life of individuals with visual and auditory impairments by offering key features such as real-time performance and multimodal AI capabilities, ensuring users can access necessary support at any moment.

### Key Features:
- **Lightweight and Ergonomic Design**: Built for comfort with a five-hour battery life.
- **Speech Translation & Transcription**: Converts speech into text and offers multilingual translation support.
- **Sign Language Interpretation**: Translates ASL gestures into text and spoken words in real-time.
- **Navigation Assistance**: Alerts users to nearby obstacles and detects cars using object detection technology.
- **Health Monitoring**: Tracks health metrics like heart rate and SpO2 levels.
- **Real-Time Q&A**: Uses advanced language models for answering questions.
- **Real-Time Scheduling & Reminders**: Helps users manage daily tasks with reminders and schedules.
- **Weather & Date/Time Display**: Contextual information based on the user’s region.

<div align="center">
  <a href="https://github.com/aghassel/Focus-AI-Glasses/blob/main/images/architecture.png">
    <img src="https://github.com/aghassel/Focus-AI-Glasses/blob/main/images/architecture.png" alt="Prototype" width="600px">
  </a>
</div>

---

## Hardware Components

The glasses are designed with a robust yet lightweight platform using the following components:

- **Raspberry Pi Zero 2W**
- **1200mAh Pi Sugar (Battery Pack)**
- **Transparent OLED Screen**
- **Pi Spy Camera**
- **3D-Printed PLA Plastic Frame**
- **Biometric Sensors** for heart rate and SpO2 monitoring (MAX30100)
- **Texas Instruments USB Microphone**

The design overcomes power consumption and weight balance challenges by extending wires and optimizing battery placement to ensure comfort for long-duration wear.

---

## Software Architecture

### Core Technologies

- **Object Detection**: Uses **YOLOv5** model, trained on the COCO dataset for real-time object detection, achieving 95% accuracy on the Kitti dataset with 22ms inference speed.
- **Speech-to-Text & Translation**: Powered by **Google Cloud** with support for 133 languages, allowing fast and accurate speech recognition.
- **Text-to-Speech (TTS)**: Utilizes **OpenAI’s Whisper API** for realistic speech synthesis.
- **Large Language Models (LLMs)**: Utilizes **Llama2-70B** model for real-time query response, processing up to 350 tokens per second.
- **ASL Recognition**: Recognizes up to **250 ASL signs** using an ensemble of 1D Convolutional Neural Networks (1DCNN) and Transformer models, achieving **89% accuracy** with 17ms latency.

  <div align="center">
  <a href="https://github.com/aghassel/Focus-AI-Glasses/blob/main/images/asl_architecture.png">
    <img src="https://github.com/aghassel/Focus-AI-Glasses/blob/main/images/asl_architecture.png" alt="Prototype" width="500px">
  </a>
</div>

### Car Detection

Utilizing the **YOLOv5 Architecture**, the glasses can detect nearby cars and highlight their locations with bounding boxes, providing real-time assistance to users in navigating their surroundings safely. This feature achieved **95% accuracy** using the **Intersection Over Union** metric on the **Kitti Dataset**.

<div align="center">
  <a href="https://github.com/aghassel/Focus-AI-Glasses/blob/main/images/cv_architecture.png">
    <img src="https://github.com/aghassel/Focus-AI-Glasses/blob/main/images/cv_architecture.png" alt="Prototype" width="500px">
  </a>
</div>

---

## Construction & Design

The frame of the glasses is constructed using **3D-printed PLA plastic**, broken into multiple components (temples, hinges, and bridge) to facilitate the design. The challenge of balancing the weight from the battery pack was solved by extending wires and optimizing the placement of components for comfort and usability.

---

## Cost Breakdown

<div align="center">

| Component                   | Cost   |
|------------------------------|--------|
| Raspberry Pi Zero 2W          | $28.99 |
| 1200mAh Pi Sugar              | $45.99 |
| Texas Instruments USB Microphone | $10.00 |
| OLED Screen                   | $35.00 |
| Kuuleyn Raspberry Pi Camera   | $10.00 |
| PLA Plastic                   | $0.30  |
| **Total**                     | **$130.28** |

</div>

---

## Evaluation Matrix

Each feature of the glasses is evaluated based on functionality, performance, usability, and innovation. Here's a summary of the evaluation criteria:

<div align="center">

| Feature                       | Functionality | Performance | Usability | Innovation |
|-------------------------------|---------------|-------------|-----------|------------|
| Designed for Accessibility     | 4.5           | N/A         | 4         | 3.5        |
| User-Friendly Design           | 4             | N/A         | 3.5       | 3.5        |
| Battery Life                   | 3.5           | 4           | 3.5       | 4.5        |
| Speech Translation             | 5             | 4.5         | 4         | 4.5        |
| Sign Language Translation      | 5             | 4           | 3.5       | 4          |
| Navigation Assistance          | 4.5           | 4.5         | 4         | 4          |
| Instant Q&A using LLMs         | 4             | 5           | 4.5       | 5          |
| Real-time Processing           | 5             | 5           | 4         | 5          |

</div>

---

## Future Work

### Planned Enhancements:
- **Custom Printed Circuit Board (PCB)**: To reduce the size of the glasses and improve aesthetics.
- **Advanced Computing**: Transition to **NVIDIA Jetson** for enhanced AI performance.
- **Expanded Sign Language Support**: Extend the model to interpret full sentences and additional languages.
- **Display Upgrades**: Explore full glass lenses or micro projectors for enhanced user experience.
- **Car Detection Improvements**: Expand the detection model to include sidewalk identification for enhanced navigation assistance.

---

## References

- [1] Canadian Association of Optometrists. (2020). World Vision Facts from IAPB.
- [2] Government of Canada. (2021). "Hearing health of Canadian adults."
- [3] R. E. Mitchell and T. A. Young. "How Many People Use Sign Language?" The Journal of Deaf Studies and Deaf Education, 2023.
- [4] Google Isolated Sign Language Recognition. (2023). Kaggle.
- [5] Sohn H. (2023). Google Isolated Sign Language Recognition—1st Place Solution.
- [6] Geiger A., Lenz P., Stiller C., & Urtason R. (2013). "Vision meets Robotics: The KITTI Dataset." IJRR.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details.
