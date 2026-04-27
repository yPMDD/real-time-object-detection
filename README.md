# Real-Time Object Detection and Recognition with Deep Learning

## Project Overview
This project focuses on real-time object detection and recognition using deep learning, specifically targeting five classes: person, bottle, phone, mouse, and the EMSI institutional logo.

## Selected Classes
- **person**: Common COCO category.
- **bottle**: Common COCO category.
- **phone**: Common COCO category.
- **mouse**: Common COCO category.
- **EMSI_logo**: Custom class for institutional recognition.

## Methodology
- **Architecture**: YOLOv8 (Ultralytics).
- **Dataset Strategy**: Fine-tuning a pre-trained detector on a mixed dataset (COCO subset + custom EMSI logo data).
- **Inference**: Real-time webcam demo and backend API (FastAPI).

## Folder Structure
(See the project plan for details)

## Resume Highlights
- Developed a real-time object detection system using CNN-based YOLO models and OpenCV.
- Built and annotated a custom EMSI logo dataset and merged it with COCO-style object classes.
- Structured the project with reproducible training, evaluation, API exposure, and demo modules.
