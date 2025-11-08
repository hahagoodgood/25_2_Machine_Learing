# SeeSpeakHangle Project Proposal  
> **Project Name:** SeeSpeakHangle  
> **Project Topic:** Development of an AI model that automatically generates Korean captions from image inputs  
> **Dataset:** AI Hub Korean Image Caption Dataset  

---

## 1. Project Overview  

### 1.1 Objective  
- This project aims to develop an AI model that analyzes an input image and automatically **generates a natural Korean sentence (caption)** describing it.  
- Example:  
  - **Input:** An image of an airplane on a runway  
  - **Output:** “비행기가 공항 활주로 위에 서 있다.” (“An airplane is standing on the airport runway.”)

### 1.2 Background and Motivation  
- Recently, **Multimodal Artificial Intelligence (AI)** has been increasingly utilized in various fields such as image captioning and video subtitle generation through the **integrated learning of visual and linguistic information**.  
- This project aims to understand and implement the core concepts and technical architecture of **multimodal AI** by developing and experimenting with an image-to-text caption generation model.

---

## 2. Dataset Information  

| Category | Description |
|-----------|-------------|
| **Dataset Name** | Korean Image Caption Dataset |
| **Source** | [AI Hub](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=261) |
| **Composition** | Approximately 120,000 COCO images with 600,000 Korean captions |
| **Size** | About 116MB (JSON text) / 19GB for COCO 2014 images |
| **Format** | JSON: image file name + list of Korean captions |
| **Example** | `"caption_ko": "두 명의 사람이 테니스를 치고 있다."` (“Two people are playing tennis.”) |
| **Reference Dataset** | Microsoft COCO 2014 (train/validation) |

---

## 3. Model Design and Considerations  

| Component | Description |
|------------|-------------|
| **Encoder** | Extracts visual features from images |
| **Decoder** | Generates Korean text captions from encoded features |
| **Tokenizer** | SentencePiece (BPE-based) or KoNLPy morphological tokenization |
| **Loss Function** | Cross Entropy Loss (with Teacher Forcing) |
| **Optimization Algorithm** | Adam Optimizer with learning rate scheduler (ReduceLROnPlateau) |
| **Evaluation Metrics** | BLEU, METEOR, CIDEr – standard metrics for text generation performance |

---

## 4. Expected Outcomes and Applications  

| Aspect | Description |
|---------|-------------|
| **Technical Aspect** | Acquire core AI concepts such as image feature extraction, natural language generation, and multimodal integration |
| **Practical Applications** | Image search, automatic photo description, assistive technology for the visually impaired, automatic SNS captions |
| **Educational Aspect** | Hands-on experience with multimodal learning using PyTorch and performance optimization techniques |

---

## 5. Project Goals  

### 5.1 Quantitative Targets  

| Metric | Target |
|---------|--------|
| **BLEU** | ≥ 0.50 |
| **METEOR** | ≥ 0.40 |

### 5.2 Deliverables  
  - Visualization samples showing image inputs and generated captions  
  - Training and evaluation logs and graphs  
  - Model architecture and data flow diagrams  
  - Final project report and presentation materials  

---

**Author:** 김동혁
**ID:** 202547002
**Date:** 2025-11-08  
