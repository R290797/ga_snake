# 🐍 Snake Gen - Version 11.5

## Overview
This project is a **genetic algorithm-based Snake AI** that allows:
- **Manual Play** 🕹️ (Control the snake yourself)
- **AI Training Mode** 🤖 (Train an AI to optimize survival & food collection)
- **Pre-Trained AI Mode** 🧠 (Use pre-trained models with optimized strategies)

The game leverages **`pygame`** for rendering, **`numpy`** for AI calculations, and **a genetic algorithm** for evolving AI behavior.

---

## **Introduction**
Welcome to **AI Snake**, where you can either **train an AI** to play Snake or take control yourself!  
Using **evolutionary principles**, the AI learns **optimal movement strategies**, balancing **exploration and efficiency**.  
This project was developed to demonstrate **machine learning in a game-based environment**.

---

## **📌 Table of Contents**
1. [Game Modes](#game-modes)  
2. [Installation](#installation)  
3. [Controls](#controls)  
4. [AI Training Guide](#ai-training-guide)  
5. [Features](#features)  
7. [Future Improvements](#future-improvements)  
8. [License](#license)  

---

## **📌 Game Modes**
### 1️⃣ Manual Play
- **Control the snake** using the **arrow keys**.
- **Eat food** to grow **without hitting walls or yourself**.
- **Goal:** Survive as long as possible!

### 2️⃣ AI Training Mode
- **Trains an AI** using a **genetic algorithm** 🧬.
- Users **set the number of snakes & generations** before training.
- **Goal:** AI learns to optimize food collection and survival.

### 3️⃣ Pre-Trained AI Mode
- Select from **pre-trained AI models** with **different strategies**:
  - **Hunter:** Aggressively moves towards food.
  - **Strategist:** Balances food-seeking & survival.
  - **Explorer:** Explores more before committing.
  - **Risk-Taker:** Moves quickly but takes high risks.
  - **AI-Mastery:** Advanced AI with optimized weights.
- **Watch how different strategies perform!**

<div align="center">
  <img src="Assets/pre-trained-ai-mastery.gif" alt="Pre-trained GIF" width="400" height="auto">
</div>

---

## **📌 Installation**
### 1️⃣ **Clone the repository**:
   ```bash
   git clone [https://github.com/yourusername/snake_ai.git](https://github.com/R290797/ga_snake)
   cd snake_ai
```

### 2️⃣ Install dependencies:
To install the game, ensure you have Python installed on your system. Then, install the required dependencies using the `requirements.txt` file:

`pip install -r requirements.txt`

### 3️⃣ Run the game:
Once the dependencies are installed, you can run the game by executing the Snake_v11.5.py file.

`python Snake_v11.5.py`

## 📌 Controls

| **Action**      | **Key**        |
|---------------|--------------|
| Move Up       | ⬆️ Arrow Key  |
| Move Down     | ⬇️ Arrow Key  |
| Move Left     | ⬅️ Arrow Key  |
| Move Right    | ➡️ Arrow Key  |
| Select Option | 🖱️ Mouse Click |

---

## 📌 AI Training Guide

### **Optimal Training Settings**

| **Training Type**    | **# Snakes per Gen** | **# Generations** | **Best Use Case**                |
|--------------------|-----------------|---------------|-------------------------------|
| **Quick Test**    | `10-15`          | `5-10`        | Test small changes in AI behavior. |
| **Balanced Training** | `20-30`      | `15-25`       | Good balance of speed & learning. |
| **Deep Optimization** | `40-50`      | `30-50`       | Best AI performance, longer training. |

---

## 📌 Features

✅ **AI Training Using Genetic Algorithms** 🧬  
✅ **Adaptive Mutation in AI Training** 🤖  
✅ **Optimized AI Models for Different Strategies** 🏆  

---
