# 🛍️ Amazon Review Sentiment Analysis

## 🎯 Project Overview  
This project analyzes customer sentiments from **Amazon product reviews** to better understand consumer feedback. It classifies reviews as **Positive**, **Neutral**, or **Negative**, and presents insights using interactive visualizations and AI-generated summaries. The app is built using **Streamlit** and integrates **OpenAI's GPT model** for review summarization.

📹 **Video Demo**: [https://drive.google.com/your-video-link](https://drive.google.com/file/d/1mpNsm5VRcIFFWCoUgbvt1eqq3TTFUv5h/view?usp=sharing)

## 📌 Introduction  
The project focuses on automating sentiment classification of Amazon product reviews and providing visual insights to help brands and customers understand sentiment trends over time.

---

## 🎯 Objectives  
- Extract and process product reviews from Amazon using scraping tools.  
- Perform sentiment analysis using NLP techniques (VADER).  
- Generate insightful graphs and statistics for deeper understanding.  
- Summarize reviews using **OpenAI's GPT model** for easier interpretation.

---

## 🌟 Features  
- 🔹 Sentiment Classification: Positive, Neutral, Negative  
- 🧠 GPT-Powered Review Summarizer  
- 📈 Yearly Sales Trends (based on review count)  
- 📊 Yearly Sentiment Distribution  
- 📆 Monthly Review Trends  
- ☁️ Word Cloud of Positive Reviews  
- 🥧 Overall Sentiment Pie Chart  
- 📉 Sentiment Trends Over Time  
- 🧩 Interactive Dashboard built using Streamlit

---

## 🏗️ System Design  

### 📐 Architecture  
Pipeline-based structure:
1. **Data Collection**: Web scraping using `apify_client`.  
2. **Preprocessing**: Cleaning & formatting using `pandas`.  
3. **Sentiment Analysis**: Using `VADER` from `nltk`.  
4. **Visualization**: Built with `matplotlib`, `seaborn`, and `wordcloud`.  
5. **Summarization**: Integrated `openai` API for review overviews.

### 🧱 Data Structure  
```python
{
  "reviewTitle": "string",
  "reviewDescription": "string",
  "sentiment": "Positive/Neutral/Negative",
  "ratingScore": "float",
  "date": "datetime"
}
