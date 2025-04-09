import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from main import get_dataframe
import openai

st.set_page_config(layout='wide')
st.title("Amazon Review Sentimental Analysis")

col1, col2 = st.columns([2, 1])
with col1:
    link = st.text_input("Enter your product link:", placeholder="https://www.amazon.in/Offbeat-Bluetooth-Wireless-Rechargeable-connectivity/dp/B0CB1FYVJZ/")

def summarize_reviews(reviews, batch_size=10):
    """Generates a very brief 3-4 line summary of the product reviews."""
    client = openai.OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", "your_api_key_here"))

    summaries = []
    for i in range(0, len(reviews), batch_size):
        prompt = (
            "Summarize the following product reviews in **3-4 concise sentences**. "
            "Focus on key themes like quality, durability, and user satisfaction:\n"
            + "\n".join(reviews[i : i + batch_size])
        )
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert product review summarizer. Keep it short and to the point."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5 
            )
            summaries.append(response.choices[0].message.content.strip())
        except openai.OpenAIError as e:
            st.error(f"OpenAI API Error: {e}")
            return "Error in generating summary."

    return " ".join(summaries)[:500] 

if st.button("Submit"):
    with st.spinner("Fetching and Analyzing the data..."):
        df = get_dataframe(link)
    
    st.dataframe(df)
    
    sentiment_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    df['sentiment_score'] = df['Sentiment'].map(sentiment_mapping)
    avg_sentiment = df['sentiment_score'].mean()
    overall_sentiment = "Positive" if avg_sentiment > 0 else "Negative" if avg_sentiment < 0 else "Neutral"
    
    st.header(f"**Overall Sentiment:** {overall_sentiment}")
    st.header(f"Score: {avg_sentiment:.2f}")
    
    reviews = df['reviewTitle'].fillna('') + " " + df['reviewDescription'].fillna('')
    review_summary = summarize_reviews(reviews.tolist())
    
    st.header("AI-Generated Review Summary:")
    st.write(review_summary)
    
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["year_month"] = df["date"].dt.to_period("M")

    sales_trend = df["year"].value_counts().sort_index()
    sentiment_trend = df.groupby("year")["Sentiment"].value_counts(normalize=True).unstack() * 100
    monthly_reviews = df["month"].value_counts().sort_index()
    rating_trend = df.groupby("year_month")["ratingScore"].mean()
    sentiment_counts = df['Sentiment'].value_counts()

    def generate_positive_wordcloud():
        positive_text = " ".join(df[df["Sentiment"] == "Positive"]["reviewTitle"] + " " + df[df["Sentiment"] == "Positive"]["reviewDescription"])
        if positive_text.strip():
            return WordCloud(width=800, height=400, background_color="white").generate(positive_text)
        return None

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sales_trend.plot(kind="line", marker="o", color="b", ax=ax1)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Total Reviews (Proxy for Sales)")
    ax1.set_title("Yearly Sales Trend")
    ax1.grid(True)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sentiment_trend.plot(kind="bar", stacked=True, color=['red', 'blue', 'green'], ax=ax2)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Percentage of Sentiments")
    ax2.set_title("Yearly Sentiment Distribution")
    ax2.legend(title="Sentiment")

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    monthly_reviews.plot(kind="bar", color="skyblue", ax=ax3)
    ax3.set_xlabel("Month")
    ax3.set_ylabel("Total Reviews")
    ax3.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], rotation=45)
    ax3.set_title("Monthly Review Trends")
    ax3.grid(axis="y")

    wordcloud_fig = generate_positive_wordcloud()
    if wordcloud_fig:
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        ax4.set_title("Word Cloud of Positive Reviews")
        ax4.imshow(wordcloud_fig, interpolation="bilinear")
        ax4.axis("off")

    fig5, ax5 = plt.subplots(figsize=(6, 6))
    ax5.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['red', 'blue', 'green'])
    ax5.axis('equal')
    plt.title("Overall Sentiment Distribution")

    fig6, ax6 = plt.subplots(figsize=(8, 5))
    rating_trend.plot(kind="line", marker="o", color="blue", ax=ax6)
    ax6.set_xlabel("Year-Month")
    ax6.set_ylabel("Average Rating")
    ax6.set_title("Sentiment Trend Over Time")
    plt.xticks(rotation=45)

    col1, col2 = st.columns([2, 2])
    with col1: st.pyplot(fig1)
    with col2: st.pyplot(fig2)
    with col1: st.pyplot(fig3)
    if wordcloud_fig:
        with col2: st.pyplot(fig4)
    with col1: st.pyplot(fig5)
    with col2: st.pyplot(fig6)
