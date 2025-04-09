from apify_client import ApifyClient
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0
mymodel = SentimentIntensityAnalyzer()

def is_english(text):
    try:
        return detect(text) == "en"
    except:
        return False

def get_dataframe(link):
    client = ApifyClient("apify_api_MTOLwZddCR23c4tfAzfAUwZ4dZVK8L3lDu6b")
    stars = ["oneStar", "twoStar", "threeStar", "fourStar", "fiveStar"]
    reviews_data = []

    for star in stars:
        run_input = {
            "productUrls": [{"url": link}],
            "maxReviews": 80,
            "sort": "helpful",
            "includeGdprSensitive": False,
            "filterByRatings": [star],
            "reviewsUseProductVariantFilter": False,
            "reviewsEnqueueProductVariants": False,
            "proxyCountry": "AUTO_SELECT_PROXY_COUNTRY",
            "scrapeProductDetails": False,
            "reviewsAlwaysSaveCategoryData": False,
            "scrapeQuickProductReviews": True,
            "deduplicateRedirectedAsins": True,
        }
        run = client.actor("R8WeJwLuzLZ6g4Bkk").call(run_input=run_input)
        reviews_data.extend(client.dataset(run["defaultDatasetId"]).iterate_items())

    keys_to_remove = [
        "reviewedIn", "position", "reviewReaction", "userProfileLink", "reviewUrl", "isAmazonVine",
        "reviewImages", "variantAttributes", "totalCategoryRatings", "totalCategoryReviews",
        "reviewCategoryUrl", "filterByRating", "filterByKeyword", "productOriginalAsin", "variantAsin",
        "product", "input", "countryCode", "variant", "userId", "reviewId", "isVerified", "country",
        "productAsin"
    ]
    for review in reviews_data:
        for key in keys_to_remove:
            review.pop(key, None)

    df = pd.DataFrame(reviews_data)
    df['date'] = pd.to_datetime(df['date']).dt.date 
    df.sort_values(by='date', ascending=True, inplace=True)
    df = df[df["reviewDescription"].apply(is_english)]
    df = df[df["reviewTitle"].apply(is_english)]
    df.reset_index(drop=True, inplace=True)

    w_title, w_desc, w_rating = 0.2, 0.4, 0.4
    midpoint, max_rating = 3, 5
    sentiment = []

    for _, row in df.iterrows():
        title_pred = mymodel.polarity_scores(str(row["reviewTitle"]))
        description_pred = mymodel.polarity_scores(str(row["reviewDescription"]))
        norm_rating = (int(row["ratingScore"]) - midpoint) / (max_rating - midpoint)
        final_score = (w_title * title_pred["compound"]) + (w_desc * description_pred["compound"]) + (w_rating * norm_rating)

        if final_score > 0.5:
            sentiment.append("Positive")
        elif -0.2 <= final_score <= 0.5:
            sentiment.append("Neutral")
        else:
            sentiment.append("Negative")

    df["Sentiment"] = sentiment
    df.to_excel("amazon_review_filtered.xlsx", index=False, engine="openpyxl")
    return pd.read_excel('amazon_review_filtered.xlsx')