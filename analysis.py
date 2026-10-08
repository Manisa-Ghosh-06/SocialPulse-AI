import pandas as pd
import numpy as np


def clean_dataframe(df):

    df = df.copy()

    df.columns = [
        str(col).strip()
        for col in df.columns
    ]

    return df


def percentage(value, total):

    if total == 0:
        return 0

    return round(
        (value / total) * 100,
        2
    )


def safe_numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0)


def top_values(
    series,
    limit=10
):

    if series is None:
        return []

    values = (
        series
        .dropna()
        .astype(str)
        .str.strip()
    )

    values = values[
        values != ""
    ]

    if values.empty:
        return []

    counts = (
        values
        .value_counts()
        .head(limit)
    )

    return [
        {
            "name": str(name),
            "count": int(count)
        }
        for name, count
        in counts.items()
    ]


def analyze_overview(
    df,
    mapping
):

    total_posts = 0

    if mapping.get("TEXT"):
        total_posts = len(df)

    total_users = None

    if mapping.get("USER_ID"):

        col = mapping["USER_ID"]

        if col in df.columns:

            total_users = int(
                df[col]
                .dropna()
                .astype(str)
                .nunique()
            )

    return {
        "total_posts": int(total_posts),
        "total_users": total_users
    }


def analyze_sentiment(
    df,
    mapping,
    vectorizer,
    sentiment_model,
    preprocess_text
):

    text_column = mapping.get("TEXT")

    if not text_column:
        return None

    if text_column not in df.columns:
        return None

    texts = (
        df[text_column]
        .fillna("")
        .astype(str)
    )

    processed_texts = [
        preprocess_text(text)
        for text in texts
    ]

    predictions = []

    if len(processed_texts) > 0:

        X = vectorizer.transform(
            processed_texts
        )

        predictions = sentiment_model.predict(
            X
        )

    if len(predictions) == 0:

        return {
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "total": 0
        }

    prediction_series = (
        pd.Series(predictions)
        .astype(str)
        .str.lower()
    )

    positive = int(
        (
            prediction_series
            == "positive"
        ).sum()
    )

    negative = int(
        (
            prediction_series
            == "negative"
        ).sum()
    )

    neutral = int(
        (
            prediction_series
            == "neutral"
        ).sum()
    )

    total = len(
        prediction_series
    )

    return {
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "positive_percentage": percentage(
            positive,
            total
        ),
        "negative_percentage": percentage(
            negative,
            total
        ),
        "neutral_percentage": percentage(
            neutral,
            total
        ),
        "total": total
    }


def analyze_demographics(
    df,
    mapping
):

    result = {}

    age_column = mapping.get("AGE")

    if age_column and age_column in df.columns:

        age = pd.to_numeric(
            df[age_column],
            errors="coerce"
        ).dropna()

        if not age.empty:

            bins = [
                0,
                17,
                24,
                34,
                44,
                54,
                64,
                200
            ]

            labels = [
                "0-17",
                "18-24",
                "25-34",
                "35-44",
                "45-54",
                "55-64",
                "65+"
            ]

            age_groups = pd.cut(
                age,
                bins=bins,
                labels=labels,
                include_lowest=True
            )

            counts = age_groups.value_counts(
                sort=False
            )

            result["age"] = [
                {
                    "name": str(index),
                    "count": int(value)
                }
                for index, value
                in counts.items()
            ]

    gender_column = mapping.get("GENDER")

    if gender_column and gender_column in df.columns:

        result["gender"] = top_values(
            df[gender_column]
        )

    location_column = mapping.get("LOCATION")

    if location_column and location_column in df.columns:

        result["location"] = top_values(
            df[location_column]
        )

    if not result:
        return None

    return result


def analyze_topics(
    df,
    mapping
):

    entity_column = mapping.get("ENTITY")

    if not entity_column:
        return None

    if entity_column not in df.columns:
        return None

    topics = top_values(
        df[entity_column],
        limit=10
    )

    if not topics:
        return None

    return {
        "topics": topics
    }


def analyze_influence(
    df,
    mapping
):

    user_column = mapping.get("USER_ID")
    mentioned_column = mapping.get("MENTIONED_USER")
    followers_column = mapping.get("FOLLOWERS")

    if not user_column or not mentioned_column:
        return None

    if user_column not in df.columns:
        return None

    if mentioned_column not in df.columns:
        return None

    user_values = (
        df[user_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    mentioned_values = (
        df[mentioned_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    network_df = pd.DataFrame({
        "source": user_values,
        "target": mentioned_values
    })

    network_df = network_df[
        (network_df["source"] != "") &
        (network_df["target"] != "")
    ]

    if network_df.empty:
        return None

    edges = (
        network_df
        .groupby(
            ["source", "target"]
        )
        .size()
        .reset_index(
            name="mentions"
        )
        .sort_values(
            "mentions",
            ascending=False
        )
        .head(20)
    )

    result = {
        "edges": edges.to_dict(
            orient="records"
        )
    }

    if followers_column:

        if followers_column in df.columns:

            followers = pd.to_numeric(
                df[followers_column],
                errors="coerce"
            )

            result["followers"] = {

                "total": float(
                    followers
                    .fillna(0)
                    .sum()
                ),

                "average": float(
                    followers.mean()
                )
                if not followers.dropna().empty
                else 0
            }

    return result


def run_analysis(
    df,
    mapping,
    vectorizer,
    sentiment_model,
    preprocess_text
):

    df = clean_dataframe(df)

    result = {}

    result["overview"] = analyze_overview(
        df,
        mapping
    )

    result["sentiment"] = analyze_sentiment(
        df,
        mapping,
        vectorizer,
        sentiment_model,
        preprocess_text
    )

    result["demographics"] = analyze_demographics(
        df,
        mapping
    )

    result["topics"] = analyze_topics(
        df,
        mapping
    )

    result["influence"] = analyze_influence(
        df,
        mapping
    )

    return result
