
import pandas as pd
import numpy as np
from dotenv import load_dotenv


from langchain_chroma import Chroma

import gradio as gr
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Load book dataset
books = pd.read_csv("books_with_emotions.csv")

books["large_thumbnail"] = books["thumbnail"].fillna("") + "&fife=w800"

books["large_thumbnail"] = np.where(
    books["thumbnail"].isna(),
    "cover_book_404.jpg",
    books["large_thumbnail"]
)

# Load existing vector database
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db_books = Chroma(
    persist_directory="./book_vector_db",
    embedding_function=embedding_model,
    collection_name="books"
)


def retrieve_semantic_recommendation(
    query: str,
    category: str = "ALL",
    tone: str = "ALL",
    initial_top_k: int = 50,
    final_top_k: int = 16
) -> pd.DataFrame:

    # Semantic similarity search
    recs = db_books.similarity_search_with_score(
        query,
        k=initial_top_k
    )

    # Extract ISBN from retrieved documents
    books_list = [
        int(doc.page_content.strip('"').split()[0])
        for doc, score in recs
    ]

    # Match retrieved ISBN with book dataset
    book_recs = books[
        books["isbn13"].isin(books_list)
    ].copy()

    # Filter category
    if category != "ALL":
        book_recs = book_recs[
            book_recs["simple_categories"] == category
        ]

    # Sort by emotional tone
    if tone == "Happy":
        book_recs = book_recs.sort_values(
            by="joy", ascending=False
        )

    elif tone == "Surprising":
        book_recs = book_recs.sort_values(
            by="surprise", ascending=False
        )

    elif tone == "Angry":
        book_recs = book_recs.sort_values(
            by="anger", ascending=False
        )

    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(
            by="fear", ascending=False
        )

    elif tone == "Sad":
        book_recs = book_recs.sort_values(
            by="sadness", ascending=False
        )

    return book_recs.head(final_top_k)


def recommend_books(query, category, tone):

    if not query.strip():
        return []

    recommendation = retrieve_semantic_recommendation(
        query, category, tone
    )

    res = []

    for _, row in recommendation.iterrows():

        # Process description
        description = str(row["description"])
        truncated_description = " ".join(
            description.split()[:30]
        ) + "..."

        # Process authors
        authors_split = str(row["authors"]).split(";")

        if len(authors_split) == 2:
            authors_str = (
                f"{authors_split[0]} and {authors_split[1]}"
            )

        elif len(authors_split) > 2:
            authors_str = (
                f"{', '.join(authors_split[:-1])} "
                f"and {authors_split[-1]}"
            )

        else:
            authors_str = authors_split[0]

        # Create caption
        caption = (
            f"{row['title']} by {authors_str}: "
            f"{truncated_description}"
        )

        # Add image and caption
        res.append((
            row["large_thumbnail"],
            caption
        ))

    return res


# Dropdown choices
categories = ["ALL"] + sorted(
    books["simple_categories"].dropna().unique().tolist()
)

tones = [
    "ALL",
    "Happy",
    "Surprising",
    "Angry",
    "Suspenseful",
    "Sad"
]


# Gradio Dashboard
with gr.Blocks(theme=gr.themes.Glass()) as dashboard:

    gr.Markdown("# Semantic Book Recommendations")

    with gr.Row():

        user_query = gr.Textbox(
            label="Please enter a description of a book:",
            placeholder="e.g., A story about a hero"
        )

        category_dropdown = gr.Dropdown(
            choices=categories,
            label="Select a category:",
            value="ALL"
        )

        tone_dropdown = gr.Dropdown(
            choices=tones,
            label="Select an emotional tone:",
            value="ALL"
        )

        submit_button = gr.Button("Find Recommendations")

    gr.Markdown("## Recommendations")

    output = gr.Gallery(
        label="Recommended Books",
        columns=4,
        rows=4
    )

    submit_button.click(
        fn=recommend_books,
        inputs=[
            user_query,
            category_dropdown,
            tone_dropdown
        ],
        outputs=output
    )


if __name__ == "__main__":
    dashboard.launch()
