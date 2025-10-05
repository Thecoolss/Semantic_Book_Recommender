import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import gradio as gr

load_dotenv()


try:
    books = pd.read_csv("books_with_emotions.csv")
    print(f"Successfully loaded {len(books)} books")
    print(f"Columns: {list(books.columns)}")
except Exception as e:
    print(f"Error loading CSV: {e}")
    books = pd.DataFrame()  


required_columns = ['isbn13', 'thumbnail', 'categories', 'title', 'authors', 'description']
emotion_columns = ['surprise', 'joy', 'sadness', 'anger', 'fear', 'disgust']

missing_columns = [col for col in required_columns + emotion_columns if col not in books.columns]
if missing_columns:
    print(f"Warning: Missing columns: {missing_columns}")

books["large_image"] = np.where(books["thumbnail"].isna(), "cover-not-found.jpg", books["thumbnail"] + "&fife=w800")


try:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    
    
    test_result = vectordb.similarity_search("test", k=1)
    print(f"Vector DB test successful. Found {len(test_result)} results")
    if len(test_result) == 0:
        print("WARNING: Vector database appears to be empty!")
except Exception as e:
    print(f"Error with vector database: {str(e)}")
    vectordb = None

def retrieve_semantic_recommendation(query: str, category: str, tone: str, initial_top_k=35, final_top_k=16):
    try:
        print(f"Starting recommendation search for: '{query}'")
        
        if vectordb is None:
            print("Vector database not available")
            return pd.DataFrame()
        
        docs = vectordb.similarity_search(query, initial_top_k)
        print(f"Found {len(docs)} documents from vector search")
        
        if len(docs) == 0:
            print("No documents found in vector search")
            return pd.DataFrame()
        
        books_list = []
        for i in range(len(docs)):  
            try:
                content = docs[i].page_content.strip()
                if content.startswith('"') and content.endswith('"'):
                    content = content[1:-1]  
                isbn_str = content.split()[0]
                isbn = int(isbn_str)
                books_list.append(isbn)
                print(f"Extracted ISBN: {isbn}")
            except (ValueError, IndexError) as e:
                print(f"Error parsing document {i}: {e}, content: {docs[i].page_content}")
                continue
        
        print(f"Successfully extracted {len(books_list)} ISBNs")
        
        if not books_list:
            print("No valid ISBNs extracted")
            return pd.DataFrame()
        
        book_rec = books[books["isbn13"].isin(books_list)]
        print(f"Found {len(book_rec)} matching books in dataset")
        
        if len(book_rec) == 0:
            print("No books found matching the ISBNs")
            return pd.DataFrame()
        

        if category != "All" and category is not None:
            original_count = len(book_rec)
            book_rec = book_rec[book_rec["categories"] == category]
            print(f"After category filter '{category}': {len(book_rec)} books (was {original_count})")
        
        if tone != "All" and tone is not None:
            original_count = len(book_rec)
            if tone == "Surprising" and "surprise" in book_rec.columns:
                book_rec = book_rec.sort_values("surprise", ascending=False)
            elif tone == "Happy" and "joy" in book_rec.columns:
                book_rec = book_rec.sort_values("joy", ascending=False)
            elif tone == "Sad" and "sadness" in book_rec.columns:
                book_rec = book_rec.sort_values("sadness", ascending=False)
            elif tone == "Angry" and "anger" in book_rec.columns:
                book_rec = book_rec.sort_values("anger", ascending=False)
            elif tone == "Fearful" and "fear" in book_rec.columns:
                book_rec = book_rec.sort_values("fear", ascending=False)
            elif tone == "Disgusted" and "disgust" in book_rec.columns:
                book_rec = book_rec.sort_values("disgust", ascending=False)
            print(f"After tone filter '{tone}': {len(book_rec)} books")
        

        final_results = book_rec.head(final_top_k)
        print(f"Returning {len(final_results)} final recommendations")
        return final_results
        
    except Exception as e:
        print(f"Error in retrieve_semantic_recommendation: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def recommend_books(query: str, category: str, tone: str):
    try:
        print(f"\n=== New Recommendation Request ===")
        print(f"Query: '{query}', Category: '{category}', Tone: '{tone}'")
        
        if not query or not query.strip():
            print("Empty query provided")
            return []
        
        recommendation = retrieve_semantic_recommendation(query, category, tone)
        
        if recommendation.empty:
            print("No recommendations found")
            return []
        
        results = []
        print(f"Processing {len(recommendation)} recommendations...")

        for idx, (_, row) in enumerate(recommendation.iterrows()):
            try:
                description = row.get("description", "No description available")
                if pd.isna(description) or description == "":
                    description = "No description available"
                
                desc_words = str(description).split()
                very_truncate_desc = " ".join(desc_words[:30]) + ("..." if len(desc_words) > 30 else "")

                authors_raw = row.get("authors", "Unknown Author")
                if pd.isna(authors_raw) or authors_raw == "":
                    authors = "Unknown Author"
                else:
                    authors_list = str(authors_raw).split(";")
                    if len(authors_list) > 1:
                        authors = f"{', '.join(authors_list[:-1])}, and {authors_list[-1]}"
                    else:
                        authors = str(authors_raw)

                title = row.get("title", "Unknown Title")
                category_info = row.get("categories", "Unknown Category")
                
                caption = f"{title} by {authors}\n\n{very_truncate_desc}\nCategory: {category_info}"

                image_url = row.get("large_image", "cover-not-found.jpg")
                
                results.append((image_url, caption))
                print(f"Added recommendation {idx + 1}: {title}")
                
            except Exception as e:
                print(f"Error processing recommendation {idx}: {str(e)}")
                continue
        
        print(f"Successfully formatted {len(results)} recommendations")
        return results
        
    except Exception as e:
        print(f"Error in recommend_books: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

try:
    if not books.empty and "categories" in books.columns:
        unique_categories = [str(x) for x in books["categories"].unique() if pd.notna(x)]
        categories = ["All"] + sorted(unique_categories)
        print(f"Available categories: {categories[:10]}...")  
    else:
        categories = ["All"]
        print("No categories available, using default")
except Exception as e:
    print(f"Error getting categories: {e}")
    categories = ["All"]

tones = ["All", "Surprising", "Happy", "Sad", "Angry", "Fearful", "Disgusted"]

with gr.Blocks(theme=gr.themes.Soft()) as grdash:
    gr.Markdown("# Book Recommendation Dashboard")
    
    with gr.Row():
        user_query = gr.Textbox(
            label="Please enter a description of a book:",
            placeholder="e.g., A story about forgiveness",
            value=""  
        )
        category_dropdown = gr.Dropdown(
            choices=categories, 
            label="Select a category:", 
            value="All"
        )
        tone_dropdown = gr.Dropdown(
            choices=tones, 
            label="Select an emotional tone:", 
            value="All"
        )
    
    with gr.Row():
        submit_button = gr.Button("Find recommendations", variant="primary")
    
    gr.Markdown("## Recommendations")
    output = gr.Gallery(
        label="Recommended books", 
        columns=4,  
        rows=4, 
        height="auto"
    )

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )

if __name__ == "__main__":
    grdash.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )