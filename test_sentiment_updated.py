#FINAL VERSION USED
from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import pandas as pd # for handling csv data
import os
import tiktoken # to count words /tokens
from dotenv import load_dotenv

load_dotenv()
# Ensure OpenAI API key is set
if 'OPENAI_API_KEY' not in os.environ:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# Load data from CSV file
csv_path = "packing_bag.csv"
df = pd.read_csv(csv_path)

# Combine title, body, and rating into one piece of text
df["processed_text"] = df[["Title", "Body", "Rating"]].astype(str).agg(". ".join, axis=1)

# --------------------------------------------------
# Token Management
# --------------------------------------------------
tokenizer = tiktoken.get_encoding("cl100k_base")

def chunk_reviews(df, text_col="processed_text", max_chunk_tokens=100000):
    """
    Split the DataFrame into multiple 'chunks' of reviews so each chunk
    stays below 'max_chunk_tokens' when combined into a single prompt.
    This REPLACES the old 'sample_reviews' approach.
    """
    chunks = []
    current_chunk = []
    current_tokens = 0

    # We can just iterate in the original row order, or sorted by rating if you like.
    for _, row in df.iterrows():
        review_text = row[text_col]
        review_tokens = len(tokenizer.encode(review_text))

        # If adding this review exceeds the limit, we start a new chunk
        if current_tokens + review_tokens > max_chunk_tokens:
            if current_chunk:
                chunks.append(pd.DataFrame(current_chunk))
            current_chunk = [row]
            current_tokens = review_tokens
        else:
            current_chunk.append(row)
            current_tokens += review_tokens

    # Add last chunk if non-empty
    if current_chunk:
        chunks.append(pd.DataFrame(current_chunk))

    return chunks

# A simple aggregator prompt that merges partial summaries into one final.
AGGREGATOR_PROMPT = """You are an AI assistant that merges multiple partial summaries 
of the same reviews into one cohesive summary. 
Here are the partial summaries:{input}

Please combine them into a single final output without repeating information. 
Maintain the same style and structure as the partial summaries, 
just unified into one coherent text.
"""

# ---------------------------------------------------------------
# Initialize LLM and embeddings
# ---------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
embeddings = OpenAIEmbeddings()

def create_chain(template):
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | llm | StrOutputParser()

# ---------------------------------------------------------------
# Prompts (unchanged)
# ---------------------------------------------------------------
sentiment_analysis_prompt = """Analyze the sentiment of these {total_reviews} customer reviews for the given product.

Reviews: ```{input}```

Provide a detailed analysis including:
1. Overall sentiment distribution
2. Key positive and negative themes
3. Emotional insights from the reviews

Format your response as a structured summary highlighting:
- Total number of reviews analyzed
- Percentage of positive, neutral, and negative reviews
- Top 3 positive aspects mentioned
- Top 3 negative aspects mentioned
- Most common emotions expressed
"""

purchase_motivation_prompt = """Analyze the customer reviews to understand the key motivations behind purchasing this product.

Reviews: ```{input}```

Investigate and explain:
1. Primary reasons customers bought this product
2. Key features that attracted buyers
3. Specific needs or problems the product addresses
4. Most compelling selling points from customer perspectives

Provide a comprehensive overview of customer purchase motivations.
"""

selling_points_prompt = """Extract and analyze the most compelling selling points from these customer reviews.

Reviews: ```{input}```

Identify:
1. Unique product features highlighted by customers
2. Standout qualities that differentiate this product
3. Specific use cases or scenarios where the product excels
4. Potential marketing angles based on customer feedback

Develop a concise list of key selling points with brief explanations.
"""

recommendations_prompt = """Generate actionable insights and improvement recommendations based on these customer reviews.

Reviews: ```{input}```

Analyze:
1. Recurring customer pain points
2. Suggested improvements from reviews
3. Potential product enhancements
4. Areas of concern or dissatisfaction

Provide specific, actionable recommendations for product development and customer experience improvement.
"""

# ---------------------------------------------------------------
# Updated process_reviews with chunking + aggregator
# ---------------------------------------------------------------
total_count = len(df)

# 2) In process_reviews, pass the total count as an additional key
def process_reviews(df, chain, context):
    try:
        df_chunks = chunk_reviews(df)  # your chunking function

        partial_summaries = []
        for chunk_df in df_chunks:
            chunk_text = "\n\n".join(chunk_df["processed_text"])
            # call the chain with "input" + "total_reviews" 
            partial_result = chain.invoke({
                "input": chunk_text,
                "total_reviews": total_count
            })
            partial_summaries.append(partial_result)
        
        # If we only have one partial summary, just return it
        if len(partial_summaries) == 1:
            return partial_summaries[0]
        elif len(partial_summaries) == 0:
            return "No reviews found in this chunk."

        # 2) If multiple partial summaries, we combine them with aggregator
        aggregator_chain = create_chain(AGGREGATOR_PROMPT)
        combined_text = "\n\n".join(partial_summaries)
        final_result = aggregator_chain.invoke({"input": combined_text})
        
        return final_result

    except Exception as e:
        print(f"Error processing reviews: {e}")
        return None

# ---------------------------------------------------------------
# Analyze reviews
# ---------------------------------------------------------------
def analyze_reviews(df):
    results = {
        "sentiment": process_reviews(
            df,
            create_chain(sentiment_analysis_prompt),
            "Analyzing sentiment of customer reviews"
        ),
        "purchase_motivation": process_reviews(
            df,
            create_chain(purchase_motivation_prompt),
            "Understanding purchase motivations"
        ),
        "selling_points": process_reviews(
            df,
            create_chain(selling_points_prompt),
            "Identifying selling points"
        ),
        "recommendations": process_reviews(
            df,
            create_chain(recommendations_prompt),
            "Generating product recommendations"
        )
    }
    return results

# Run the analysis
if __name__ == "__main__":
    analysis_results = analyze_reviews(df)
    
    # Print the responses for each task
    for key, value in analysis_results.items():
        print(f"{key.replace('_', ' ').title()} Response:")
        print(value)
        print("\n" + "=" * 50 + "\n")


