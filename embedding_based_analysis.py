# working code using embedding-based solution for summarizing product reviews.
#Install required libraries
#pip install openai scikit-learn pandas numpy

import os
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import openai
#from google.colab import files

# Upload your CSV files if in colab
# print("Please upload your CSV files containing product reviews...")
# uploaded = files.upload()  # This will prompt you to upload files

# Set your OpenAI API key
OPENAI_API_KEY = "your openapi key" 
openai.api_key = OPENAI_API_KEY

def get_embeddings(texts):
    """Get embeddings for a list of texts using OpenAI's embedding model."""
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=texts
    )
    return [item.embedding for item in response.data]

def summarize_with_openai(prompt, model="gpt-4o", max_tokens=1000):
    """Generate a summary using OpenAI models."""
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert product analyst who specializes in extracting insights from customer reviews."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.2
    )
    return response.choices[0].message.content

def process_product_reviews(csv_path, df):
    """Process reviews for a single product from CSV."""
    # Extract product name from filename
    product_name = os.path.basename(csv_path).replace('.csv', '')
    print(f"Processing {product_name}...")
    
    # Handle missing values
    for col in ['Title', 'Body', 'Rating']:
        if col in df.columns:
            df[col] = df[col].fillna('')
        else:
            df[col] = ''
    
    # Combine columns for embedding
    df['full_text'] = df.apply(
        lambda row: f"Title: {row['Title']}. Body: {row['Body']}. Rating: {row['Rating']}", 
        axis=1
    )
    
    # Get review texts and create a clean dataframe with all info
    review_texts = df['full_text'].tolist()
    review_df = df[['Title', 'Body', 'Rating', 'full_text']].copy()
    
    # Get embeddings
    print(f"Generating embeddings for {len(review_texts)} reviews...")
    embeddings = get_embeddings(review_texts)
    
    # Determine number of clusters based on review count
    n_clusters = min(max(3, len(review_texts) // 10), 8)  # Between 3-8 clusters
    
    # Cluster the reviews
    print(f"Clustering reviews into {n_clusters} groups...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    review_df['cluster'] = clusters
    
    # Process each cluster
    cluster_summaries = []
    for cluster_id in range(n_clusters):
        # Get reviews in this cluster
        cluster_reviews = review_df[review_df['cluster'] == cluster_id]
        if len(cluster_reviews) == 0:
            continue
        
        # Get embeddings for this cluster
        cluster_indices = cluster_reviews.index.tolist()
        cluster_embeddings = [embeddings[i] for i in cluster_indices]
        
        # Calculate centroid
        centroid = np.mean(cluster_embeddings, axis=0)
        
        # Calculate similarity to centroid
        similarities = [cosine_similarity([centroid], [emb])[0][0] for emb in cluster_embeddings]
        
        # Add similarities to the dataframe
        similarity_series = pd.Series(similarities, index=cluster_indices)
        cluster_reviews = cluster_reviews.assign(similarity=similarity_series)
        
        # Select most representative reviews (closest to centroid)
        n_representatives = min(3, len(cluster_reviews))
        representative_reviews = cluster_reviews.sort_values('similarity', ascending=False).head(n_representatives)
        
        # Format review texts for summarization
        review_texts_for_summary = []
        for i, (_, review) in enumerate(representative_reviews.iterrows()):
            formatted_review = f"Review {i+1}:\n"
            formatted_review += f"Title: {review['Title']}\n"
            formatted_review += f"Rating: {review['Rating']}\n"
            formatted_review += f"Content: {review['Body']}\n"
            review_texts_for_summary.append(formatted_review)
        
        all_review_text = "\n\n".join(review_texts_for_summary)
        
        # Summarize cluster
        print(f"Summarizing cluster {cluster_id} ({len(cluster_reviews)} reviews)...")
        prompt = f"""I have a cluster of reviews for the product "{product_name}". 
        These reviews are grouped together because they share similar themes or sentiments.
        
        Please analyze these reviews and identify:
        1. The key theme(s) of this cluster
        2. Sentiment (positive, negative, mixed)
        3. Specific product aspects mentioned
        4. Common praises or complaints
        5. Customer needs being addressed or unmet
        
        Be concise and focus on what makes this group of reviews distinct.
        
        Reviews:
        {all_review_text}"""
        
        summary = summarize_with_openai(prompt)
        
        cluster_summaries.append({
            'cluster_id': cluster_id,
            'review_count': len(cluster_reviews),
            'summary': summary
        })
    
    # Create final product summary
    all_summaries = "\n\n".join([
        f"Cluster {summary['cluster_id']} Summary ({summary['review_count']} reviews):\n{summary['summary']}" 
        for summary in cluster_summaries
    ])
    
    print(f"Creating final summary for {product_name}...")
    prompt = f"""I have summaries of different clusters of reviews for the product "{product_name}".
    Each cluster represents reviews with similar themes or sentiments.
    
    Please synthesize these cluster summaries into a comprehensive product review analysis with:
    
    1. Overall sentiment analysis (show positive, negative, mixed)
    2. Key strengths and benefits highlighted by customers
    3. Major pain points or areas for improvement
    4. What customer needs this product addresses well or poorly
    5. Distinctive features compared to alternatives (if mentioned)
    6. Recommendations based on customer feedback
    
    Here are the cluster summaries:
    
    {all_summaries}"""
    
    final_summary = summarize_with_openai(prompt, model="gpt-4-turbo")
    
    return {
        'product': product_name,
        'summary': final_summary,
        'cluster_data': {
            'review_count': len(review_df),
            'cluster_count': n_clusters,
            'cluster_summaries': cluster_summaries
        }
    }

def main():
    start_time = time.time()
    
    # Process each uploaded file
    all_product_results = []
    
    for filename, content in uploaded.items():
        # Read the CSV content
        df = pd.read_csv(filename)
        product_result = process_product_reviews(filename, df)
        all_product_results.append(product_result)
        
        # Print individual product summary
        print(f"\n=== {product_result['product']} Summary ===")
        print(product_result['summary'])
    
    # If multiple products, create a comparative analysis
    if len(all_product_results) > 1:
        print("\nCreating cross-product comparison...")
        all_summaries = "\n\n".join([
            f"PRODUCT: {result['product']} ({result['cluster_data']['review_count']} reviews)\n\n{result['summary']}" 
            for result in all_product_results
        ])
        
        prompt = f"""I have review summaries for multiple products. Please create a comparative analysis that:
        
        1. Identifies strengths and weaknesses of each product
        2. Compares customer satisfaction across products
        3. Highlights unique features or benefits of each product
        4. Provides recommendations for different customer needs
        
        Here are the product summaries:
        
        {all_summaries}"""
        
        comparison = summarize_with_openai(prompt, model="gpt-4-turbo", max_tokens=2000)
        
        print("\n=== Cross-Product Comparative Analysis ===")
        print(comparison)
    
    end_time = time.time()
    print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")

# Run the main function
if _name_ == "__main__":
    main()