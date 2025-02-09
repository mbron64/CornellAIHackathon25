from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import AIMessage
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import os
import streamlit as st

class StyleAnalyzer:
    def __init__(self, llm):
        self.llm = llm
        self.embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            model="amazon.titan-text-embeddings.v2"
        )
        # Use semantic chunking for better style analysis
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ";"]  # Semantic boundaries
        )
        
        # Initialize prompts for detailed style analysis
        self.style_extraction_prompt = PromptTemplate(
            template="""Analyze the following text segments and extract their collective stylistic characteristics. Consider:

1. Voice and Perspective:
- Point of view (first-person, second-person, third-person)
- Level of formality and authority
- Author's attitude, tone, and emotional resonance
- Relationship with the reader

2. Language Patterns:
- Sentence structure complexity and variation
- Paragraph organization and flow
- Transitional phrases and their usage
- Rhythm and pacing
- Use of active vs passive voice

3. Rhetorical Devices:
- Use of metaphors, analogies, or other figurative language
- Persuasive techniques and argumentation style
- Emotional appeals and their execution
- Use of evidence and examples
- Logical flow and reasoning patterns

4. Vocabulary and Word Choice:
- Level of technical language and jargon
- Word sophistication and variety
- Recurring phrases or expressions
- Connotative vs denotative language use
- Industry-specific terminology

5. Structural Elements:
- Opening and closing patterns
- Paragraph length and structure
- Information density
- Use of lists, quotes, or other formatting
- Topic progression and development

6. Common Errors and Quirks:
- Recurring grammatical patterns (even if incorrect)
- Consistent spelling variations or mistakes
- Punctuation habits and errors
- Run-on sentences or fragments
- Word choice inconsistencies

7. Writing Improvement Areas:
- Grammar and syntax errors
- Sentence structure issues
- Clarity and conciseness problems
- Redundancy patterns
- Awkward phrasings
- Tense consistency issues
- Common spelling mistakes

Text Segments:
{text}

Provide a comprehensive analysis that captures:
1. Common patterns across all segments
2. Notable variations or style shifts
3. Distinctive stylistic markers
4. Context-dependent style adaptations
5. Overall writing personality
6. Recurring errors and improvement areas

Analysis:""",
            input_variables=["text"]
        )
        
        self.style_synthesis_prompt = PromptTemplate(
            template="""Based on the following detailed style analyses, synthesize a comprehensive style guide that captures both consistent patterns and contextual variations:

Style Analyses:
{style_analyses}

Create a detailed style guide that includes:

1. Core Writing Principles:
- Primary voice and tone guidelines
- Fundamental approach to communication
- Key stylistic values and priorities

2. Structural Patterns:
- Preferred sentence structures and variations
- Paragraph organization principles
- Document-level organization patterns

3. Language Usage:
- Vocabulary preferences and restrictions
- Technical language guidelines
- Phrase patterns and expressions
- Grammar and syntax preferences

4. Rhetorical Approach:
- Preferred persuasion techniques
- Evidence and example usage
- Argumentation patterns
- Emotional appeal guidelines

5. Contextual Adaptations:
- Style variations for different contexts
- Tone modulation guidelines
- Format-specific adjustments

6. Distinctive Markers:
- Unique stylistic features
- Signature phrases or patterns
- Special formatting preferences

7. Error Patterns and Improvements:
- Common grammatical errors to maintain authenticity
- Typical sentence structure issues
- Recurring clarity problems
- Characteristic spelling mistakes
- Punctuation patterns and errors
- Word choice inconsistencies
- Areas needing improvement while preserving voice

Style Guide:""",
            input_variables=["style_analyses"]
        )

    def learn_style_from_samples(self, samples: List[str]) -> Dict:
        """Learn writing style from sample documents using sophisticated analysis"""
        progress_text = st.empty()
        
        def update_progress(text):
            progress_text.text(f"⏳ Style Analysis: {text}")
            
        def get_content(response):
            """Extract content from LLM response"""
            if isinstance(response, AIMessage):
                return response.content
            elif isinstance(response, str):
                return response
            elif hasattr(response, 'content'):
                return response.content
            return str(response)
        
        try:
            update_progress("Performing semantic text splitting...")
            # Convert samples to strings if they're not already
            text_samples = [str(sample) for sample in samples]
            chunks = []
            for sample in text_samples:
                chunks.extend(self.text_splitter.split_text(sample))
            
            update_progress(f"Generating semantic embeddings for {len(chunks)} chunks...")
            chunk_embeddings = [self.embeddings.embed_query(chunk) for chunk in chunks]
            
            update_progress("Performing dimensional reduction and clustering...")
            # Use PCA for initial dimension reduction
            pca = PCA(n_components=min(10, len(chunks)-1))
            reduced_dims = pca.fit_transform(chunk_embeddings)
            
            # Use hierarchical clustering for better style grouping
            n_clusters = min(15, len(chunks))  # Allow more clusters for better granularity
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='euclidean',
                linkage='ward'
            )
            cluster_labels = clustering.fit_predict(reduced_dims)
            
            # Find representative chunks using similarity analysis
            update_progress("Identifying representative style patterns...")
            representative_chunks = []
            similarity_matrix = cosine_similarity(chunk_embeddings)
            
            for i in range(n_clusters):
                cluster_indices = np.where(cluster_labels == i)[0]
                if len(cluster_indices) == 0:
                    continue
                    
                # Find chunk with highest average similarity to others in cluster
                cluster_similarities = similarity_matrix[cluster_indices][:, cluster_indices]
                avg_similarities = np.mean(cluster_similarities, axis=1)
                central_idx = cluster_indices[np.argmax(avg_similarities)]
                
                # Also get some variation if cluster is large enough
                if len(cluster_indices) > 3:
                    # Get chunk that's different but still representative
                    variation_idx = cluster_indices[
                        np.argsort(avg_similarities)[-3]  # Third most representative
                    ]
                    representative_chunks.extend([chunks[central_idx], chunks[variation_idx]])
                else:
                    representative_chunks.append(chunks[central_idx])
            
            update_progress("Creating analysis chain...")
            analysis_chain = self.style_extraction_prompt | self.llm
            
            update_progress("Performing detailed style analysis...")
            # Analyze style patterns in small batches
            batch_size = 2  # Smaller batches for more detailed analysis
            style_analyses = []
            for i in range(0, len(representative_chunks), batch_size):
                batch = representative_chunks[i:i+batch_size]
                update_progress(f"Analyzing batch {i//batch_size + 1}/{(len(representative_chunks) + batch_size - 1)//batch_size}...")
                batch_text = "\n\n=====\n\n".join(batch)
                analysis = get_content(analysis_chain.invoke({"text": batch_text}))
                style_analyses.append(analysis)
            
            update_progress("Synthesizing comprehensive style guide...")
            synthesis_chain = self.style_synthesis_prompt | self.llm
            style_guide = get_content(synthesis_chain.invoke({"style_analyses": "\n\n---\n\n".join(style_analyses)}))
            
            # Calculate style variation metrics
            style_variations = {
                "cluster_sizes": np.bincount(cluster_labels).tolist(),
                "principal_components": pca.components_.tolist()[:3],  # Top 3 style dimensions
                "variance_explained": pca.explained_variance_ratio_.tolist(),
                "cluster_cohesion": [
                    float(np.mean(similarity_matrix[cluster_labels == i][:, cluster_labels == i]))
                    for i in range(n_clusters)
                ]
            }
            
            update_progress("✅ Complete!")
            return {
                "style_dimensions": reduced_dims.tolist(),
                "style_analyses": style_analyses,
                "style_guide": style_guide,
                "chunk_embeddings": chunk_embeddings,
                "style_variations": style_variations
            }
        except Exception as e:
            update_progress(f"❌ Error in style analysis: {str(e)}")
            raise e

    def get_style_prompt(self) -> PromptTemplate:
        """Generate a prompt template that guides the LLM to maintain the learned style"""
        return PromptTemplate(
            template="""You are an expert writer who has studied and internalized the following writing style guide:

{style_guide}

Using this style guide as your foundation, respond to the following query while maintaining the exact same writing style, voice, patterns, and characteristic errors described in the guide.

Context from relevant documents:
{context}

Query: {query}

Remember to:
1. Match the tone and formality level
2. Use similar sentence structures
3. Employ comparable rhetorical devices
4. Choose vocabulary that aligns with the style
5. Maintain consistent perspective and voice
6. Incorporate relevant information from the context
7. Replicate characteristic writing quirks:
   - Use similar grammatical patterns (including any common errors)
   - Maintain typical punctuation habits
   - Include characteristic spelling variations
   - Mirror sentence structure tendencies (including fragments or run-ons if present)
   - Keep consistent error patterns in word choice and usage
   - Preserve any tense consistency issues
   - Maintain the same level of clarity/redundancy

Important: The goal is to authentically match the original writing style, including both its strengths and imperfections. Do not try to improve or correct the characteristic errors - they are part of the authentic voice.

Response:""",
            input_variables=["style_guide", "query", "context"]
        )

    def calculate_style_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts' writing styles"""
        # Get embeddings
        emb1 = self.embeddings.embed_query(text1)
        emb2 = self.embeddings.embed_query(text2)
        
        # Calculate cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity) 