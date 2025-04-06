import streamlit as st
import numpy as np
import pandas as pd
import time
from transformers import pipeline, set_seed

def show():
    st.title("🧠 Introduction to Generative AI & LLMs")
    
    st.markdown("""
    This module introduces you to Generative AI and Large Language Models (LLMs). 
    You'll learn about different types of models and try out various text generation and analysis tasks.
    """)
    
    # Create tabs for different topics
    tabs = st.tabs([
        "What is GenAI?", 
        "Hugging Face Models", 
        "Text Generation", 
        "Summarization", 
        "Sentiment Analysis"
    ])
    
    # What is GenAI tab
    with tabs[0]:
        st.header("What is Generative AI?")
        
        st.markdown("""
        **Generative AI** refers to artificial intelligence systems that can generate new content, 
        rather than just analyzing or categorizing existing data.
        
        ### Types of Generative AI
        
        1. **Text Generation** (GPT, BART, T5)
           - Creates human-like text based on prompts
           - Applications: chatbots, content creation, code generation
        
        2. **Image Generation** (DALL-E, Stable Diffusion, Midjourney)
           - Creates images from text descriptions
           - Applications: art, design, visualization
        
        3. **Audio Generation** (WaveNet, Jukebox, AudioLM)
           - Creates realistic speech or music
           - Applications: text-to-speech, music composition
        
        4. **Video Generation** (Sora, Gen-2, Phenaki)
           - Creates videos from text or images
           - Applications: animation, simulation
        
        ### How LLMs Work
        
        LLMs (Large Language Models) are a type of generative AI focused on text. Here's a simplified explanation of how they work:
        """)
        
        # Create columns for the explanation
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.markdown("""
            **Key Concepts:**
            
            1. **Tokens**
               - Words/subwords the model processes
               - Example: "I love AI" → ["I", "love", "AI"]
            
            2. **Embeddings**
               - Numerical representations of tokens
               - Capture meaning and relationships
            
            3. **Attention**
               - Mechanism to focus on relevant context
               - Allows understanding relationships between words
            
            4. **Parameters**
               - The weights learned during training
               - More parameters = more capacity
            
            5. **Transformer Architecture**
               - Current dominant approach
               - Uses self-attention mechanisms
            """)
        
        with col2:
            st.markdown("""
            **The Basic Process:**
            
            1. **Pre-training**
               - Model learns from vast amounts of text
               - Predicts next word given previous words
               - Develops general language understanding
            
            2. **Fine-tuning (optional)**
               - Additional training on specific tasks
               - Adapts general knowledge to specific domains
            
            3. **Inference (generation)**
               - Given a prompt, predicts next tokens
               - Generation strategies control output quality
               - Uses probability distributions to choose words
            
            4. **Evaluation**
               - Assessing quality, factuality, bias
               - Comparing to human preferences
            """)
        
        st.markdown("---")
        
        st.subheader("LLM Evolution Timeline")
        
        # Timeline data
        timeline_data = {
            "Model": ["BERT", "GPT-2", "T5", "GPT-3", "BLOOM", "LLaMA", "GPT-4", "Claude", "Gemini"],
            "Year": [2018, 2019, 2020, 2020, 2022, 2023, 2023, 2023, 2023],
            "Parameters": ["340M", "1.5B", "11B", "175B", "176B", "65B", "~1.8T", "~150B", "~1.5T"],
            "Key Innovation": [
                "Bidirectional training", 
                "Zero-shot capabilities", 
                "Text-to-Text approach", 
                "Few-shot learning", 
                "Multilingual focus", 
                "Open weights for research", 
                "Multimodal capabilities", 
                "Constitutional AI", 
                "Improved reasoning"
            ]
        }
        
        timeline_df = pd.DataFrame(timeline_data)
        st.dataframe(timeline_df, hide_index=True, use_container_width=True)
    
    # Hugging Face Models tab
    with tabs[1]:
        st.header("Hugging Face Models")
        
        st.markdown("""
        [Hugging Face](https://huggingface.co/) is a platform that hosts thousands of pre-trained models, 
        datasets, and a library called `transformers` that makes it easy to work with state-of-the-art NLP models.
        
        ### Key Advantages
        
        - **Accessibility**: Easy-to-use APIs for complex models
        - **Variety**: Thousands of pre-trained models for different tasks
        - **Community**: Open-source contributions and sharing
        - **Documentation**: Extensive guides and examples
        
        ### Popular Model Families
        """)
        
        # Create two columns for the model families
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **BERT & RoBERTa**
            - Bidirectional context understanding
            - Great for classification, NER, Q&A
            - Not designed for text generation
            
            **GPT Family**
            - Unidirectional (left-to-right) generation
            - Excellent for creative text generation
            - Models like GPT-2, DialoGPT
            
            **T5 & BART**
            - Sequence-to-sequence models
            - Good for summarization, translation
            - Flexible for various text tasks
            """)
        
        with col2:
            st.markdown("""
            **Multilingual Models**
            - XLM-RoBERTa, mBART, mT5
            - Support dozens of languages
            - Transfer learning across languages
            
            **Specialized Models**
            - CodeBERT/CodeGen for code
            - BioMed-RoBERTa for medical text
            - DistilBERT for efficiency
            
            **Open LLMs**
            - BLOOM, LLaMA, Mistral
            - Open alternatives to proprietary models
            - Various parameter sizes (7B to 176B)
            """)
        
        st.markdown("---")
        
        st.subheader("The Transformers Library")
        
        st.markdown("""
        The `transformers` library provides a simple way to use these models with just a few lines of code:
        """)
        
        st.code("""
from transformers import pipeline

# Create a text generation pipeline
generator = pipeline('text-generation', model='gpt2')

# Generate text
result = generator("Artificial intelligence is", max_length=50, num_return_sequences=1)

print(result[0]['generated_text'])
        """)
        
        st.markdown("""
        ### Main Components
        
        1. **Tokenizers**: Convert text to token IDs and back
        2. **Models**: The neural networks that process tokens
        3. **Pipelines**: High-level abstractions for common tasks
        4. **Configuration**: Settings for model behavior
        
        In the next tabs, we'll try out different pipelines for various tasks.
        """)
    
    # Text Generation tab
    with tabs[2]:
        st.header("Text Generation")
        
        st.markdown("""
        Text generation is one of the most common applications of LLMs. In this section, 
        we'll experiment with generating text using models like GPT-2.
        """)
        
        st.warning("Note: The examples below will use smaller models that can run in this environment. Production systems typically use larger, more capable models.")
        
        # Model selection
        model_options = ["gpt2", "distilgpt2", "EleutherAI/gpt-neo-125M"]
        selected_model = st.selectbox("Select a model", model_options)
        
        # Text prompt
        default_prompts = [
            "Artificial intelligence is",
            "The future of technology will be shaped by",
            "Once upon a time in a digital world",
            "The most interesting application of LLMs is"
        ]
        
        default_prompt = st.selectbox("Choose a starter prompt (or write your own below)", 
                                     ["Custom"] + default_prompts)
        
        if default_prompt == "Custom":
            user_prompt = st.text_area("Enter your prompt", "")
        else:
            user_prompt = st.text_area("Enter your prompt", default_prompt)
        
        # Generation parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_length = st.slider("Max Length", min_value=10, max_value=200, value=50, step=10, 
                                 help="Maximum number of tokens to generate")
        
        with col2:
            temperature = st.slider("Temperature", min_value=0.1, max_value=1.5, value=0.7, step=0.1,
                                  help="Higher values make output more random, lower values more deterministic")
        
        with col3:
            num_sequences = st.slider("Number of Sequences", min_value=1, max_value=5, value=1, step=1,
                                    help="Number of different completions to generate")
        
        # Generate button
        if st.button("Generate Text") and user_prompt:
            try:
                with st.spinner(f"Generating text with {selected_model}..."):
                    # Set up the pipeline
                    generator = pipeline('text-generation', model=selected_model)
                    set_seed(42)  # For reproducibility
                    
                    # Generate text
                    results = generator(
                        user_prompt, 
                        max_length=max_length, 
                        num_return_sequences=num_sequences,
                        temperature=temperature,
                        do_sample=True
                    )
                    
                    # Display results
                    for i, result in enumerate(results):
                        st.subheader(f"Generation {i+1}")
                        st.write(result['generated_text'])
                        st.markdown("---")
            except Exception as e:
                st.error(f"Error generating text: {e}")
        
        st.markdown("""
        ### How It Works
        
        1. The model receives your prompt as input
        2. It tokenizes the text into smaller units
        3. For each position, it predicts probabilities for the next token
        4. It selects tokens based on these probabilities, influenced by parameters like temperature
        5. The process repeats until reaching max length or a stop token
        
        ### Key Parameters
        
        - **Temperature**: Controls randomness (higher = more creative, lower = more focused)
        - **Top-k/Top-p**: Filtering strategies for next-token selection
        - **Max Length**: Maximum number of tokens to generate
        - **Repetition Penalty**: Reduces likelihood of repeating the same phrases
        """)
    
    # Summarization tab
    with tabs[3]:
        st.header("Text Summarization")
        
        st.markdown("""
        Summarization is the task of condensing a longer text into a shorter version while preserving the key information.
        
        There are two main approaches:
        1. **Extractive Summarization**: Selects and combines existing sentences from the text
        2. **Abstractive Summarization**: Generates new sentences that capture the essence of the text
        
        LLMs typically perform abstractive summarization, creating new text rather than just extracting sentences.
        """)
        
        # Sample texts
        sample_texts = {
            "AI News": """
Artificial intelligence continues to transform industries worldwide. Recent breakthroughs in large language models have enabled more natural human-computer interaction, while advances in computer vision are revolutionizing healthcare diagnostics. Researchers are now focusing on reducing the computational requirements of these models to make AI more accessible and environmentally friendly. Meanwhile, concerns about AI ethics and governance remain at the forefront of policy discussions, with several countries drafting new regulatory frameworks to address potential risks while fostering innovation. Industry leaders argue that responsible AI development requires collaboration between technologists, ethicists, and policymakers to ensure these powerful tools benefit humanity.
            """,
            
            "Climate Science": """
Climate scientists have documented a concerning acceleration in global warming trends over the past decade. Recent studies indicate that Arctic sea ice is melting faster than predicted by previous models, potentially leading to more extreme weather events worldwide. The relationship between ocean temperatures and atmospheric patterns has been clarified through new data collection methods, revealing complex feedback mechanisms that may amplify warming effects. Despite these challenges, technological advances in renewable energy production have shown promise, with solar and wind power becoming increasingly cost-competitive with fossil fuels. International cooperation remains essential, as climate change impacts transcend national boundaries and require coordinated responses. Adaptation strategies are being developed alongside mitigation efforts to help vulnerable communities cope with unavoidable changes already set in motion.
            """,
            
            "Medical Research": """
A groundbreaking study published this week has identified a potential new treatment approach for Alzheimer's disease. Researchers discovered that targeting specific protein aggregates in brain cells could significantly slow cognitive decline in early-stage patients. The clinical trial, which followed 500 participants over three years, showed a 27% reduction in symptom progression compared to standard care. This approach differs from previous treatments by focusing on cellular repair mechanisms rather than just removing amyloid plaques. The research team emphasized that while these results are promising, larger studies are needed to confirm efficacy and safety profiles across diverse populations. Funding agencies have already committed additional resources to accelerate this research, given the growing public health challenge posed by neurodegenerative disorders in aging societies. The potential treatment could enter advanced clinical trials within two years if current results are validated.
            """
        }
        
        selected_text = st.selectbox("Choose a sample text", list(sample_texts.keys()))
        
        if selected_text == "Custom":
            text_to_summarize = st.text_area("Enter text to summarize", "", height=200)
        else:
            text_to_summarize = st.text_area("Text to summarize", sample_texts[selected_text], height=200)
        
        # Model selection for summarization
        summarization_models = ["facebook/bart-large-cnn", "t5-small", "google/pegasus-xsum"]
        selected_sum_model = st.selectbox("Select a summarization model", summarization_models)
        
        # Parameters
        col1, col2 = st.columns(2)
        
        with col1:
            min_length = st.slider("Minimum Length", min_value=10, max_value=100, value=30, step=5,
                                 help="Minimum length of the summary in tokens")
        
        with col2:
            max_length = st.slider("Maximum Length", min_value=30, max_value=200, value=100, step=5,
                                 help="Maximum length of the summary in tokens")
        
        # Generate summary
        if st.button("Generate Summary") and text_to_summarize:
            try:
                with st.spinner(f"Generating summary with {selected_sum_model}..."):
                    # Set up the pipeline
                    summarizer = pipeline('summarization', model=selected_sum_model)
                    
                    # Generate summary
                    summary = summarizer(
                        text_to_summarize, 
                        min_length=min_length, 
                        max_length=max_length
                    )
                    
                    # Display result
                    st.subheader("Summary")
                    st.write(summary[0]['summary_text'])
                    
                    # Show word count comparison
                    original_words = len(text_to_summarize.split())
                    summary_words = len(summary[0]['summary_text'].split())
                    reduction = ((original_words - summary_words) / original_words) * 100
                    
                    st.info(f"Reduced from {original_words} words to {summary_words} words (reduced by {reduction:.1f}%)")
            except Exception as e:
                st.error(f"Error generating summary: {e}")
        
        st.markdown("""
        ### Applications of Summarization
        
        - **News Digests**: Condensing multiple news articles
        - **Research Papers**: Creating abstracts or executive summaries
        - **Meeting Notes**: Summarizing conversations or recordings
        - **Content Curation**: Processing large volumes of text for relevant information
        - **Legal Document Analysis**: Extracting key points from lengthy legal texts
        """)
    
    # Sentiment Analysis tab
    with tabs[4]:
        st.header("Sentiment Analysis")
        
        st.markdown("""
        Sentiment analysis is the process of determining the emotional tone behind text - 
        whether it's positive, negative, or neutral. LLMs excel at this task by understanding 
        context and nuance in language.
        """)
        
        # Sample texts
        sentiment_samples = [
            "I absolutely love this product! It has exceeded all my expectations and makes my life easier.",
            "This movie was disappointing. The plot was predictable and the acting was mediocre at best.",
            "The service was okay. Nothing special, but it got the job done without any major issues.",
            "While there were some positive aspects, overall I wouldn't recommend this experience.",
            "The new update completely ruined the application. It's slower and harder to use now."
        ]
        
        selected_sentiment = st.selectbox("Choose a sample text or enter your own below", 
                                        ["Custom"] + sentiment_samples)
        
        if selected_sentiment == "Custom":
            text_for_sentiment = st.text_area("Enter text for sentiment analysis", "")
        else:
            text_for_sentiment = st.text_area("Text for sentiment analysis", selected_sentiment)
        
        # Model selection
        sentiment_models = ["distilbert-base-uncased-finetuned-sst-2-english", "cardiffnlp/twitter-roberta-base-sentiment"]
        selected_sent_model = st.selectbox("Select a sentiment analysis model", sentiment_models)
        
        # Analyze sentiment
        if st.button("Analyze Sentiment") and text_for_sentiment:
            try:
                with st.spinner(f"Analyzing sentiment with {selected_sent_model}..."):
                    # Set up the pipeline
                    sentiment_analyzer = pipeline('sentiment-analysis', model=selected_sent_model)
                    
                    # Analyze sentiment
                    result = sentiment_analyzer(text_for_sentiment)
                    
                    # Display result
                    sentiment = result[0]['label']
                    score = result[0]['score']
                    
                    # Create a visual indicator
                    if 'POSITIVE' in sentiment or 'positive' in sentiment:
                        color = "green"
                        emoji = "😃"
                    elif 'NEGATIVE' in sentiment or 'negative' in sentiment:
                        color = "red"
                        emoji = "😞"
                    else:
                        color = "gray"
                        emoji = "😐"
                    
                    # Display with styling
                    st.markdown(f"<h3 style='color: {color};'>{sentiment} {emoji}</h3>", unsafe_allow_html=True)
                    st.progress(float(score))
                    st.write(f"Confidence: {score:.2f}")
            except Exception as e:
                st.error(f"Error analyzing sentiment: {e}")
        
        st.markdown("""
        ### Beyond Basic Sentiment
        
        Modern LLMs can detect more nuanced emotions:
        
        - Joy/Happiness
        - Sadness/Grief
        - Anger/Frustration
        - Fear/Anxiety
        - Surprise
        - Disgust
        
        They can also identify:
        
        - **Sarcasm and Irony**: "Oh great, another meeting. Just what I needed."
        - **Mixed Sentiments**: "The phone has an amazing camera but terrible battery life."
        - **Implicit Sentiments**: "This definitely matches the quality of their previous products..." (implicit negative)
        
        ### Applications
        
        - **Brand Monitoring**: Tracking customer sentiment on social media
        - **Product Feedback**: Analyzing reviews at scale
        - **Customer Service**: Prioritizing urgent negative feedback
        - **Market Research**: Understanding emotional responses to products/services
        - **Content Recommendation**: Matching content to desired emotional states
        """)
    
    st.markdown("---")
    st.info("This module introduced you to the basics of Generative AI and LLMs. In the next module, we'll explore more advanced LLM concepts and techniques.")
