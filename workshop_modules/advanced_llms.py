import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline, set_seed
import json
import time

def show():
    st.title("🔍 Advanced LLM Concepts")
    
    st.markdown("""
    This module dives deeper into advanced concepts and techniques for working with Large Language Models. 
    You'll learn about prompt engineering, tokenization, parameter tuning, and more.
    """)
    
    # Create tabs for different topics
    tabs = st.tabs([
        "Prompt Engineering", 
        "Tokenization", 
        "Fine-tuning & PEFT", 
        "Decoding Strategies", 
        "Evaluation Metrics"
    ])
    
    # Prompt Engineering tab
    with tabs[0]:
        st.header("Prompt Engineering")
        
        st.markdown("""
        Prompt engineering is the practice of designing and optimizing inputs to language models 
        to get desired outputs. It's become an essential skill for working with LLMs.
        
        ### Why Prompt Engineering Matters
        
        - LLMs are incredibly sensitive to how prompts are phrased
        - The right prompting technique can dramatically improve results
        - It allows better control over model outputs without changing the model itself
        - Enables adaptation of general models to specific tasks
        """)
        
        st.subheader("Key Prompting Techniques")
        
        techniques = {
            "Zero-shot Prompting": {
                "description": "Asking the model to perform a task without examples",
                "example": "Translate the following English text to French: 'Hello, how are you?'"
            },
            "Few-shot Prompting": {
                "description": "Providing a few examples of the task before asking the model to complete it",
                "example": """English: Hello
French: Bonjour
English: How are you?
French: Comment allez-vous?
English: I love AI
French:"""
            },
            "Chain-of-Thought": {
                "description": "Encouraging the model to break down complex reasoning steps",
                "example": "To solve 15 × 7 + 22 × 3, I'll first multiply 15 × 7 = 105, then multiply 22 × 3 = 66, and finally add 105 + 66 = 171."
            },
            "Role Prompting": {
                "description": "Assigning a specific role to the AI to frame its responses",
                "example": "You are an expert physicist explaining complex concepts to a high school student. Explain quantum entanglement."
            },
            "Instruction Prompting": {
                "description": "Giving the model clear, structured instructions",
                "example": "Write a concise summary of the following text. Focus on the main arguments and key points. Text: [your text here]"
            }
        }
        
        technique = st.selectbox("Select a prompting technique to explore", list(techniques.keys()))
        
        st.markdown(f"**{technique}**")
        st.markdown(techniques[technique]["description"])
        st.markdown("**Example:**")
        st.code(techniques[technique]["example"])
        
        st.markdown("---")
        
        st.subheader("Interactive Prompt Workshop")
        
        st.markdown("""
        Let's experiment with different prompting techniques using a smaller model (GPT-2). 
        In production, these techniques work even better with more powerful models like GPT-4 or Claude.
        """)
        
        # Example task selection
        tasks = {
            "Summarization": {
                "description": "Create a summary of a given text",
                "base_prompt": "Summarize the following text:",
                "sample_content": "Climate change is a global challenge requiring immediate action. Rising temperatures are causing sea level rise, extreme weather events, and disruptions to ecosystems worldwide. Many countries have committed to reducing carbon emissions through international agreements, while scientists work on technological solutions ranging from renewable energy to carbon capture. Public awareness continues to grow, but political and economic obstacles remain significant barriers to comprehensive action."
            },
            "Classification": {
                "description": "Classify text into categories",
                "base_prompt": "Classify the following text as either Business, Technology, Sports, or Entertainment:",
                "sample_content": "Apple unveiled its latest iPhone model yesterday, featuring an improved camera system and faster processor. The CEO presented the device during their annual September event, which was livestreamed to millions of viewers worldwide."
            },
            "Question Answering": {
                "description": "Answer questions based on provided context",
                "base_prompt": "Answer the following question based on the given context.\nQuestion: When was the company founded?\nContext:",
                "sample_content": "TechCorp was established in 2005 by Jane Smith and Mark Johnson. The company initially focused on mobile app development before expanding into cloud services in 2012. Their headquarters is in Boston, with additional offices in San Francisco and London. Last year, they reached 500 employees and reported annual revenue of $75 million."
            }
        }
        
        selected_task = st.selectbox("Select a task", list(tasks.keys()))
        task_info = tasks[selected_task]
        
        # Prompt components
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Prompt Structure")
            prompt_instruction = st.text_area("Instructions", task_info["base_prompt"], height=100)
            
            # Advanced options
            advanced_options = st.expander("Advanced Options")
            with advanced_options:
                add_examples = st.checkbox("Include few-shot examples")
                
                if add_examples:
                    if selected_task == "Summarization":
                        examples = st.text_area("Examples (before your main content)", 
                                               """Text: The meeting lasted for two hours and covered budget planning, marketing strategy, and upcoming product launches. Team leads presented their quarterly goals.
Summary: Two-hour meeting covering budget, marketing, and product plans with quarterly goals from team leads.

Text: """, height=150)
                    elif selected_task == "Classification":
                        examples = st.text_area("Examples (before your main content)", 
                                               """Text: The stock market fell by 2% yesterday following the Federal Reserve's announcement.
Category: Business

Text: """, height=150)
                    else:
                        examples = st.text_area("Examples (before your main content)", 
                                               """Question: What is the company's main product?
Context: Acme Corp manufactures industrial equipment, specializing in conveyor belt systems for factories.
Answer: Conveyor belt systems.

Question: When was the company founded?
Context: """, height=150)
                
                add_cot = st.checkbox("Add chain-of-thought instruction")
                if add_cot:
                    cot_instruction = st.text_area("Chain-of-thought instruction", 
                                                 "Let's think through this step by step.", height=80)
                
                add_role = st.checkbox("Add role prompting")
                if add_role:
                    role_instruction = st.text_area("Role instruction", 
                                                  "You are an expert assistant specializing in clear and concise analysis.", height=80)
        
        with col2:
            st.markdown("#### Content")
            content = st.text_area("Content to process", task_info["sample_content"], height=200)
            
            # Build the complete prompt
            complete_prompt = ""
            
            if add_role and 'role_instruction' in locals():
                complete_prompt += f"{role_instruction}\n\n"
            
            if add_examples and 'examples' in locals():
                complete_prompt += f"{examples}"
            
            complete_prompt += f"{prompt_instruction}\n\n{content}"
            
            if add_cot and 'cot_instruction' in locals():
                complete_prompt += f"\n\n{cot_instruction}"
            
            st.markdown("#### Complete Prompt")
            st.text_area("Final prompt", complete_prompt, height=200)
        
        # Process the prompt
        if st.button("Run Prompt"):
            with st.spinner("Processing with model..."):
                try:
                    # Use text generation pipeline
                    generator = pipeline('text-generation', model='gpt2')
                    set_seed(42)  # For reproducibility
                    
                    response = generator(
                        complete_prompt,
                        max_length=len(complete_prompt.split()) + 100,  # Add tokens for response
                        num_return_sequences=1,
                        temperature=0.7
                    )
                    
                    # Extract the generated text (excluding the prompt)
                    generated_text = response[0]['generated_text'][len(complete_prompt):]
                    
                    st.subheader("Model Response")
                    st.write(generated_text)
                    
                    # Add explanation
                    st.info("Note: This example uses GPT-2, which is a smaller model with limited capabilities. In production, you would use more advanced models like GPT-4, Claude, or Llama 2 for better results.")
                except Exception as e:
                    st.error(f"Error generating response: {e}")
        
        st.markdown("---")
        
        st.markdown("""
        ### Prompt Engineering Best Practices
        
        1. **Be specific and clear** in your instructions
        2. **Break complex tasks into steps** for better reasoning
        3. **Provide context** that helps the model understand the task
        4. **Use consistent formatting** for better pattern recognition
        5. **Experiment and iterate** - prompt engineering is largely empirical
        6. **Consider the model's limitations** - no model is perfect
        7. **Use system prompts** (in models that support them) to set overall behavior
        """)
    
    # Tokenization tab
    with tabs[1]:
        st.header("Tokenization")
        
        st.markdown("""
        Tokenization is the process of converting text into tokens that language models can process. 
        It's a fundamental step in how LLMs understand and generate text.
        
        ### What are Tokens?
        
        Tokens are the basic units that LLMs process. They can be:
        - Words
        - Parts of words (subwords)
        - Characters
        - Punctuation
        
        Different models use different tokenization methods, which affects how they process text.
        """)
        
        st.subheader("Tokenization Visualization")
        
        # Text input for tokenization
        sample_texts = [
            "Hello world! How are you doing today?",
            "Artificial intelligence and machine learning are transforming industries worldwide.",
            "GPT-4 can understand and generate human-like text based on the input it receives.",
            "The transformer architecture revolutionized NLP in 2017 with the paper 'Attention is All You Need'.",
            "Tokenization splits text into smaller units that can be processed by language models."
        ]
        
        selected_sample = st.selectbox("Choose a sample text or write your own", ["Custom"] + sample_texts)
        
        if selected_sample == "Custom":
            text_to_tokenize = st.text_area("Enter text to tokenize", "")
        else:
            text_to_tokenize = st.text_area("Text to tokenize", selected_sample)
        
        if text_to_tokenize:
            # Simulate tokenization since we don't want to load actual tokenizers here
            # In a real app, you would use actual tokenizers from transformers library
            
            # GPT-2 tokenizer simulation (simplified)
            gpt2_tokens = []
            current_pos = 0
            words = text_to_tokenize.split()
            
            for word in words:
                if len(word) <= 4:
                    gpt2_tokens.append(word)
                else:
                    # Simulate subword tokenization
                    prefix = word[:3]
                    suffix = word[3:]
                    if len(suffix) > 4:
                        gpt2_tokens.extend([prefix, suffix[:3], suffix[3:]])
                    else:
                        gpt2_tokens.extend([prefix, suffix])
            
            # BERT tokenizer simulation (simplified)
            bert_tokens = []
            for word in words:
                if word.lower() in ["the", "and", "is", "of", "to", "in", "a"]:
                    bert_tokens.append(word)
                elif len(word) <= 3:
                    bert_tokens.append(word)
                else:
                    # Simulate wordpiece tokenization
                    if word[0].isupper():
                        prefix = "##" + word[:2].lower()
                    else:
                        prefix = word[:2]
                    suffix = "##" + word[2:]
                    bert_tokens.extend([prefix, suffix])
            
            # Character tokenization (extreme case)
            char_tokens = list(text_to_tokenize)
            
            # Display the tokens
            st.markdown("### Tokenization Comparison")
            
            comparison_data = {
                "Tokenizer": ["Word-level", "GPT-2 Style (subword)", "BERT Style (wordpiece)", "Character-level"],
                "Tokens": [
                    " ".join(words),
                    " ".join(gpt2_tokens),
                    " ".join(bert_tokens),
                    " ".join(char_tokens)
                ],
                "Token Count": [
                    len(words),
                    len(gpt2_tokens),
                    len(bert_tokens),
                    len(char_tokens)
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.table(comparison_df)
            
            st.info("Note: This is a simplified simulation of tokenization for educational purposes. Real tokenizers use more sophisticated algorithms and vocabulary files.")
        
        st.markdown("---")
        
        st.markdown("""
        ### Why Tokenization Matters
        
        1. **Context Window**: The number of tokens a model can process at once is limited (e.g., 4K, 8K, 16K tokens)
        
        2. **Cost**: Many API-based LLMs charge per token processed
        
        3. **Processing Efficiency**: Effective tokenization can reduce computational requirements
        
        4. **Language Support**: Different tokenization approaches work better for different languages
        
        5. **Out-of-Vocabulary Handling**: How models deal with unfamiliar words or terms
        
        ### Common Tokenization Approaches
        
        1. **Byte-Pair Encoding (BPE)** - Used by GPT models, starts with characters and merges common pairs
        
        2. **WordPiece** - Used by BERT, similar to BPE but uses likelihood rather than frequency
        
        3. **SentencePiece** - Language-agnostic approach, treats text as a sequence of unicode characters
        
        4. **Unigram Language Model** - Probabilistic approach used by some multilingual models
        """)
    
    # Fine-tuning & PEFT tab
    with tabs[2]:
        st.header("Fine-tuning & Parameter-Efficient Fine-Tuning (PEFT)")
        
        st.markdown("""
        Fine-tuning adapts pre-trained language models to specific tasks or domains. Traditional fine-tuning updates all model parameters, while Parameter-Efficient Fine-Tuning (PEFT) methods modify only a small subset.
        
        ### Fine-tuning Process
        """)
        
        # Visual representation of fine-tuning
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.markdown("#### Pre-trained Model")
            st.markdown("""
            - Trained on vast general data
            - Broad knowledge
            - General capabilities
            - Not specialized
            """)
        
        with col2:
            st.markdown("#### Fine-tuning Process")
            st.markdown("""
            1. Prepare task-specific dataset
            2. Initialize with pre-trained weights
            3. Train on new data with lower learning rate
            4. Evaluate on validation data
            5. Iterate until performance plateaus
            """)
        
        with col3:
            st.markdown("#### Fine-tuned Model")
            st.markdown("""
            - Specialized for task
            - Maintains general knowledge
            - Better performance on target domain
            - May lose some generalization
            """)
        
        st.markdown("---")
        
        st.subheader("Traditional Fine-tuning vs. PEFT")
        
        comparison_data = {
            "Aspect": [
                "Parameters Modified", 
                "Memory Required", 
                "Training Time", 
                "Storage Space", 
                "Performance Gain", 
                "Catastrophic Forgetting", 
                "Transferability"
            ],
            "Traditional Fine-tuning": [
                "All parameters (could be billions)", 
                "Very High", 
                "Long (hours to days)", 
                "Large (full model size)", 
                "High", 
                "More likely", 
                "Limited to similar tasks"
            ],
            "PEFT (e.g., LoRA)": [
                "Small subset (<1% of parameters)", 
                "Low", 
                "Short (minutes to hours)", 
                "Small (adapter size only)", 
                "Comparable to full fine-tuning", 
                "Less likely", 
                "Base model can be reused for multiple tasks"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.table(comparison_df)
        
        st.markdown("---")
        
        st.subheader("Popular PEFT Methods")
        
        peft_methods = {
            "LoRA (Low-Rank Adaptation)": {
                "description": "Adds low-rank matrices to existing weights, updating only these adapters during training",
                "advantages": "Memory efficient, fast training, good performance, composable adapters",
                "limitations": "Additional inference latency, needs careful rank selection",
                "example": """
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")

# Configure LoRA
lora_config = LoraConfig(
    r=8,                    # Rank of the update matrices
    lora_alpha=32,          # Scaling factor
    target_modules=["query", "value"],  # Which modules to apply LoRA to
    lora_dropout=0.05,      # Dropout probability
    bias="none"             # Whether to train bias parameters
)

# Create PEFT model
peft_model = get_peft_model(model, lora_config)

# Only LoRA parameters are trained
for name, param in peft_model.named_parameters():
    if "lora" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
                """
            },
            "Prefix Tuning": {
                "description": "Prepends trainable virtual tokens to input, keeping the original model frozen",
                "advantages": "Very parameter efficient, doesn't affect model architecture",
                "limitations": "Can be less effective than other methods for some tasks",
                "example": """
from peft import PrefixTuningConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")

# Configure Prefix Tuning
prefix_config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,   # Number of virtual tokens to add
    encoder_hidden_size=512,  # Dimension of hidden layer in prefix encoder
    prefix_projection=True    # Whether to project the prefix embeddings
)

# Create PEFT model
peft_model = get_peft_model(model, prefix_config)

# Only prefix parameters are trained
for name, param in peft_model.named_parameters():
    if "prefix" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
                """
            },
            "P-Tuning": {
                "description": "Inserts trainable continuous prompts into the input, optimizing them for specific tasks",
                "advantages": "Doesn't modify model weights at all, works well for classification tasks",
                "limitations": "Less effective for generation tasks",
                "example": """
from peft import PromptTuningConfig, get_peft_model, TaskType
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# Configure P-Tuning
ptuning_config = PromptTuningConfig(
    task_type=TaskType.SEQ_CLS,
    num_virtual_tokens=10,     # Number of virtual tokens
    token_dim=768,             # Must match model hidden size
    prompt_tuning_init="TEXT",  # Initialize from text
    prompt_tuning_init_text="Classify if this text is positive or negative:"
)

# Create PEFT model
peft_model = get_peft_model(model, ptuning_config)
                """
            }
        }
        
        selected_peft = st.selectbox("Select a PEFT method to explore", list(peft_methods.keys()))
        
        method_info = peft_methods[selected_peft]
        
        st.markdown(f"**{selected_peft}**")
        st.markdown(method_info["description"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Advantages**")
            st.markdown(method_info["advantages"])
        
        with col2:
            st.markdown("**Limitations**")
            st.markdown(method_info["limitations"])
        
        st.markdown("**Example Code**")
        st.code(method_info["example"])
        
        st.markdown("---")
        
        st.markdown("""
        ### When to Use Different Approaches
        
        **Use Full Fine-tuning When:**
        - You have substantial computational resources
        - Maximum performance is critical
        - The task differs significantly from the pre-training corpus
        - You want to create a new base model
        
        **Use PEFT When:**
        - You have limited computational resources
        - You want to quickly adapt models to multiple tasks
        - The task is reasonably similar to pre-training data
        - You want to avoid catastrophic forgetting
        - You need a smaller deployment footprint
        
        **Use LoRA Specifically When:**
        - You need a good balance of performance and efficiency
        - You plan to combine multiple adaptations (composition)
        - You want the fastest training times
        
        **Use Prompt/Prefix Tuning When:**
        - Extreme parameter efficiency is required
        - You want to preserve the base model exactly
        - You need the smallest possible adaptation
        """)
    
    # Decoding Strategies tab
    with tabs[3]:
        st.header("Decoding Strategies")
        
        st.markdown("""
        Decoding strategies control how language models generate text by determining which tokens to select from the probability distribution at each step. Different strategies produce different qualities of text.
        
        ### Why Decoding Strategies Matter
        
        The same model with different decoding strategies can generate drastically different outputs:
        - More creative vs. more focused text
        - More diverse vs. more coherent responses
        - Avoiding repetition and logical inconsistencies
        """)
        
        # Interactive demonstration
        st.subheader("Interactive Decoding Demo")
        
        # Sample prompt for demonstration
        default_prompt = "Once upon a time in a magical forest, there lived a"
        user_prompt = st.text_area("Prompt for generation", default_prompt)
        
        # Decoding parameters
        st.markdown("#### Decoding Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            strategy = st.selectbox(
                "Decoding Strategy", 
                ["Greedy Decoding", "Beam Search", "Top-K Sampling", "Top-p (Nucleus) Sampling", "Temperature Sampling", "Combined (Top-K + Temperature)"]
            )
            
            if strategy == "Beam Search":
                num_beams = st.slider("Number of Beams", min_value=2, max_value=10, value=4)
            
            if strategy == "Top-K Sampling":
                top_k = st.slider("Top-K", min_value=1, max_value=100, value=40)
            
            if strategy == "Top-p (Nucleus) Sampling":
                top_p = st.slider("Top-p", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
            
            if strategy == "Temperature Sampling" or strategy == "Combined (Top-K + Temperature)":
                temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=0.7, step=0.1)
                
            if strategy == "Combined (Top-K + Temperature)":
                combined_top_k = st.slider("Combined Top-K", min_value=1, max_value=100, value=50)
        
        with col2:
            st.markdown("#### Strategy Description")
            
            if strategy == "Greedy Decoding":
                st.write("Always selects the token with the highest probability at each step.")
                st.markdown("""
                - **Pros**: Deterministic, high precision
                - **Cons**: Often repetitive, can get stuck in loops
                - **Use for**: Factual generation, when creativity isn't needed
                """)
            
            elif strategy == "Beam Search":
                st.write(f"Keeps track of the {num_beams} most likely sequences at each step.")
                st.markdown("""
                - **Pros**: More coherent than greedy decoding
                - **Cons**: Still lacks diversity, computationally expensive
                - **Use for**: Translation, summarization, structured text
                """)
            
            elif strategy == "Top-K Sampling":
                st.write(f"Randomly samples from the top {top_k} most likely tokens at each step.")
                st.markdown("""
                - **Pros**: More diverse than greedy/beam, avoids low-probability tokens
                - **Cons**: Fixed K isn't adaptive to confidence distribution
                - **Use for**: Creative text, story generation
                """)
            
            elif strategy == "Top-p (Nucleus) Sampling":
                st.write(f"Samples from the smallest set of tokens whose cumulative probability exceeds {top_p}.")
                st.markdown("""
                - **Pros**: Adaptive to confidence distribution, balances quality and diversity
                - **Cons**: Can still generate low-probability tokens in some contexts
                - **Use for**: General-purpose text generation, creative tasks
                """)
            
            elif strategy == "Temperature Sampling":
                st.write(f"Adjusts the probability distribution using temperature of {temperature} before sampling.")
                st.markdown("""
                - **Pros**: Controls randomness/creativity with a single parameter
                - **Cons**: High temperatures can lead to nonsensical text
                - **Use for**: Controlling the creativity/randomness tradeoff
                """)
            
            elif strategy == "Combined (Top-K + Temperature)":
                st.write(f"Applies temperature of {temperature} and then samples from top {combined_top_k} tokens.")
                st.markdown("""
                - **Pros**: Benefits of both approaches, better control
                - **Cons**: More hyperparameters to tune
                - **Use for**: Production systems where quality control is important
                """)
        
        # Generate text button
        if st.button("Generate Text with Selected Strategy"):
            with st.spinner("Generating text..."):
                try:
                    # Set up the pipeline
                    generator = pipeline('text-generation', model='gpt2')
                    set_seed(42)  # For reproducibility
                    
                    # Set parameters based on selected strategy
                    params = {
                        "max_length": len(user_prompt.split()) + 50,
                        "num_return_sequences": 3
                    }
                    
                    if strategy == "Greedy Decoding":
                        params["do_sample"] = False
                        params["num_return_sequences"] = 1  # Only one possible output
                    
                    elif strategy == "Beam Search":
                        params["do_sample"] = False
                        params["num_beams"] = num_beams
                        params["no_repeat_ngram_size"] = 2  # Prevent repetition of bigrams
                    
                    elif strategy == "Top-K Sampling":
                        params["do_sample"] = True
                        params["top_k"] = top_k
                    
                    elif strategy == "Top-p (Nucleus) Sampling":
                        params["do_sample"] = True
                        params["top_p"] = top_p
                        params["top_k"] = 0  # Disable top-k when using top-p
                    
                    elif strategy == "Temperature Sampling":
                        params["do_sample"] = True
                        params["temperature"] = temperature
                        params["top_k"] = 0  # Disable top-k
                    
                    elif strategy == "Combined (Top-K + Temperature)":
                        params["do_sample"] = True
                        params["temperature"] = temperature
                        params["top_k"] = combined_top_k
                    
                    # Generate text
                    results = generator(user_prompt, **params)
                    
                    # Display results
                    st.subheader("Generated Text")
                    
                    for i, result in enumerate(results):
                        generated_text = result['generated_text'][len(user_prompt):]
                        st.markdown(f"**Sequence {i+1}**")
                        st.markdown(f"**Prompt:** {user_prompt}")
                        st.markdown(f"**Continuation:** {generated_text}")
                        st.markdown("---")
                except Exception as e:
                    st.error(f"Error generating text: {e}")
        
        st.markdown("---")
        
        st.markdown("""
        ### Visual Comparison of Decoding Strategies
        """)
        
        # Create a sample visualization of token probabilities
        probabilities = {
            "the": 0.25,
            "a": 0.20,
            "young": 0.15,
            "wise": 0.12,
            "mysterious": 0.10,
            "friendly": 0.08,
            "powerful": 0.05,
            "ancient": 0.03,
            "magical": 0.01,
            "talking": 0.01
        }
        
        # Convert to DataFrame for plotting
        prob_df = pd.DataFrame({
            'token': list(probabilities.keys()),
            'probability': list(probabilities.values())
        }).sort_values('probability', ascending=False)
        
        # Plot the probability distribution
        st.bar_chart(prob_df.set_index('token'))
        
        st.markdown("""
        #### How Different Strategies Would Select from this Distribution:
        
        - **Greedy**: Always selects "the" (p=0.25)
        
        - **Beam Search (k=2)**: Keeps "the" and "a" as candidates
        
        - **Top-K (k=3)**: Randomly samples from ["the", "a", "young"]
        
        - **Top-p (p=0.6)**: Randomly samples from ["the", "a", "young"] (cumulative p=0.60)
        
        - **Temperature (t=0.5)**: Makes "the" even more likely, reducing chances of low-probability tokens
        
        - **Temperature (t=2.0)**: Makes distribution more uniform, giving lower tokens more chance
        """)
    
    # Evaluation Metrics tab
    with tabs[4]:
        st.header("Evaluation Metrics")
        
        st.markdown("""
        Evaluating language models is complex because language quality is subjective. 
        Different metrics capture different aspects of model performance.
        
        ### Categories of Evaluation
        """)
        
        # Overview of evaluation approaches
        evaluation_types = {
            "Automatic Metrics": {
                "description": "Algorithmic measures that can be computed without human judgment",
                "examples": ["BLEU", "ROUGE", "METEOR", "BERTScore", "Perplexity"],
                "pros": "Fast, reproducible, scalable, objective",
                "cons": "Often don't correlate well with human judgment, can be gamed"
            },
            "Human Evaluation": {
                "description": "Direct assessment by human raters",
                "examples": ["Likert scales", "Pairwise comparisons", "Detailed rubrics", "Expert reviews"],
                "pros": "Better captures quality and nuance, more aligned with end user experience",
                "cons": "Expensive, time-consuming, subjective, harder to scale"
            },
            "Benchmark Datasets": {
                "description": "Standardized tasks and datasets for comparison",
                "examples": ["GLUE/SuperGLUE", "MMLU", "BIG-bench", "HumanEval", "TruthfulQA"],
                "pros": "Enables direct model comparison, covers diverse capabilities",
                "cons": "Can become saturated, may not reflect real-world performance"
            },
            "Capability Testing": {
                "description": "Evaluating specific abilities and limitations",
                "examples": ["Factuality", "Reasoning", "Safety", "Bias", "Robustness"],
                "pros": "Targeted assessment of important aspects, reveals specific weaknesses",
                "cons": "May not capture overall quality, requires careful test design"
            }
        }
        
        evaluation_type = st.selectbox("Select an evaluation approach", list(evaluation_types.keys()))
        
        eval_info = evaluation_types[evaluation_type]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**{evaluation_type}**")
            st.markdown(eval_info["description"])
            st.markdown("**Examples:**")
            st.markdown("\n".join([f"- {ex}" for ex in eval_info["examples"]]))
        
        with col2:
            st.markdown("**Advantages:**")
            st.markdown(eval_info["pros"])
            st.markdown("**Limitations:**")
            st.markdown(eval_info["cons"])
        
        st.markdown("---")
        
        # Detailed metrics
        st.subheader("Common Automatic Metrics")
        
        metrics = {
            "BLEU (Bilingual Evaluation Understudy)": {
                "description": "Measures overlap of n-grams between generated text and reference text",
                "best_for": "Machine translation, text generation with clear reference text",
                "interpretation": "Higher is better, scores range from 0 to 1 (or 0 to 100)",
                "limitations": "Doesn't capture semantic meaning, requires good reference, penalizes valid paraphrasing"
            },
            "ROUGE (Recall-Oriented Understudy for Gisting Evaluation)": {
                "description": "Family of metrics measuring overlap of n-grams, focused on recall",
                "best_for": "Summarization tasks, where coverage of key information is important",
                "interpretation": "Higher is better, scores range from 0 to 1",
                "limitations": "Multiple variants (ROUGE-1, ROUGE-2, ROUGE-L) measure different aspects"
            },
            "Perplexity": {
                "description": "Measures how well a model predicts a sample (inverse probability of the text)",
                "best_for": "Language modeling quality, intrinsic model evaluation",
                "interpretation": "Lower is better, theoretically 1 is perfect (but unachievable)",
                "limitations": "Only meaningful for comparing similar models on the same text, doesn't measure usefulness"
            },
            "BERTScore": {
                "description": "Uses contextual embeddings to measure semantic similarity",
                "best_for": "Semantic evaluation when exact wording isn't critical",
                "interpretation": "Higher is better, ranges from 0 to 1",
                "limitations": "Computationally expensive, results vary based on embedding model"
            },
            "Exact Match / F1": {
                "description": "Measures exact matches (EM) or token overlap (F1) for answers",
                "best_for": "Question answering tasks with clear correct answers",
                "interpretation": "Higher is better, ranges from 0 to 1 or 0 to 100%",
                "limitations": "Doesn't account for semantically correct but differently worded answers"
            }
        }
        
        metric = st.selectbox("Select a metric", list(metrics.keys()))
        
        metric_info = metrics[metric]
        
        st.markdown(f"**{metric}**")
        st.markdown(metric_info["description"])
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("**Best For:**")
            st.markdown(metric_info["best_for"])
            st.markdown("**Interpretation:**")
            st.markdown(metric_info["interpretation"])
        
        with col4:
            st.markdown("**Limitations:**")
            st.markdown(metric_info["limitations"])
        
        st.markdown("---")
        
        st.subheader("Evaluation Example")
        
        # Example text and references for evaluation
        generated_text = "The new phone has excellent camera quality and long battery life."
        reference_1 = "The latest smartphone features an impressive camera and extended battery performance."
        reference_2 = "The new phone boasts exceptional photo capabilities and strong battery endurance."
        
        st.markdown("**Generated Text:**")
        st.write(generated_text)
        
        st.markdown("**Reference Texts:**")
        st.write(f"Reference 1: {reference_1}")
        st.write(f"Reference 2: {reference_2}")
        
        # Simulated metrics calculation
        st.markdown("**Simulated Metrics:**")
        
        # Create simulated metrics results
        metrics_results = {
            "Metric": ["BLEU", "ROUGE-1", "ROUGE-L", "BERTScore"],
            "vs Reference 1": [0.42, 0.56, 0.48, 0.83],
            "vs Reference 2": [0.51, 0.67, 0.52, 0.88],
            "Average": [0.465, 0.615, 0.50, 0.855]
        }
        
        metrics_df = pd.DataFrame(metrics_results)
        st.table(metrics_df)
        
        st.markdown("""
        ### Holistic Evaluation Framework
        
        For comprehensive LLM evaluation, consider these dimensions:
        
        1. **Accuracy**: Factual correctness, absence of hallucinations
        
        2. **Relevance**: Appropriateness to the query or task
        
        3. **Coherence**: Logical flow and consistency
        
        4. **Fluency**: Natural language quality
        
        5. **Safety**: Avoidance of harmful, biased, or inappropriate content
        
        6. **Usefulness**: Practical value to the end user
        
        7. **Efficiency**: Response time, token usage, computational requirements
        
        The best evaluation approaches combine multiple metrics and human judgment,
        tailored to the specific use case and requirements.
        """)
    
    st.markdown("---")
    st.info("This module covered advanced concepts in working with LLMs. In the next module, we'll explore RAG (Retrieval-Augmented Generation) for enhancing LLM outputs with external knowledge.")
