import runpod
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import logging
import gc
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
MODEL = None
TOKENIZER = None
MODEL_NAME = "MahendraMedapati27/Text2SQL_Model"

def load_model():
    """Load the model and tokenizer."""
    global MODEL, TOKENIZER
    
    try:
        logger.info(f"Loading model: {MODEL_NAME}")
        
        # Load tokenizer
        TOKENIZER = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            token=os.getenv("HUGGINGFACE_TOKEN")  # Optional: if your model is private
        )
        
        # Add padding token if it doesn't exist
        if TOKENIZER.pad_token is None:
            TOKENIZER.pad_token = TOKENIZER.eos_token
        
        # Load model with optimizations
        MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,  # Use half precision for memory efficiency
            device_map="auto",
            trust_remote_code=True,
            token=os.getenv("HUGGINGFACE_TOKEN")  # Optional: if your model is private
        )
        
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e

def generate_sql(input_text, max_length=512, temperature=0.7, top_p=0.9):
    """Generate SQL query from natural language input."""
    try:
        # Tokenize input
        inputs = TOKENIZER(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        ).to(MODEL.device)
        
        # Generate response
        with torch.no_grad():
            outputs = MODEL.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=TOKENIZER.eos_token_id,
                eos_token_id=TOKENIZER.eos_token_id,
            )
        
        # Decode the generated text
        generated_text = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (remove input prompt)
        if input_text in generated_text:
            sql_query = generated_text.replace(input_text, "").strip()
        else:
            sql_query = generated_text.strip()
        
        return sql_query
        
    except Exception as e:
        logger.error(f"Error generating SQL: {str(e)}")
        return f"Error: {str(e)}"

def handler(job):
    """Main handler function for RunPod serverless."""
    try:
        # Get input from job
        job_input = job["input"]
        
        # Extract parameters
        query = job_input.get("query", "")
        max_length = job_input.get("max_length", 512)
        temperature = job_input.get("temperature", 0.7)
        top_p = job_input.get("top_p", 0.9)
        
        # Validate input
        if not query:
            return {"error": "No query provided"}
        
        # Generate SQL
        logger.info(f"Processing query: {query[:100]}...")
        sql_result = generate_sql(query, max_length, temperature, top_p)
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return {
            "sql_query": sql_result,
            "input_query": query,
            "parameters": {
                "max_length": max_length,
                "temperature": temperature,
                "top_p": top_p
            }
        }
        
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        return {"error": str(e)}

# Initialize the model when the container starts
if __name__ == "__main__":
    logger.info("Initializing Text2SQL model...")
    load_model()
    logger.info("Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
