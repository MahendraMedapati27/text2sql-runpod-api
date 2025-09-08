#!/usr/bin/env python3
"""
Local testing script for the Text2SQL model handler.
Run this to test your model locally before deploying to RunPod.
"""

import json
from handler import handler, load_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_handler():
    """Test the handler function locally."""
    
    # Load model first
    print("Loading model...")
    load_model()
    print("Model loaded successfully!")
    
    # Test cases
    test_cases = [
        {
            "query": "Show me all users from the database",
            "max_length": 256,
            "temperature": 0.7
        },
        {
            "query": "Get the count of orders for each customer",
            "max_length": 512,
            "temperature": 0.5
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Input: {test_case['query']}")
        
        # Create job format
        job = {"input": test_case}
        
        # Run handler
        result = handler(job)
        
        print(f"Result: {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    test_handler()
