#!/usr/bin/env python3
"""
Legal LLM Training Framework
A comprehensive system for training an LLM to interpret posts using legal codes and sample responses.
"""

import json
import torch
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import Dataset
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LegalCase:
    """Structure for storing legal cases/examples"""
    post: str
    relevant_codes: List[str]
    good_response: str
    metadata: Optional[Dict] = None

class LegalDataProcessor:
    """Handles processing of legal codes and training data"""
    
    def __init__(self, data_dir: str = "legal_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.legal_codes = {}
        self.training_cases = []
    
    def load_legal_codes(self, codes_file: str) -> Dict[str, str]:
        """Load legal codes from a JSON file"""
        try:
            with open(codes_file, 'r', encoding='utf-8') as f:
                self.legal_codes = json.load(f)
            logger.info(f"Loaded {len(self.legal_codes)} legal codes")
            return self.legal_codes
        except FileNotFoundError:
            logger.warning(f"Legal codes file {codes_file} not found. Creating template.")
            self.create_legal_codes_template(codes_file)
            return {}
    
    def create_legal_codes_template(self, filename: str):
        """Create a template file for legal codes"""
        template = {
            "CODE_001": "Definition of contract breach: A contract breach occurs when...",
            "CODE_002": "Privacy regulations: Personal data must be handled according to...",
            "CODE_003": "Intellectual property rights: Copyright protection extends to..."
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2)
        logger.info(f"Created legal codes template at {filename}")
    
    def load_training_cases(self, cases_file: str) -> List[LegalCase]:
        """Load training cases from JSON file"""
        try:
            with open(cases_file, 'r', encoding='utf-8') as f:
                cases_data = json.load(f)
            
            self.training_cases = [
                LegalCase(
                    post=case['post'],
                    relevant_codes=case['relevant_codes'],
                    good_response=case['good_response'],
                    metadata=case.get('metadata', {})
                )
                for case in cases_data
            ]
            logger.info(f"Loaded {len(self.training_cases)} training cases")
            return self.training_cases
        except FileNotFoundError:
            logger.warning(f"Training cases file {cases_file} not found. Creating template.")
            self.create_training_cases_template(cases_file)
            return []
    
    def create_training_cases_template(self, filename: str):
        """Create a template file for training cases"""
        template = [
            {
                "post": "I think my employer is violating my privacy by reading my emails.",
                "relevant_codes": ["CODE_002"],
                "good_response": "Based on privacy regulations (CODE_002), employers generally have limited rights to monitor employee communications. However, this depends on your employment contract and local laws. I recommend reviewing your employee handbook and consulting with HR or legal counsel.",
                "metadata": {"category": "employment_privacy", "difficulty": "medium"}
            },
            {
                "post": "Someone copied my website design exactly. Can I sue them?",
                "relevant_codes": ["CODE_003"],
                "good_response": "This may constitute intellectual property infringement under CODE_003. Website designs can be protected by copyright if they meet originality requirements. You should document the similarities and consult with an intellectual property attorney to evaluate your case.",
                "metadata": {"category": "ip_infringement", "difficulty": "high"}
            }
        ]
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2)
        logger.info(f"Created training cases template at {filename}")

class LegalLLMTrainer:
    """Main class for training the legal interpretation LLM"""
    
    def __init__(self, model_name: str = "distilgpt2", max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.data_processor = LegalDataProcessor()
    
    def setup_model(self):
        """Initialize tokenizer and model"""
        logger.info(f"Loading model: {self.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Explicit dtype for compatibility
                trust_remote_code=False     # Security best practice
            )
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            logger.info("Trying alternative model: gpt2")
            try:
                self.model_name = "gpt2"
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                self.model = AutoModelForCausalLM.from_pretrained("gpt2")
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Alternative model loaded successfully")
            except Exception as e2:
                logger.error(f"Failed to load alternative model: {e2}")
                raise
    
    def create_training_prompt(self, case: LegalCase) -> str:
        """Create a structured prompt for training"""
        # Get relevant legal code text
        relevant_code_text = ""
        for code_id in case.relevant_codes:
            if code_id in self.data_processor.legal_codes:
                relevant_code_text += f"{code_id}: {self.data_processor.legal_codes[code_id]}\n"
        
        prompt = f"""Legal Codes:
                    {relevant_code_text}

                    User Post: {case.post}

                    Legal Analysis: {case.good_response}"""
        
        return prompt
    
    def prepare_dataset(self, cases: List[LegalCase]) -> Dataset:
        """Prepare dataset for training"""
        prompts = [self.create_training_prompt(case) for case in cases]
        
        # Tokenize the prompts
        encodings = self.tokenizer(
            prompts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Create dataset
        dataset = Dataset.from_dict({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"]
        })
        
        return dataset
    
    def train_model(self, output_dir: str = "legal_llm_model", num_epochs: int = 3):
        """Train the model on legal cases"""
        if not self.training_cases:
            logger.error("No training cases loaded. Please load training data first.")
            return
        
        logger.info("Preparing dataset...")
        dataset = self.prepare_dataset(self.data_processor.training_cases)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            warmup_steps=100,
            logging_steps=50,
            save_steps=500,
            evaluation_strategy="no",
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )
        
        logger.info("Starting training...")
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")
    
    def load_trained_model(self, model_dir: str):
        """Load a previously trained model"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        logger.info(f"Loaded trained model from {model_dir}")
    
    def analyze_post(self, post: str, relevant_codes: List[str] = None) -> str:
        """Generate legal analysis for a post"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Please setup or load a trained model first.")
        
        # Create prompt
        relevant_code_text = ""
        if relevant_codes:
            for code_id in relevant_codes:
                if code_id in self.data_processor.legal_codes:
                    relevant_code_text += f"{code_id}: {self.data_processor.legal_codes[code_id]}\n"
        
        prompt = f"""Legal Codes:
{relevant_code_text}

User Post: {post}

Legal Analysis:"""
        
        # Tokenize and generate
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the generated part
        response = full_response[len(prompt):].strip()
        return response

def main():
    """Main function demonstrating usage"""
    # Initialize the trainer
    trainer = LegalLLMTrainer()
    
    # Setup model
    trainer.setup_model()
    
    # Load legal codes and training cases
    trainer.data_processor.load_legal_codes("legal_codes.json")
    trainer.data_processor.load_training_cases("training_cases.json")
    
    # Train the model (uncomment to train)
    # trainer.train_model()
    
    # Example usage for inference
    sample_post = "My landlord is trying to evict me without proper notice."
    relevant_codes = ["CODE_001"]  # Assuming you have relevant codes
    
    try:
        analysis = trainer.analyze_post(sample_post, relevant_codes)
        print(f"Legal Analysis: {analysis}")
    except ValueError as e:
        print(f"Error: {e}")
        print("Please train the model first or load a pre-trained model.")

if __name__ == "__main__":
    # Create example usage script
    
    main()