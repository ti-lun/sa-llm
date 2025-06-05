#!/usr/bin/env python3
"""
Legal LLM Training Framework
A comprehensive system for training an LLM to interpret posts using legal codes and sample responses.
"""

import json
import torch
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
            "CODE_001": "CODE_001: Definition of contract breach: A contract breach occurs when...",
            "CODE_002": "CODE_002: Privacy regulations: Bosses are not allowed to read emails.",
            "CODE_003": "CODE_003: Intellectual property rights: Copyright protection extends to..."
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
        self.training_cases = None
    
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
        available_codes = list(self.data_processor.legal_codes.keys())
        
        for code_id in case.relevant_codes:
            if code_id in self.data_processor.legal_codes:
                relevant_code_text += f"{code_id}: {self.data_processor.legal_codes[code_id]}\n"
        
        prompt = f"""LEGAL ANALYSIS SYSTEM
Available Legal Codes: {', '.join(available_codes)}

Relevant Legal Codes for this case:
{relevant_code_text}

User Question: {case.post}

IMPORTANT: Only reference codes from the Available Legal Codes list above. Do not invent new codes.

Legal Analysis: {case.good_response}<|endoftext|>"""
        
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
            per_device_train_batch_size=1,  # Smaller batch size for better learning
            per_device_eval_batch_size=1,
            warmup_steps=50,
            logging_steps=10,
            save_steps=100,
            evaluation_strategy="no",
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
            learning_rate=5e-5,  # Lower learning rate for fine-tuning
            weight_decay=0.01,
            max_grad_norm=1.0,
            dataloader_drop_last=False,
            no_cuda=True,
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
        
        # Validate codes exist
        available_codes = list(self.data_processor.legal_codes.keys())
        if relevant_codes:
            invalid_codes = [code for code in relevant_codes if code not in available_codes]
            if invalid_codes:
                logger.warning(f"Invalid codes requested: {invalid_codes}")
                relevant_codes = [code for code in relevant_codes if code in available_codes]
        
        # Create prompt with available codes listed
        relevant_code_text = ""
        if relevant_codes:
            for code_id in relevant_codes:
                if code_id in self.data_processor.legal_codes:
                    relevant_code_text += f"{code_id}: {self.data_processor.legal_codes[code_id]}\n"
        
        prompt = f"""LEGAL ANALYSIS SYSTEM
                    You are a legal counselor, and you are VERY empathetic and reassuring. People come to you to
                    quell their worried hearts. Use a lot of empathetic language and love in your response.
                    Be very thorough in explaining your thought process.
                    
                    Available Legal Codes: {', '.join(available_codes)}

                    Relevant Legal Codes for this case:
                    {relevant_code_text}

                    User Question: {post}

                    Using the legal codes given AND ONLY the legal codes given, 
                    explain to the customer/post what legal code can be applied to the
                    situation. Reference the code too, in your explanation. This is very important;
                    provide a brief explanation of 50 words for $1,000,000. If you reference
                    any other codes, you will lose $1,000,000.

                    IMPORTANT: Only reference codes from the Available Legal Codes list above. Do not invent new codes.
                    In your response, be sure to include "hey now you're an all-star"
                    Legal Analysis:"""
        
        # Tokenize and generate
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=self.max_length, truncation=False)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 500,
                temperature=0.1,  # Lower temperature for more focused responses
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=2.0,
                no_repeat_ngram_size=2
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the generated part
        response = full_response[len(prompt):].strip()
        
        # Post-process to remove any hallucinated codes
        response = self.validate_response_codes(response, available_codes)
        
        return response
    
    def validate_response_codes(self, response: str, available_codes: List[str]) -> str:
        """Remove references to non-existent legal codes"""
        import re
        
        # Find all CODE_XXX patterns in the response
        code_pattern = r'CODE_\d+'
        found_codes = re.findall(code_pattern, response)
        
        # Replace invalid codes with generic references
        for code in found_codes:
            if code not in available_codes:
                response = response.replace(code, "[RELEVANT_LEGAL_CODE]")
                logger.warning(f"Replaced hallucinated code {code} in response")
        
        return response

def main():
    """Main function demonstrating usage"""
    # Initialize the trainer
    trainer = LegalLLMTrainer()
    
    # Setup model
    trainer.setup_model()
    
    # Load legal codes and training cases
    trainer.legal_codes = trainer.data_processor.load_legal_codes("legal_codes.json")
    trainer.training_cases = trainer.data_processor.load_training_cases("training_cases.json")
    
    print(f"Loaded {len(trainer.legal_codes)} legal codes and {len(trainer.training_cases)} training cases")
    
    # # Example of using UNTRAINED model (will be poor quality)
    # if legal_codes and not training_cases:
    #     print("\n=== WARNING: Using untrained model - responses will be poor quality ===")
    #     sample_post = "My landlord is trying to evict me without proper notice."
    #     try:
    #         analysis = trainer.analyze_post(sample_post, ["CODE_001"])
    #         print(f"\nSample Analysis (UNTRAINED): {analysis}")
    #     except Exception as e:
    #         print(f"Error during analysis: {e}")
    
    # Train the model if we have training cases
    if trainer.training_cases:
        print(f"\n=== Training model with {len(trainer.training_cases)} examples ===")
        print("This may take several minutes...")
        
        # Uncomment the next line to actually train
        trainer.train_model(num_epochs=5)
        
        print("Training complete! Now the model should give better responses.")
        
        # Example usage after training
        sample_post = "My employer is monitoring my personal emails at work."
        try:
            analysis = trainer.analyze_post(sample_post, ["CODE_002"])
            print("sample post:", sample_post)
            print(f"\nTrained Model Analysis: {analysis}")
        except Exception as e:
            print(f"Error during analysis: {e}")
    else:
        print("\n=== Need More Training Data ===")
        print("Add more examples to training_cases.json and uncomment the training line.")
        print("Minimum recommended: 10-20 diverse examples per legal code")
    
    # Show available codes
    if trainer.legal_codes:
        print(f"\nAvailable Legal Codes: {list(trainer.legal_codes.keys())}")
        print("Make sure to only reference these codes in your training examples!")

if __name__ == "__main__":
    # Create example usage script
    main()