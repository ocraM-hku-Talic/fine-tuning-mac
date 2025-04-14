#!/usr/bin/env python
import argparse
import subprocess
import json
import os
from mlx_lm import generate, load

def generate_with_mlx(prompt, max_tokens=100, temp=0.7):
    """Generate text using the MLX fine-tuned model"""
    print("=== MLX Fine-tuned Model ===")
    model_path = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    adapter_path = "./adapters"
    
    print(f"Loading model {model_path} with adapter from {adapter_path}...")
    
    try:
        model, tokenizer = load(model_path, adapter_path=adapter_path)
        
        print(f"Generating response for prompt: {prompt}")
        
        # Remove temperature parameter as it's not supported
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens
            # temperature parameter removed
        )
        
        print("\nMLX Fine-tuned Response:")
        print(response)
        return response
    except Exception as e:
        print(f"Error with MLX generation: {e}")
        return None

def generate_with_ollama(prompt, model_name="deepseek-r1:8b", max_tokens=100, temp=0.7):
    """Generate text using Ollama"""
    print(f"\n=== Ollama Model ({model_name}) ===")
    try:
        cmd = [
            "ollama", "run", model_name, 
            "--format", "json",
            "--options", json.dumps({
                "num_predict": max_tokens,
                "temperature": temp
            }),
            prompt
        ]
        
        print(f"Running Ollama with model: {model_name}")
        print(f"Prompt: {prompt}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error running Ollama: {result.stderr}")
            return None
        
        # Parse the output - Ollama may output multiple JSON objects
        full_response = ""
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            try:
                data = json.loads(line)
                if "response" in data:
                    full_response += data["response"]
            except json.JSONDecodeError:
                full_response += line
        
        print("\nOllama Response:")
        print(full_response)
        return full_response
    except Exception as e:
        print(f"Error with Ollama generation: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Compare MLX fine-tuned model with Ollama models")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt to generate from")
    parser.add_argument("--ollama-model", type=str, default="deepseek-r1:8b", 
                        help="The Ollama model to use (must be pulled first with 'ollama pull <model>')")
    parser.add_argument("--max-tokens", type=int, default=10000, help="Maximum number of tokens to generate")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--mlx-only", action="store_true", help="Only run the MLX model")
    parser.add_argument("--ollama-only", action="store_true", help="Only run the Ollama model")
    
    args = parser.parse_args()

    # Check if Ollama model exists
    if not args.mlx_only:
        print(f"Checking if Ollama model '{args.ollama_model}' exists...")
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if args.ollama_model not in result.stdout:
            print(f"Ollama model '{args.ollama_model}' not found. You may need to pull it first with:")
            print(f"  ollama pull {args.ollama_model}")
            if not args.ollama_only:
                print("Continuing with MLX model only.")
                args.ollama_only = False
            else:
                return

    mlx_response = None
    ollama_response = None
    
    if not args.ollama_only:
        mlx_response = generate_with_mlx(args.prompt, args.max_tokens, args.temp)
    
    if not args.mlx_only:
        ollama_response = generate_with_ollama(args.prompt, args.ollama_model, args.max_tokens, args.temp)

    if mlx_response and ollama_response:
        print("\n=== Comparison ===")
        print("You can compare the responses to see how your fine-tuned model differs from the Ollama model.")

if __name__ == "__main__":
    main()