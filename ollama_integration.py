#!/usr/bin/env python3
"""
Script to use your fine-tuned model with Ollama by setting up a proxy API endpoint
that applies LoRA weights before generating responses
"""

import argparse
import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import mlx.core as mx
from mlx_lm import load, generate
from typing import List, Dict, Any, Optional, Union
import threading

# Default port for the proxy server
DEFAULT_PORT = 8765

def parse_args():
    parser = argparse.ArgumentParser(description="Serve your fine-tuned MLX model through a proxy API")
    parser.add_argument(
        "--model", 
        type=str, 
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Base model name/path"
    )
    parser.add_argument(
        "--adapter", 
        type=str, 
        default="./adapters",
        help="Path to adapter weights"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=DEFAULT_PORT,
        help=f"Port to run the server on (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.2,
        help="Temperature for generation (default: 0.2)"
    )
    parser.add_argument(
        "--max_tokens", 
        type=int, 
        default=1024,
        help="Maximum number of tokens to generate (default: 1024)"
    )
    return parser.parse_args()

class ModelServer:
    def __init__(self, model_path: str, adapter_path: str, temperature: float = 0.2, max_tokens: int = 1024):
        print(f"Loading model {model_path} with adapter {adapter_path}...")
        self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)
        self.temperature = temperature
        self.max_tokens = max_tokens
        print("Model loaded successfully!")
        
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        # Prepare the input prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        else:
            full_prompt = prompt
            
        print(f"Generating response for prompt: {full_prompt[:50]}...")
        
        # Generate using MLX
        results = generate(
            self.model,
            self.tokenizer,
            prompt=full_prompt,
            max_tokens=self.max_tokens,
            temp=self.temperature,
        )
        
        return results[0]

class ProxyHandler(BaseHTTPRequestHandler):
    def __init__(self, model_server, *args, **kwargs):
        self.model_server = model_server
        super().__init__(*args, **kwargs)
    
    def _set_headers(self, status_code=200):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
    def do_OPTIONS(self):
        self._set_headers()
        
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode('utf-8'))
            
            # Handle different API endpoints
            if self.path == '/api/generate':
                prompt = data.get('prompt', '')
                system = data.get('system', None)
                response = self.model_server.generate(prompt, system)
                
                response_data = {
                    'model': 'fine-tuned-deepseek',
                    'response': response,
                    'done': True
                }
                
                self._set_headers()
                self.wfile.write(json.dumps(response_data).encode('utf-8'))
            else:
                self._set_headers(404)
                self.wfile.write(json.dumps({'error': 'Not found'}).encode('utf-8'))
                
        except Exception as e:
            self._set_headers(500)
            self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))

def run_server(port, model_server):
    # Create a handler that includes the model server
    handler = lambda *args, **kwargs: ProxyHandler(model_server, *args, **kwargs)
    
    # Start the server
    server = HTTPServer(('localhost', port), handler)
    print(f"Starting server at http://localhost:{port}")
    server.serve_forever()

if __name__ == "__main__":
    args = parse_args()
    
    # Initialize the model server
    model_server = ModelServer(
        args.model, 
        args.adapter,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    # Run the server
    run_server(args.port, model_server)
