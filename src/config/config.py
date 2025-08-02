"""
Configuration management for the RAG application.
Centralizes all configuration settings and API key management.
"""
from typing import Optional
from dataclasses import dataclass


@dataclass
class Config:
    """
    Application configuration class.
    Stores all configuration parameters in a centralized location.
    """
    # OpenAI settings
    openai_api_key: Optional[str] = None
    embedding_model: str = "text-embedding-ada-002"
    chat_model: str = "gpt-4o"
    
    # Document processing settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Directory settings
    input_files_dir: str = "input_files"
    processed_files_dir: str = "processed_files"
    faiss_index_path: str = "faiss_index"
    
    # Search settings
    default_search_k: int = 4
    default_fetch_k: int = 20
    default_lambda_mult: float = 0.5
    
    # Model temperature
    temperature: float = 0.7
    seed: int = 42
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: float = 30.0
    max_retries: int = 3

    def update_api_key(self, api_key: str):
        """Update the OpenAI API key."""
        self.openai_api_key = api_key
    
    def update_chat_model(self, model: str):
        """Update the chat model."""
        self.chat_model = model
    
    def update_temperature(self, temperature: float) -> tuple[bool, str]:
        """
        Update model temperature with validation.
        
        Args:
            temperature: New temperature value (0.0 to 2.0)
            
        Returns:
            tuple: (success, error_message)
        """
        # Store original value for rollback
        original_temperature = self.temperature
        
        # Update value
        self.temperature = temperature
        
        # Validate the new configuration
        is_valid, error_msg = self._validate_temperature()
        
        if not is_valid:
            # Rollback to original value
            self.temperature = original_temperature
            return False, error_msg
        
        return True, ""
    
    def _validate_temperature(self) -> tuple[bool, str]:
        """
        Validate temperature parameter.
        
        Returns:
            tuple: (is_valid, error_message)
        """
        if self.temperature < 0.0:
            return False, "Temperature must be non-negative"
        
        if self.temperature > 2.0:
            return False, "Temperature must be 2.0 or less"
        
        return True, ""
    
    def update_chunk_config(self, chunk_size: int = None, chunk_overlap: int = None) -> tuple[bool, str]:
        """
        Update chunk size and/or overlap with validation.
        
        Args:
            chunk_size: New chunk size (optional)
            chunk_overlap: New chunk overlap (optional)
            
        Returns:
            tuple: (success, error_message)
        """
        # Store original values for rollback
        original_size = self.chunk_size
        original_overlap = self.chunk_overlap
        
        # Update values if provided
        if chunk_size is not None:
            self.chunk_size = chunk_size
        if chunk_overlap is not None:
            self.chunk_overlap = chunk_overlap
        
        # Validate the new configuration
        is_valid, error_msg = self._validate_chunk_config()
        
        if not is_valid:
            # Rollback to original values
            self.chunk_size = original_size
            self.chunk_overlap = original_overlap
            return False, error_msg
        
        return True, ""
    
    def _validate_chunk_config(self) -> tuple[bool, str]:
        """
        Validate chunk configuration parameters.
        
        Returns:
            tuple: (is_valid, error_message)
        """
        if self.chunk_size <= 0:
            return False, "Chunk size must be positive"
        
        if self.chunk_overlap < 0:
            return False, "Chunk overlap cannot be negative"
        
        if self.chunk_overlap >= self.chunk_size:
            return False, "Chunk overlap must be less than chunk size"
        
        return True, ""
    
    def is_configured(self) -> bool:
        """Check if the essential configuration is set."""
        return bool(self.openai_api_key and self.openai_api_key.startswith("sk-"))
    
    def validate(self) -> tuple[bool, str]:
        """
        Validate the configuration.
        
        Returns:
            tuple: (is_valid, error_message)
        """
        if not self.openai_api_key:
            return False, "OpenAI API key is required"
        
        if not self.openai_api_key.startswith("sk-"):
            return False, "Invalid OpenAI API key format"
        
        # Validate chunk configuration
        chunk_valid, chunk_error = self._validate_chunk_config()
        if not chunk_valid:
            return False, chunk_error
        
        return True, ""