"""
Certification Manager - handles model card lifecycle and blockchain integration
"""
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

from ai_standards.certification.model_card import ModelCard
from ai_standards.certification.validator import ModelCardValidator
from ai_standards.utils.crypto import hash_model_card, generate_model_id
from ai_standards.utils.blockchain import BlockchainConnector


class CertificationManager:
    """
    Manages the certification lifecycle for AI models.
    
    Responsibilities:
    - Create and validate model cards
    - Mint NFTs on blockchain
    - Track certification status
    - Store and retrieve model cards
    """
    
    def __init__(
        self,
        blockchain_connector: Optional[BlockchainConnector] = None,
        storage_path: Optional[Path] = None
    ):
        self.validator = ModelCardValidator()
        self.blockchain = blockchain_connector or BlockchainConnector(use_mock=True)
        self.storage_path = storage_path or Path("./model_cards")
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def create_model_card(
        self,
        model_name: str,
        model_size: int,
        architecture: str,
        output_modality: str,
        usage_instructions: str,
        licensing_terms: str,
        **kwargs
    ) -> ModelCard:
        """
        Create a new model card.
        
        Args:
            model_name: Name of the model
            model_size: Number of parameters
            architecture: Model architecture type
            output_modality: Output modality (discrete, text, vision, audio)
            usage_instructions: How to use the model
            licensing_terms: License information
            **kwargs: Additional optional parameters
            
        Returns:
            ModelCard instance
        """
        model_id = kwargs.get('model_id') or generate_model_id(model_name)
        
        model_card = ModelCard(
            model_id=model_id,
            model_name=model_name,
            model_size=model_size,
            architecture=architecture,
            output_modality=output_modality,
            usage_instructions=usage_instructions,
            licensing_terms=licensing_terms,
            **kwargs
        )
        
        # Generate hash
        model_card.card_hash = hash_model_card(model_card.to_dict())
        
        return model_card
    
    def validate_model_card(self, model_card: ModelCard):
        """
        Validate a model card.
        
        Args:
            model_card: ModelCard to validate
            
        Returns:
            ValidationResult
        """
        return self.validator.validate(model_card)
    
    def certify_model(
        self,
        model_card: ModelCard,
        mint_nft: bool = True
    ) -> Dict[str, Any]:
        """
        Certify a model by validating and optionally minting NFT.
        
        Args:
            model_card: ModelCard to certify
            mint_nft: Whether to mint NFT on blockchain
            
        Returns:
            Certification result with status and details
        """
        # Validate first
        validation_result = self.validate_model_card(model_card)
        
        if not validation_result.is_valid:
            return {
                "success": False,
                "errors": validation_result.errors,
                "warnings": validation_result.warnings,
            }
        
        # Mark as certified
        certification_metadata = {
            "validation_warnings": validation_result.warnings,
            "validator_version": "1.0.0",
        }
        model_card.mark_certified(certification_metadata)
        
        # Mint NFT if requested
        nft_id = None
        if mint_nft:
            nft_metadata = self._prepare_nft_metadata(model_card)
            nft_id = self.blockchain.mint_nft(nft_metadata)
            model_card.nft_id = nft_id
        
        # Save to storage
        self._save_model_card(model_card)
        
        return {
            "success": True,
            "model_id": model_card.model_id,
            "nft_id": nft_id,
            "warnings": validation_result.warnings,
            "certified": True,
        }
    
    def _prepare_nft_metadata(self, model_card: ModelCard) -> Dict[str, Any]:
        """Prepare metadata for NFT minting"""
        return {
            "name": f"Model Card: {model_card.model_name}",
            "description": f"Certified AI Model Card for {model_card.model_name}",
            "model_card": model_card.to_dict(),
            "hash": model_card.card_hash,
        }
    
    def _save_model_card(self, model_card: ModelCard):
        """Save model card to local storage"""
        file_path = self.storage_path / f"{model_card.model_id}.json"
        with open(file_path, 'w') as f:
            json.dump(model_card.to_dict(), f, indent=2, default=str)
    
    def load_model_card(self, model_id: str) -> Optional[ModelCard]:
        """Load model card from storage"""
        file_path = self.storage_path / f"{model_id}.json"
        if not file_path.exists():
            return None
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return ModelCard.from_dict(data)
    
    def list_model_cards(self) -> List[str]:
        """List all stored model card IDs"""
        return [
            f.stem for f in self.storage_path.glob("*.json")
        ]
    
    def get_model_card_from_blockchain(self, nft_id: str) -> Optional[ModelCard]:
        """Retrieve model card from blockchain via NFT ID"""
        nft_data = self.blockchain.get_model_card(nft_id)
        if nft_data and "model_card" in nft_data:
            return ModelCard.from_dict(nft_data["model_card"])
        return None

