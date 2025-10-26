"""
Model Card validation logic
"""
from typing import List
from ai_standards.certification.model_card import ModelCard


class ValidationResult:
    """Result of model card validation"""
    
    def __init__(self):
        self.is_valid = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def add_error(self, error: str):
        """Add validation error"""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add validation warning"""
        self.warnings.append(warning)
    
    def __repr__(self):
        status = "VALID" if self.is_valid else "INVALID"
        return f"ValidationResult(status={status}, errors={len(self.errors)}, warnings={len(self.warnings)})"


class ModelCardValidator:
    """
    Validates model cards against certification requirements
    """
    
    def __init__(self):
        self.min_model_size = 1_000_000  # Minimum 1M parameters
        self.required_license_keywords = ["license", "terms", "permission"]
    
    def validate(self, model_card: ModelCard) -> ValidationResult:
        """
        Comprehensive validation of model card.
        
        Args:
            model_card: ModelCard to validate
            
        Returns:
            ValidationResult with errors and warnings
        """
        result = ValidationResult()
        
        # Validate fixed parameters
        self._validate_usage_instructions(model_card, result)
        self._validate_model_size(model_card, result)
        self._validate_licensing(model_card, result)
        self._validate_architecture(model_card, result)
        
        # Check optional parameters (warnings only)
        self._check_optional_parameters(model_card, result)
        
        # Validate consistency
        self._validate_consistency(model_card, result)
        
        return result
    
    def _validate_usage_instructions(self, card: ModelCard, result: ValidationResult):
        """Validate usage instructions are adequate"""
        if not card.usage_instructions or len(card.usage_instructions) < 50:
            result.add_error(
                "Usage instructions must be detailed (at least 50 characters)"
            )
    
    def _validate_model_size(self, card: ModelCard, result: ValidationResult):
        """Validate model size is reasonable"""
        if card.model_size < self.min_model_size:
            result.add_warning(
                f"Model size ({card.model_size:,}) is below recommended minimum "
                f"({self.min_model_size:,} parameters)"
            )
        
        if card.model_size > 1_000_000_000_000:  # 1T parameters
            result.add_warning(
                f"Model size ({card.model_size:,}) is extremely large - verify accuracy"
            )
    
    def _validate_licensing(self, card: ModelCard, result: ValidationResult):
        """Validate licensing terms are specified"""
        if not card.licensing_terms or len(card.licensing_terms) < 10:
            result.add_error(
                "Licensing terms must be explicitly specified"
            )
    
    def _validate_architecture(self, card: ModelCard, result: ValidationResult):
        """Validate architecture is specified"""
        if card.architecture is None:
            result.add_error("Model architecture must be specified")
    
    def _check_optional_parameters(self, card: ModelCard, result: ValidationResult):
        """Check if optional parameters are provided (best practices)"""
        if not card.training_data_sources:
            result.add_warning(
                "Training data sources not specified - recommended for transparency"
            )
        
        if not card.ethical_considerations:
            result.add_warning(
                "Ethical considerations not specified - recommended for responsible AI"
            )
        
        if not card.intended_use_cases:
            result.add_warning(
                "Intended use cases not specified - helps users understand scope"
            )
        
        if not card.limitations:
            result.add_warning(
                "Model limitations not specified - important for setting expectations"
            )
    
    def _validate_consistency(self, card: ModelCard, result: ValidationResult):
        """Validate internal consistency of model card"""
        # Check that token limits are reasonable
        if card.token_limits:
            if "max_input_tokens" in card.token_limits:
                if card.token_limits["max_input_tokens"] <= 0:
                    result.add_error("max_input_tokens must be positive")
            
            if "max_output_tokens" in card.token_limits:
                if card.token_limits["max_output_tokens"] <= 0:
                    result.add_error("max_output_tokens must be positive")
        
        # Validate costs are non-negative
        if card.associated_costs:
            for key, value in card.associated_costs.items():
                if isinstance(value, (int, float)) and value < 0:
                    result.add_error(f"Cost '{key}' cannot be negative")

