"""
Example 1: Model Certification

This example demonstrates how to:
1. Create a model card
2. Validate it
3. Certify the model
4. Mint an NFT on blockchain
"""

from ai_standards.certification import CertificationManager

def main():
    print("=" * 60)
    print("Example 1: Model Certification")
    print("=" * 60)
    
    # Initialize certification manager
    manager = CertificationManager()
    
    # Step 1: Create a model card
    print("\n1. Creating model card...")
    model_card = manager.create_model_card(
        model_name="Sentiment Analyzer v1",
        model_size=110_000_000,  # 110M parameters
        architecture="transformer",
        output_modality="discrete",
        usage_instructions=(
            "This model performs sentiment analysis on text input. "
            "Send POST request to /predict with JSON: {'text': 'your text'}. "
            "Returns: {'sentiment': 'positive'|'negative'|'neutral', 'confidence': float}"
        ),
        licensing_terms="MIT License - Free to use, modify, and distribute",
        training_data_sources=["IMDB reviews", "Twitter sentiment dataset"],
        ethical_considerations="Model may exhibit bias on underrepresented demographics",
        intended_use_cases=[
            "Customer feedback analysis",
            "Social media monitoring",
            "Product review classification"
        ],
        limitations=[
            "Not suitable for sarcasm detection",
            "Limited to English language",
            "May struggle with domain-specific terminology"
        ],
        associated_costs={
            "per_query_cost": 0.001,  # ADA
            "compute_requirements": "1 CPU core, 2GB RAM"
        },
        token_limits={
            "max_input_tokens": 512,
            "max_output_tokens": 10
        }
    )
    
    print(f"✓ Model card created")
    print(f"  Model ID: {model_card.model_id}")
    print(f"  Hash: {model_card.card_hash}")
    
    # Step 2: Validate the model card
    print("\n2. Validating model card...")
    validation_result = manager.validate_model_card(model_card)
    
    if validation_result.is_valid:
        print("✓ Model card is VALID")
    else:
        print("✗ Model card is INVALID")
        for error in validation_result.errors:
            print(f"  Error: {error}")
    
    if validation_result.warnings:
        print("  Warnings:")
        for warning in validation_result.warnings:
            print(f"    - {warning}")
    
    # Step 3: Certify the model
    print("\n3. Certifying model...")
    certification_result = manager.certify_model(
        model_card=model_card,
        mint_nft=True  # Mint NFT on blockchain
    )
    
    if certification_result["success"]:
        print("✓ Model certified successfully!")
        print(f"  Model ID: {certification_result['model_id']}")
        print(f"  NFT ID: {certification_result['nft_id']}")
        print(f"  Certified: {certification_result['certified']}")
    else:
        print("✗ Certification failed")
        for error in certification_result.get("errors", []):
            print(f"  Error: {error}")
    
    # Step 4: Retrieve model card
    print("\n4. Retrieving certified model card...")
    retrieved_card = manager.load_model_card(model_card.model_id)
    
    if retrieved_card:
        print("✓ Model card retrieved")
        print(f"  Name: {retrieved_card.model_name}")
        print(f"  Certified: {retrieved_card.certified}")
        print(f"  Size: {retrieved_card.model_size:,} parameters")
    
    # Step 5: List all model cards
    print("\n5. Listing all model cards...")
    all_cards = manager.list_model_cards()
    print(f"Total model cards: {len(all_cards)}")
    
    print("\n" + "=" * 60)
    print("Certification example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

