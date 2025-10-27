"""
CLI for certification management
"""
import click
import json
from ai_standards.certification import CertificationManager, ModelCard


@click.group()
def main():
    """AI Model Certification CLI"""
    pass


@main.command()
@click.option('--name', required=True, help='Model name')
@click.option('--size', required=True, type=int, help='Number of parameters')
@click.option('--architecture', required=True, help='Model architecture')
@click.option('--modality', required=True, help='Output modality (discrete/text/vision/audio)')
@click.option('--usage', required=True, help='Usage instructions')
@click.option('--license', required=True, help='Licensing terms')
@click.option('--output', default='model_card.json', help='Output file path')
def create(name, size, architecture, modality, usage, license, output):
    """Create a new model card"""
    manager = CertificationManager()
    
    model_card = manager.create_model_card(
        model_name=name,
        model_size=size,
        architecture=architecture,
        output_modality=modality,
        usage_instructions=usage,
        licensing_terms=license,
    )
    
    # Save to file
    with open(output, 'w') as f:
        json.dump(model_card.to_dict(), f, indent=2, default=str)
    
    click.echo(f"✓ Model card created: {output}")
    click.echo(f"  Model ID: {model_card.model_id}")
    click.echo(f"  Hash: {model_card.card_hash}")


@main.command()
@click.argument('model_card_file')
def validate(model_card_file):
    """Validate a model card"""
    with open(model_card_file, 'r') as f:
        data = json.load(f)
    
    model_card = ModelCard.from_dict(data)
    manager = CertificationManager()
    result = manager.validate_model_card(model_card)
    
    if result.is_valid:
        click.echo("✓ Model card is VALID")
    else:
        click.echo("✗ Model card is INVALID")
        click.echo("\nErrors:")
        for error in result.errors:
            click.echo(f"  - {error}")
    
    if result.warnings:
        click.echo("\nWarnings:")
        for warning in result.warnings:
            click.echo(f"  - {warning}")


@main.command()
@click.argument('model_card_file')
@click.option('--mint-nft', is_flag=True, help='Mint NFT on blockchain')
def certify(model_card_file, mint_nft):
    """Certify a model card"""
    with open(model_card_file, 'r') as f:
        data = json.load(f)
    
    model_card = ModelCard.from_dict(data)
    manager = CertificationManager()
    
    click.echo("Certifying model card...")
    result = manager.certify_model(model_card, mint_nft=mint_nft)
    
    if result["success"]:
        click.echo("✓ Model certified successfully!")
        click.echo(f"  Model ID: {result['model_id']}")
        if result.get('nft_id'):
            click.echo(f"  NFT ID: {result['nft_id']}")
    else:
        click.echo("✗ Certification failed")
        for error in result.get("errors", []):
            click.echo(f"  - {error}")


@main.command()
@click.argument('model_id')
def show(model_id):
    """Show model card details"""
    manager = CertificationManager()
    model_card = manager.load_model_card(model_id)
    
    if not model_card:
        click.echo(f"✗ Model card not found: {model_id}")
        return
    
    click.echo(f"Model Card: {model_card.model_name}")
    click.echo(f"  ID: {model_card.model_id}")
    click.echo(f"  Version: {model_card.version}")
    click.echo(f"  Architecture: {model_card.architecture}")
    click.echo(f"  Size: {model_card.model_size:,} parameters")
    click.echo(f"  Output: {model_card.output_modality}")
    click.echo(f"  Certified: {model_card.certified}")
    if model_card.nft_id:
        click.echo(f"  NFT ID: {model_card.nft_id}")


@main.command()
def list():
    """List all model cards"""
    manager = CertificationManager()
    cards = manager.list_model_cards()
    
    if not cards:
        click.echo("No model cards found")
        return
    
    click.echo(f"Found {len(cards)} model cards:")
    for card_id in cards:
        card = manager.load_model_card(card_id)
        if card:
            status = "✓" if card.certified else "✗"
            click.echo(f"  {status} {card.model_name} ({card_id})")


if __name__ == '__main__':
    main()
