"""
CLI for benchmarking
"""
import click
from ai_standards.benchmarking import BenchmarkHarness
from ai_standards.benchmarking.tasks import register_default_tasks


@click.group()
def main():
    """AI Model Benchmarking CLI"""
    pass


@main.command()
def list_tasks():
    """List available benchmark tasks"""
    register_default_tasks()
    harness = BenchmarkHarness()
    tasks = harness.task_registry.list_tasks()
    
    if not tasks:
        click.echo("No benchmark tasks registered")
        return
    
    click.echo(f"Available benchmark tasks ({len(tasks)}):")
    for task_name in tasks:
        task = harness.task_registry.get(task_name)
        click.echo(f"  • {task_name} ({task.task_type})")
        click.echo(f"    {task.description}")


@main.command()
@click.argument('model_id')
@click.option('--task', help='Specific task name (if not specified, shows all results)')
def show_results(model_id, task):
    """Show benchmark results for a model"""
    harness = BenchmarkHarness()
    results = harness.load_results(model_id)
    
    if not results.results:
        click.echo(f"No benchmark results found for model: {model_id}")
        return
    
    click.echo(f"Benchmark Results: {results.model_name}")
    click.echo(f"  Model ID: {results.model_id}")
    click.echo(f"  Overall Score: {results.overall_score:.4f}")
    click.echo(f"\nTask Results:")
    
    for result in results.results:
        if task and result.task_name != task:
            continue
        
        click.echo(f"\n  {result.task_name} ({result.task_type})")
        click.echo(f"    Score: {result.benchmark_score:.4f}")
        click.echo(f"    Samples: {result.num_samples}")
        click.echo(f"    Time: {result.execution_time:.2f}s")
        
        if result.metrics:
            click.echo(f"    Metrics:")
            for metric_name, value in result.metrics.items():
                click.echo(f"      - {metric_name}: {value:.4f}")


@main.command()
@click.option('--task', help='Filter by task name')
@click.option('--limit', default=10, help='Number of results to show')
def leaderboard(task, limit):
    """Show model leaderboard"""
    harness = BenchmarkHarness()
    rankings = harness.get_leaderboard(task_name=task)[:limit]
    
    if not rankings:
        click.echo("No benchmark results available")
        return
    
    title = f"Leaderboard" + (f" - {task}" if task else "")
    click.echo(title)
    click.echo("=" * len(title))
    
    for i, entry in enumerate(rankings, 1):
        click.echo(f"{i}. {entry['model_name']} ({entry['task_name']})")
        click.echo(f"   Score: {entry['score']:.4f}")


if __name__ == '__main__':
    main()
