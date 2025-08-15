"""
CLI tool for Enterprise RAG System
"""
import click
import requests
import json
from pathlib import Path
from typing import List
import sys

API_URL = "http://localhost:8000"

@click.group()
def cli():
    """Enterprise RAG CLI"""
    pass

@cli.command()
@click.argument('files', nargs=-1, type=click.Path(exists=True))
@click.option('--index-name', default='default', help='Name for the index')
def index(files: tuple, index_name: str):
    """Index documents"""
    if not files:
        click.echo("No files specified")
        return
    
    file_paths = [str(Path(f).absolute()) for f in files]
    
    click.echo(f"Indexing {len(file_paths)} files...")
    
    response = requests.post(
        f"{API_URL}/index",
        json={"file_paths": file_paths, "index_name": index_name}
    )
    
    if response.status_code == 200:
        result = response.json()
        click.echo(f"✓ Indexed {result['documents_processed']} documents")
        click.echo(f"✓ Created {result['chunks_created']} chunks")
        click.echo(f"✓ Index ID: {result.get('index_id')}")
    else:
        click.echo(f"✗ Error: {response.text}")

@cli.command()
@click.argument('query')
@click.option('--session-id', default=None, help='Session ID')
@click.option('--k', default=20, help='Number of documents to retrieve')
def query(query: str, session_id: str, k: int):
    """Query the system"""
    click.echo(f"Querying: {query}")
    
    response = requests.post(
        f"{API_URL}/query",
        json={
            "query": query,
            "session_id": session_id,
            "retrieval_k": k,
            "use_reranker": True
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        click.echo("\n" + "="*50)
        click.echo("Answer:")
        click.echo(result['answer'])
        click.echo("\n" + "="*50)
        
        if result.get('source_documents'):
            click.echo("\nSources:")
            for i, doc in enumerate(result['source_documents'], 1):
                click.echo(f"\n{i}. {doc['text'][:200]}...")
    else:
        click.echo(f"✗ Error: {response.text}")

@cli.command()
def health():
    """Check system health"""
    response = requests.get(f"{API_URL}/health")
    
    if response.status_code == 200:
        result = response.json()
        click.echo(f"✓ System Status: {result['status']}")
        click.echo(f"✓ Ollama: {'Running' if result['ollama_running'] else 'Not Running'}")
        
        for component, status in result['components'].items():
            click.echo(f"✓ {component}: {status}")
    else:
        click.echo("✗ System is not responding")

if __name__ == "__main__":
    cli()
