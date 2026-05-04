"""CLI entry point for the RAG pipeline."""

import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.pipeline import (
    run_full_pipeline,
    run_convert_only,
    run_ingest_only,
    run_query_only,
)

console = Console()


def print_help():
    """Print usage information."""
    table = Table(title="Available Commands", show_header=True, header_style="bold cyan")
    table.add_column("Command", style="bold green")
    table.add_column("Description")

    table.add_row("convert", "Convert documents to Markdown using MarkItDown")
    table.add_row("ingest", "Chunk and ingest documents into ChromaDB")
    table.add_row("query", "Start interactive query mode")
    table.add_row("run", "Run full pipeline: convert → ingest → query")
    table.add_row("help", "Show this help message")

    console.print(Panel.fit(table, border_style="cyan"))


def main():
    """CLI entry point with subcommands."""
    if len(sys.argv) < 2:
        console.print("[yellow]No command provided. Showing help...[/yellow]\n")
        print_help()
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "convert":
        run_convert_only()
    elif command == "ingest":
        run_ingest_only()
    elif command == "query":
        run_query_only()
    elif command == "run":
        run_full_pipeline()
    elif command in {"help", "-h", "--help"}:
        print_help()
    else:
        console.print(f"[red]Unknown command: '{command}'[/red]\n")
        print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
