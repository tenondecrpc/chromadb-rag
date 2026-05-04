"""Pipeline orchestrator for the RAG workflow."""

from rich.console import Console
from rich.panel import Panel

from src.markdown_converter import convert_documents
from src.ingest import ingest_documents
from src.ask import interactive_mode
from src.config import DOCS_PATH

console = Console()


def run_full_pipeline():
    """Run the complete pipeline: convert -> ingest -> query."""
    console.print(Panel.fit(
        "[bold cyan]RAG Pipeline[/bold cyan]\n"
        "[dim]Convert → Ingest → Query[/dim]",
        border_style="cyan",
    ))

    # Step 1: Convert
    console.print("\n[bold]=== STEP 1: Document Conversion ===[/bold]")
    convert_documents(source_dir=DOCS_PATH)

    # Step 2: Ingest
    console.print("\n[bold]=== STEP 2: Ingestion ===[/bold]")
    ingest_documents(convert_first=False)

    # Step 3: Query
    console.print("\n[bold]=== STEP 3: Interactive Query ===[/bold]")
    interactive_mode()


def run_convert_only():
    """Run only the document conversion step."""
    console.print("[bold]=== Document Conversion ===[/bold]")
    convert_documents(source_dir=DOCS_PATH)


def run_ingest_only():
    """Run only the ingestion step (assumes docs are already converted)."""
    console.print("[bold]=== Ingestion ===[/bold]")
    ingest_documents(convert_first=False)


def run_query_only():
    """Run only the query step."""
    console.print("[bold]=== Interactive Query ===[/bold]")
    interactive_mode()
