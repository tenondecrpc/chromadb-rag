"""Convert various document formats to Markdown using MarkItDown."""

import os
import re
from pathlib import Path

from markitdown import MarkItDown
from rich.console import Console

from src.config import CONVERTED_DIR, SUPPORTED_DOC_EXTENSIONS

console = Console()

# Initialize MarkItDown converter
md_converter = MarkItDown()


def sanitize_filename(name: str) -> str:
    """Create a valid filename from a string."""
    valid = re.sub(r"[^\w\s-]", "", name.lower())
    valid = re.sub(r"[-\s]+", "_", valid)
    return valid


def convert_file_to_markdown(file_path: Path) -> str | None:
    """Convert a single file to Markdown using MarkItDown."""
    try:
        result = md_converter.convert(str(file_path))
        return result.markdown if result else None
    except Exception as e:
        console.print(f"[red]Error converting {file_path}: {e}[/red]")
        return None


def convert_url_to_markdown(url: str) -> str | None:
    """Convert a URL to Markdown using MarkItDown."""
    try:
        result = md_converter.convert(url)
        return result.markdown if result else None
    except Exception as e:
        console.print(f"[red]Error converting URL {url}: {e}[/red]")
        return None


def get_supported_files(directory: Path) -> list[Path]:
    """Get all supported document files from a directory."""
    if not directory.exists():
        return []

    files = []
    for ext in SUPPORTED_DOC_EXTENSIONS:
        files.extend(directory.glob(f"*{ext}"))
    return sorted(files)


def convert_documents(
    source_dir: Path = None,
    urls: list[dict] = None,
    output_dir: Path = CONVERTED_DIR,
) -> list[Path]:
    """
    Convert documents to Markdown.

    Args:
        source_dir: Directory with documents to convert
        urls: List of {"url": "...", "name": "..."} dicts
        output_dir: Where to save converted .md files

    Returns:
        List of paths to converted Markdown files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    converted_files = []

    # Convert local files
    if source_dir and source_dir.exists():
        files = get_supported_files(source_dir)
        console.print(f"[cyan]Found {len(files)} files to convert in {source_dir}[/cyan]")

        for file_path in files:
            console.print(f"[yellow]Converting {file_path.name}...[/yellow]")
            markdown = convert_file_to_markdown(file_path)

            if markdown:
                output_name = sanitize_filename(file_path.stem) + ".md"
                output_path = output_dir / output_name
                output_path.write_text(markdown, encoding="utf-8")
                converted_files.append(output_path)
                console.print(f"[green]Saved: {output_path}[/green]")

    # Convert URLs
    if urls:
        for item in urls:
            url = item["url"]
            name = item.get("name", "document")
            console.print(f"[yellow]Converting URL: {name}...[/yellow]")
            markdown = convert_url_to_markdown(url)

            if markdown:
                output_name = sanitize_filename(name) + ".md"
                output_path = output_dir / output_name
                # Add source URL as metadata at the top
                content = f"<!-- source: {url} -->\n\n{markdown}"
                output_path.write_text(content, encoding="utf-8")
                converted_files.append(output_path)
                console.print(f"[green]Saved: {output_path}[/green]")

    console.print(f"[bold green]Total converted: {len(converted_files)} files[/bold green]")
    return converted_files


def get_markdown_files(directory: Path = CONVERTED_DIR) -> list[Path]:
    """Get all Markdown files from a directory."""
    if not directory.exists():
        return []
    return sorted(directory.glob("*.md"))
