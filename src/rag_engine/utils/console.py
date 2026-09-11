import os
import sys
import time
import threading
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..core.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EXTRACTED_IMAGES_DIR,
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME,
    VLM_BASE_URL,
    VLM_ENABLED,
    VLM_MODEL_NAME,
    VLM_PROVIDER,
    VLM_TIMEOUT,
)

console = Console()


def print_banner(
    source_path: str | Path | None = None,
    vlm_provider: str | None = None,
):
    """Print an aesthetic pipeline setup banner."""
    has_gemini = bool(GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY"))
    prov = (vlm_provider or VLM_PROVIDER).lower().strip()

    if prov in ("none", "off", "disabled", "false"):
        provider_info = "[bold red]Disabled (Text-Only)[/bold red]"
    elif prov == "gemini" or (prov == "auto" and has_gemini):
        provider_info = f"[bold green]Gemini API[/bold green] ({GEMINI_MODEL_NAME})"
    elif prov == "lmstudio" or (prov == "auto" and not has_gemini):
        provider_info = f"[bold yellow]LM Studio[/bold yellow] ({VLM_MODEL_NAME} @ {VLM_BASE_URL})"
    else:
        provider_info = f"[bold yellow]Local LM Studio[/bold yellow] ({VLM_MODEL_NAME})"

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_row("[bold cyan]Source Path[/bold cyan]", f": {source_path or 'Default Data Dir'}")
    table.add_row("[bold cyan]Embed Model[/bold cyan]", f": {EMBEDDING_MODEL}")
    table.add_row("[bold cyan]VLM Provider[/bold cyan]", f": {provider_info}")
    table.add_row("[bold cyan]VLM Timeout[/bold cyan]", f": {int(VLM_TIMEOUT)}s")
    table.add_row("[bold cyan]Chroma DB[/bold cyan]", f": {CHROMA_DIR} (Collection: {COLLECTION_NAME})")
    table.add_row("[bold cyan]Image Storage[/bold cyan]", f": {EXTRACTED_IMAGES_DIR}")

    console.print()
    console.print(
        Panel(
            table,
            title="[bold magenta]⚡ SIH Multimodal RAG Engine[/bold magenta]",
            border_style="magenta",
            expand=False,
        )
    )
    console.print()


def print_step(step_num: int, total_steps: int, description: str):
    """Print step section divider."""
    console.print(
        f"\n[bold blue]━━━ Step [{step_num}/{total_steps}] {description} ━━━[/bold blue]"
    )


def print_info(msg: str):
    console.print(f"[bold blue][ℹ][/bold blue] {msg}")


def print_success(msg: str):
    console.print(f"[bold green][✓][/bold green] {msg}")


def print_warning(msg: str):
    console.print(f"[bold yellow][⚠][/bold yellow] {msg}")


def print_error(msg: str):
    console.print(f"[bold red][❌][/bold red] {msg}")


class VLMTimerStatus:
    """Context manager for live updating seconds count-up display during VLM API calls."""

    def __init__(
        self,
        image_name: str,
        provider_name: str = "VLM",
        timeout: float = VLM_TIMEOUT,
    ):
        self.image_name = image_name
        self.provider_name = provider_name
        self.timeout = int(timeout)
        self.start_time = 0.0
        self.elapsed = 0.0
        self.running = False
        self.thread = None
        self.status = None

    def _timer_loop(self):
        while self.running:
            self.elapsed = time.time() - self.start_time
            secs_str = f"{int(self.elapsed):02d}s"
            if self.status:
                if self.elapsed >= self.timeout:
                    self.status.update(
                        f"[bold red][⌛ EXCEEDED TIMEOUT ({secs_str} / {self.timeout}s)][/bold red] "
                        f"Cancelling request for [cyan]{self.image_name}[/cyan]..."
                    )
                else:
                    self.status.update(
                        f"[bold yellow][⏳ {secs_str} / {self.timeout}s][/bold yellow] "
                        f"Captioning [cyan]{self.image_name}[/cyan] via {self.provider_name}..."
                    )
            time.sleep(0.5)

    def __enter__(self):
        self.start_time = time.time()
        self.running = True
        self.status = console.status(
            f"[bold yellow][⏳ 00s / {self.timeout}s][/bold yellow] Captioning [cyan]{self.image_name}[/cyan] via {self.provider_name}...",
            spinner="dots",
        )
        self.status.start()
        self.thread = threading.Thread(target=self._timer_loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.status:
            self.status.stop()

        total_time = time.time() - self.start_time
        if exc_type is None:
            console.print(
                f"[bold green][✓ {total_time:.1f}s][/bold green] Captioning complete for [cyan]{self.image_name}[/cyan]"
            )
        else:
            console.print(
                f"[bold red][❌ {total_time:.1f}s][/bold red] Captioning failed/timed out for [cyan]{self.image_name}[/cyan]"
            )
        return False
