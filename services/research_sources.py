"""Public RSS/Atom sources used by the deterministic research service."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResearchSource:
    """Configuration for one public feed."""

    name: str
    url: str
    source_type: str
    category: str


# URLs are the publishers' public feed endpoints; unavailable feeds are skipped at runtime.
DEFAULT_RESEARCH_SOURCES: tuple[ResearchSource, ...] = (
    ResearchSource("OpenAI", "https://openai.com/blog/rss.xml", "rss", "Generative AI"),
    ResearchSource("Google AI", "https://blog.google/technology/ai/rss/", "rss", "General AI"),
    ResearchSource("NVIDIA AI", "https://blogs.nvidia.com/feed/", "rss", "AI Tools"),
    ResearchSource("Hugging Face", "https://huggingface.co/blog/feed.xml", "rss", "AI Tools"),
    ResearchSource("arXiv AI", "https://export.arxiv.org/rss/cs.AI", "rss", "AI Research"),
    ResearchSource("arXiv Machine Learning", "https://export.arxiv.org/rss/cs.LG", "rss", "Machine Learning"),
)