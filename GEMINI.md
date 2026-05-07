# LoL RAG Python Project

This project implements a Hybrid Retrieval-Augmented Generation (RAG) system focused on League of Legends lore. It combines vector-based retrieval (using PostgreSQL/PGVector) and graph-based retrieval (using Neo4j) to provide accurate and context-aware answers about LoL champions, regions, and relationships.

## Project Structure

- `src/`: Core source code.
    - `core/`: Fundamental components like configuration, engine, and interfaces.
    - `ingestion/`: Scripts for processing and inserting data into databases.
    - `retrievers/`: Different retrieval strategies (Vector, Graph, BM25, Hybrid).
- `data/`: Raw HTML data files for champions.
- `processed_data/`: Cleaned and structured JSON data.
- `main.py`: A demonstration script for entity and relation extraction using GLiNER and REBEL.
- `src/chat.py`: Interactive CLI chat interface for the RAG system.
- `pyproject.toml`: Project metadata and dependencies (managed by `uv`).

## Technologies Used

- **LLM/Embeddings:** Ollama (default: `gemma3:1b` and `embeddinggemma:300m`).
- **Vector Database:** PostgreSQL with `pgvector`.
- **Graph Database:** Neo4j.
- **NLP Tools:** GLiNER (Entity Extraction), REBEL (Relation Extraction).
- **Frameworks:** LangChain, Pydantic, Python-dotenv.
- **Dependency Management:** `uv`.

## Setup and Installation

### Prerequisites

- Python >= 3.12
- [uv](https://github.com/astral-sh/uv) package manager.
- Running instances of PostgreSQL (with `pgvector`), Neo4j, and Ollama.
- Docker (optional, see `docker-compose.yaml`).

### Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   uv sync
   ```
3. Create a `.env` file in the root directory and configure your environment variables (refer to `src/core/config.py` for keys).

## Building and Running

### Data Ingestion

1. **Entity Extraction (GLiNER):**
   ```bash
   uv run python -m src.ingestion.extract_entities
   ```
2. **Vector Ingestion:**
   ```bash
   uv run python -m src.ingestion.insert_vector
   ```
3. **Graph Ingestion:**
   ```bash
   uv run python -m src.ingestion.insert_graph
   ```

### Running the Chat Interface

To start the interactive chat session:
```bash
uv run python -m src.chat
```

### Testing Entity Extraction

To test the GLiNER and REBEL extraction logic:
```bash
uv run python main.py
```

## Development Conventions

- **Modular Design:** New retrievers should implement the base interface defined in `src/core/interfaces.py`.
- **Configuration:** Use `src/core/config.py` for managing all environment-related settings.
- **Logging:** Use standard print statements or a logging library for progress and error reporting.
- **Data Integrity:** The `alias_mapping.json` file is crucial for normalizing champion names across the graph and vector stores.
