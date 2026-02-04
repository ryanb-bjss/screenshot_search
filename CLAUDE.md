# Python setup
- Use `UV_NATIVE_TLS=1 uv sync` to install dependencies (native TLS needed for corporate proxy)
- Use `uv run <command>` to run commands in the virtual environment

# Code style
- Use functional programming where possible

# Workflow
- Prefer running single tests, and not the whole test suite, for performance