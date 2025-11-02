# Gyrus

A Multilayer Perceptron from scratch built in Python for educational purposes.

This library is built with numpy as its only external dependency.

# Getting Started

You can set up the development environment in two ways: using a standard Python virtual environment or using Nix.

1. Standard Python Setup (using venv)

This is the recommended method for most users.

- On macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

- On Windows

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies and the library: This command installs the required dependencies and then installs gyrus in "editable" mode (-e), so your code changes are reflected immediately.

```bash
pip install -r requirements.txt
pip install -e .
```

2. Nix Setup

If you are a Nix user, you can enter a fully-configured development shell with a single command from the project root:

```bash
nix-shell
```

This will automatically provide Python, numpy, and make your local gyrus package importable.
