# Benchmark Evaluation System

This directory contains tools for running benchmark evaluations against multiple code assistant tools (PotPie, Cursor, Codex, Claude Code, etc.) and comparing their performance.

## Overview

The benchmark system:
1. Reads questions and expected answers from a CSV file
2. Clones repositories and sets up worktrees for specific commits
3. Runs questions through each tool (PotPie API or CLI tools)
4. Evaluates responses using DeepEval's correctness metric
5. Saves results to CSV files and optionally uploads to Phoenix for visualization

## Prerequisites

### Required Dependencies

Install the required Python packages:

```bash
pip install deepeval pandas aiohttp python-dotenv arize-phoenix-client arize-phoenix
```

Or install from the project's `requirements.txt` which includes these dependencies.

**Note**: `arize-phoenix` is required for Phoenix visualization (used by default). If you only want CSV results, you can use the `--no-phoenix` flag instead.

### Environment Variables

For PotPie benchmarks, you need to set these environment variables (typically in a `.env` file):

```bash
defaultUsername=your_user_id
INTERNAL_ADMIN_SECRET=your_api_key
```

The PotPie API server should be running at `http://localhost:8001` (or modify `BASE_URL` in `run_benchmark.py`).

### CLI Tool Setup

For CLI tool benchmarks (Cursor, Codex, Claude Code, etc.), ensure the tools are installed and available in your system PATH:

- **Cursor**: Install with `curl https://cursor.com/install -fsS | bash`
- **Codex**: Install Codex CLI and ensure it's in your PATH
- **Claude Code**: Install Claude Code CLI and ensure it's in your PATH
- **Factory AI**: Install Factory AI CLI (Droids) and ensure it's in your PATH

The benchmark system will verify tool availability before running evaluations.

### Phoenix Setup

**Phoenix must be running before executing benchmarks** (unless using the `--no-phoenix` flag).

Phoenix is used for visualizing and comparing benchmark results. To set up Phoenix:

1. **Install Phoenix** (if not already installed):
   ```bash
   pip install arize-phoenix
   ```

2. **Start Phoenix server**:
   ```bash
   phoenix serve
   ```
   
   This will start Phoenix on `http://localhost:6006` by default. The Phoenix client will automatically connect to this server.

3. **Verify Phoenix is running**:
   - Open `http://localhost:6006` in your browser
   - You should see the Phoenix UI

**Note**: If you don't want to use Phoenix, you can run benchmarks with the `--no-phoenix` flag to skip uploads. Results will still be saved to CSV files.

## CSV File Format

Create a CSV file with the following columns:

- `repo_url`: Repository URL (e.g., `https://github.com/user/repo.git`)
- `commit_id`: Git commit SHA to evaluate against
- `question`: The question to ask the tool
- `expected_answer`: The expected/correct answer for evaluation

Example:

```csv
repo_url,commit_id,question,expected_answer
https://github.com/huggingface/trl.git,aaed6c1600ff4f3e0ccc6b3b8183c98d26390491,"How do I configure GRPO?",The GRPO configuration requires...
```

## Usage

### Basic Usage

Run benchmarks for a single tool:

```bash
python3 -m benchmark.run_benchmark --tools cursor
```

Run benchmarks for multiple tools concurrently:

```bash
python3 -m benchmark.run_benchmark --tools potpie cursor codex
```

### Available Tools

- `potpie`: PotPie API (requires running PotPie server)
- `cursor`: Cursor CLI tool
- `codex`: Codex CLI tool
- `claude_code`: Claude Code CLI tool
- `factory_ai`: Factory AI CLI tool (Droids)

### Command Line Options

```bash
python3 -m benchmark.run_benchmark [OPTIONS]

Options:
  --tools TOOLS [TOOLS ...]  Tool names to run benchmarks for (required)
  --csv CSV_FILE              Path to CSV file (default: benchmark_sub.csv)
  --no-phoenix                Don't upload results to Phoenix
```

### Examples

Run Cursor benchmark with default CSV:

```bash
python3 -m benchmark.run_benchmark --tools cursor
```

Run PotPie and Cursor with custom CSV:

```bash
python3 -m benchmark.run_benchmark --tools potpie cursor --csv my_benchmark.csv
```

Run without uploading to Phoenix:

```bash
python3 -m benchmark.run_benchmark --tools cursor --no-phoenix
```

## How It Works

### 1. Repository Setup

The system:
- Reads the CSV file to extract unique repository URLs and commit IDs
- Clones bare repositories to `repos/` directory
- Creates git worktrees for each commit to evaluate
- Reuses existing repositories/worktrees when possible

### 2. Question Processing

For **PotPie**:
- Parses each repository/commit through the PotPie API
- Waits for parsing to complete
- Sends questions via the PotPie conversation API

For **CLI Tools**:
- Executes the CLI tool command in the appropriate worktree directory
- Captures the output as the answer

### 3. Evaluation

- Uses DeepEval's `GEval` metric with a custom correctness criteria
- Compares generated answers against expected answers
- Generates scores (0.0-1.0), success flags, and reasoning

### 4. Results

Results are saved to:
- **CSV files**: `benchmark_results/{tool_name}_benchmark_results_{timestamp}.csv`
  - Contains: repo_url, commit_id, question, expected_answer, generated_answer, eval_score, eval_success, eval_reason, etc.
- **Phoenix** (optional): Uploads datasets and experiments for visualization and comparison

## Output Files

Results are saved in the `benchmark_results/` directory with the format:

```
{tool_name}_benchmark_results_{timestamp}.csv
```

Each CSV file contains:
- Original question and expected answer
- Generated answer from the tool
- Evaluation metrics (score, success, reason)
- Timestamp and metadata

## Phoenix Integration

By default, results are uploaded to Phoenix for visualization. Phoenix:
- Creates versioned datasets when content changes
- Tracks experiments over time
- Provides comparison views between tools

**Important**: Phoenix must be running before you execute benchmarks. Start Phoenix with:
```bash
phoenix serve
```

The Phoenix server should be accessible at `http://localhost:6006`. The benchmark system will automatically connect to it when uploading results.

To disable Phoenix uploads, use the `--no-phoenix` flag. This is useful if:
- Phoenix is not installed or running
- You only want CSV results
- You're running in an environment without Phoenix access

## Troubleshooting

### CLI Tool Not Found

If you get an error that a CLI tool is not found:
1. Verify the tool is installed: `which cursor-agent` (or equivalent)
2. Ensure it's in your system PATH
3. Check the tool's installation instructions

### PotPie Connection Errors

If PotPie benchmarks fail:
1. Verify the PotPie server is running at `http://localhost:8001`
2. Check your `.env` file has correct `defaultUsername` and `INTERNAL_ADMIN_SECRET`
3. Verify network connectivity to the server

### Repository Clone Failures

If repository cloning fails:
1. Check network connectivity
2. Verify repository URLs are correct and accessible
3. Ensure you have sufficient disk space in the `repos/` directory

### Phoenix Connection Errors

If you get errors about Phoenix connection:
1. Verify Phoenix is installed: `pip install arize-phoenix`
2. Start Phoenix server: `phoenix serve`
3. Check that Phoenix is accessible at `http://localhost:6006`
4. If you don't need Phoenix, use the `--no-phoenix` flag

### Evaluation Errors

If evaluation fails:
1. Check that DeepEval dependencies are installed
2. Verify your CSV file has the required columns
3. Check that expected answers are properly formatted

## File Structure

```
benchmark/
├── README.md              # This file
├── run_benchmark.py      # Main orchestrator script
├── potpie.py             # PotPie API integration
├── cli.py                 # CLI tool execution module
├── download.py            # Repository cloning and worktree setup
├── eval.py                # Evaluation metrics (DeepEval)
└── benchmark_sub.csv      # Example benchmark CSV file
```

## Advanced Usage

### Running PotPie Only

The original `potpie.py` script can be run standalone:

```bash
python3 -m benchmark.potpie
```

This runs PotPie benchmarks with hardcoded settings (uses `benchmark_sub.csv` and `http://localhost:8001`).

### Custom Evaluation Metrics

Modify `benchmark/eval.py` to customize the evaluation criteria. The current metric uses `GEval` with a technical correctness focus.

## Notes

- Repositories are cloned to `repos/` directory and reused across runs
- Worktrees are created for each commit and cleaned up/recreated as needed
- Multiple tools can be evaluated concurrently for faster benchmarking
- Results are timestamped to track performance over time

