# Usage

You can run benchmarks using `main.py` or the provided shell script.

## Configuration

First, create a `.env` file in the project root and set the following:

```bash
OPENAI_API_KEY=your_openai_api_key
MODEL_NAME=gpt-4o-mini
OPENAI_API_BASE=https://api.openai.com/v1
```

## Using `main.py`

### Basic Usage

```bash
# Run a math benchmark with a single agent
python main.py --benchmark math --agent-system single_agent --limit 5

# Run with supervisor-based multi-agent system
python main.py --benchmark math --agent-system supervisor_mas --limit 10

# Run with swarm-based multi-agent system
python main.py --benchmark math --agent-system swarm --limit 5
```

### Using the Shell Runner

A convenience script `run_benchmark.sh` is provided for quick runs.

```bash
# Syntax: ./run_benchmark.sh <benchmark_name> <agent_system> <limit>
./run_benchmark.sh math supervisor_mas 10
```
### Advanced Usage: Asynchronous Execution

For benchmarks that support concurrency, you can run them asynchronously to speed up evaluation.

```bash
# Run the humaneval benchmark with a concurrency of 10
python main.py --benchmark humaneval --async-run --concurrency 10
```
*Note: Benchmarks that do not support concurrency (e.g., `math`, `aime`) will automatically run in synchronous mode, even if `--async-run` is specified.*



## Command-Line Arguments

Here are some of the most common arguments for `main.py`:

| Argument              | Description                                                              | Default                  |
| --------------------- | ------------------------------------------------------------------------ | ------------------------ |
| `--benchmark`         | The name of the benchmark to run.                                        | `math`                   |
| `--agent-system`      | The agent system to use for the benchmark.                               | `single_agent`           |
| `--limit`             | The maximum number of problems to evaluate.                              | `10`                     |
| `--data`              | Path to a custom benchmark data file (JSONL format).                     | `data/{benchmark}_test.jsonl` |
| `--async-run`         | Run the benchmark asynchronously for faster evaluation.                  | `False`                  |
| `--concurrency`       | Set the concurrency level for asynchronous runs.                         | `10`                     |
| `--results-dir`       | Directory to store detailed JSON results.                                | `results/`               |
| `--metrics-dir`       | Directory to store performance and operational metrics.                  | `metrics/`               |
| `--use-tools`         | Enable the agent to use integrated tools (e.g., code interpreter).       | `False`                  |
| `--use-mcp-tools`     | Enable the agent to use tools via the Multi-Agent Communication Protocol. | `False`                  |
| `--mcp-config-file`   | Path to the MCP server configuration file. Required if using MCP tools.  | `None`                   |

## Example Output

After a run, a summary is printed to the console:

```bash
================================================================================
Benchmark Summary
================================================================================
Agent system: swarm
Accuracy: 70.00% (7/10)
Total duration: 335125ms
Results saved to: results/math_swarm_20250616_203434.json
Summary saved to: results/math_swarm_20250616_203434_summary.json

Run visualization:
$ python mas_arena/visualization/visualize_benchmark.py visualize \
  --summary results/math_swarm_20250616_203434_summary.json
```
