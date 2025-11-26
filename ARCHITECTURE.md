# GEAK-Eval Architecture

## Overview

GEAK-Eval is an **improved evaluation framework for Triton GPU kernels** that provides accurate correctness testing and performance measurement. It addresses critical issues in existing evaluation frameworks by using proper output comparison (`torch.allclose`), consistent seeding, and comprehensive performance profiling.

The framework supports multiple kernel benchmarks (TritonBench-G, ROCm) and provides both command-line and programmatic interfaces for evaluation.

---

## Core Components

### 1. Evaluators
Test kernel correctness through cascaded evaluation:
- **Call test**: Can the kernel execute without errors?
- **Execution test**: Does the output match ground truth?
- **Performance test**: What is the speedup vs. reference?

### 2. Metrics
Quantify evaluation results:
- **Accuracy**: Ratio of correct predictions
- **Pass@k**: Probability of k correct answers in n attempts

### 3. Processors
Handle LLM outputs and code extraction:
- Parse markdown code blocks
- Clean and format code for execution

### 4. Performance Evaluators
Measure kernel efficiency:
- Latency (speedup vs. golden reference)
- GPU efficiency (TFLOPS, GB/s)
- Hardware utilization metrics

---

## Architecture Flow

```
Input Code → Process → Call Test → Exec Test → Perf Test → Results
                ↓           ↓           ↓           ↓
              Clean      Import    AllClose    Benchmark
              Code       Module    Compare     Speedup
```

---

## Key Design Improvements

### Issues with Original TritonBench Eval Framework

1. **Inaccurate comparison**: Used stdout string comparison instead of numerical comparison
2. **Subprocess-only execution**: Both generated and ground truth run via subprocess
3. **Seed inconsistency**: Random seed not properly set across runs
4. **Missing print statements**: ~150 ground truth files lacked `print(result_gold)`
5. **Incomplete tests**: Some files missing `result_gold = test_*()` call
6. **Memory access faults**: 7 kernel files had bugs causing crashes

### GEAK-Eval Solutions

1. ✅ **`torch.allclose` comparison**: Proper numerical tolerance checking
2. ✅ **Module imports**: Direct Python import for accurate testing
3. ✅ **Consistent seeding**: Set random seed for reproducibility
4. ✅ **Fixed ground truth**: All files include proper test structure
5. ✅ **Memory fixes**: Corrected all faulty kernels
6. ✅ **Integrated performance**: Complete evaluation pipeline
7. ✅ **Cascaded evaluation**: Stop early if kernel fails call test

---

## Evaluation Pipeline

### Phase 1: Setup (One-Time)

**Purpose:** Record golden reference performance for target GPU

```bash
geak-eval setup -ds tbg    # TritonBench-G setup
geak-eval setup -ds rocm   # ROCm setup
```

**Actions:**
1. Run all ground truth kernels
2. Measure baseline performance
3. Save golden metrics to disk
4. Create performance database

**Output:**
- `{GPU}_golden_metrics/`: Performance baseline data
- `{GPU}_golden_results/`: Reference execution results

### Phase 2: Correctness Evaluation

**Input:** Generated code (from LLM or file)

#### Step 2.1: Code Processing

**Processor:** `LLMOutputProcessor`

```python
# Extract code from markdown blocks
code_blocks = extract_code_blocks(response)
code = combine_code_blocks(code_blocks)
```

**Handles:**
- Markdown code blocks (` ```python ... ``` `)
- Multiple code blocks in single response
- Raw Python code (no markers)

#### Step 2.2: Test Preparation

**Evaluator:** `TestAllCloseEvaluatorTBG` or `TestAllCloseEvaluatorROCm`

```python
# 1. Get ground truth file
triton_file = get_ground_truth_fpath(fname)

# 2. Extract test code (after separator line)
test_code = get_tests_code(triton_file)

# 3. Combine generated code + test code
gen_file = generated_code + test_separator + test_code

# 4. Write to temporary file
write_file(gen_file, tmp_dir)
```

**File Structure:**
```python
# Generated kernel code
import torch
import triton
import triton.language as tl

@triton.jit
def my_kernel(...):
    # kernel implementation
    pass

def wrapper_function(...):
    # launch kernel
    pass

##############################################################################
# Test code (below separator)

def test_correctness():
    # test implementation
    result = wrapper_function(...)
    return result

result_gold = test_correctness()
```

#### Step 2.3: Call Test

**Goal:** Check if code can execute without errors

```python
call_status, stdout, stderr = _call_file(gen_file, timeout=120)
```

**Tests:**
- Syntax errors
- Import errors
- Runtime exceptions
- Compilation failures

**Outcome:**
- `call_status=True`: Code runs without errors → proceed to execution test
- `call_status=False`: Return immediately with error message

#### Step 2.4: Execution Test

**Goal:** Verify output matches ground truth

**Module:** `TB_correctness.py` (TritonBench) or `ROCm_correctness.py` (ROCm)

```python
# 1. Set consistent seed
set_seed(42)

# 2. Import both files as modules
gen_result = import_variable_from_file(gen_file, "result_gold")
ref_result = import_variable_from_file(ref_file, "result_gold")

# 3. Compare using torch.allclose or np.allclose
exec_status = compare(ref_result, gen_result, atol=1e-3, rtol=1e-1)
```

**Comparison Function:**
```python
def compare(ref, gen, fname, atol, rtol):
    if type(gen) == torch.Tensor:
        return torch.allclose(ref, gen, atol=atol, rtol=rtol)
    elif type(gen) == np.ndarray:
        return np.allclose(ref, gen, atol=atol, rtol=rtol)
    elif type(gen) in [list, tuple, dict]:
        # Recursive comparison for nested structures
        return all(compare(r, g, fname, atol, rtol) 
                   for r, g in zip(ref, gen))
    else:
        return ref == gen
```

**Tolerance Settings:**
- Default: `atol=1e-3, rtol=1e-1`
- Configurable per evaluation
- Handles different data types (fp16, fp32, int)

**Outcome:**
- `exec_status=True`: Output matches → save to exec_root and proceed to performance test
- `exec_status=False`: Return with mismatch details

### Phase 3: Performance Evaluation

**Goal:** Measure kernel performance and efficiency

**Trigger:** Only runs if `exec_status=True`

#### Step 3.1: Write Performance Test Files

**Module:** `perf/run_bench/write_file.py`

```python
# Extract kernel configurations from passed codes
for code_file in exec_root:
    # Parse autotuning configs
    configs = extract_autotune_configs(code_file)
    
    # Generate benchmark script
    benchmark_script = create_benchmark(code_file, configs)
    
    # Write to perf folder
    write_file(benchmark_script, gen_perf_folder)
```

**Generated benchmark includes:**
- Warmup iterations
- Timing loops
- Memory bandwidth calculations
- TFLOPS computations

#### Step 3.2: Multi-Process GPU Execution

**Module:** `perf/run_bench/multiprocess_gpu_run.py`

```python
# Run all benchmarks in parallel
with multiprocessing.Pool() as pool:
    results = pool.map(run_benchmark, benchmark_files)
```

**Measures:**
- Latency (ms): Time per kernel invocation
- Throughput: Operations per second
- Memory bandwidth: GB/s
- GPU utilization

#### Step 3.3: Efficiency Analysis

**Module:** `perf/2_efficiency.py`

```python
# Compare against golden reference
speedup = golden_latency / generated_latency

# Calculate efficiency
efficiency = (operations / latency) / theoretical_peak

# Aggregate results
results = {
    "filename": fname,
    "speedup": speedup,
    "latency_ms": latency,
    "efficiency_tflops": efficiency_tflops,
    "efficiency_gbps": efficiency_gbps,
    "memory_bandwidth": memory_bandwidth
}
```

**Output:**
- `efficiency.json`: Performance metrics per kernel
- `performance_analysis.txt`: Human-readable summary

---

## Evaluation Modes

### Mode 1: JSON-Based Evaluation (Default)

**Input Format:**
```json
[
  {
    "predict": "LLM generated code...",
    "label": "Golden reference code...",
    "file": "kernel_name.py",
    "difficulty": 3
  },
  ...
]
```

**Usage:**
```bash
geak-eval -f predictions.json -o results -ds tbg
```

**Workflow:**
1. Load JSON file
2. Extract `predict` field
3. Process LLM output (extract code blocks)
4. Run evaluation pipeline
5. Aggregate metrics

### Mode 2: Direct Code Evaluation

**Input Format:** Raw Python files (`.py`)

**Usage:**
```bash
geak-eval -f generated_kernels/ -o results -ds rocm -c
```

**Workflow:**
1. Read Python files directly
2. Skip LLM output processing
3. Run evaluation pipeline
4. No difficulty scoring

### Mode 3: Custom Test Path

**Purpose:** Use different test suite (e.g., autotuning variants)

**Usage:**
```bash
geak-eval -f kernels/ -o results -ds rocm -tp custom_tests/
```

**Workflow:**
1. Load generated code
2. Override test code from custom path
3. Run evaluation with custom tests

---

## Pass@k Metric

**Definition:** Probability that at least one of k code samples is correct

**Formula:**
```
Pass@k = 1 - Π(1 - k / (n - c + i)) for i in [1, k]

where:
- n: total number of samples
- c: number of correct samples
- k: number of samples to consider
```

**Usage:**
```bash
geak-eval -f results/ -o output -ds tbg -k 1,5,10
```

**Example:**
```
n=10 samples, c=3 correct

Pass@1 = 3/10 = 0.30 (30%)
Pass@5 = 1 - (7/10 × 6/9 × 5/8 × 4/7 × 3/6) = 0.79 (79%)
Pass@10 = 1.0 (100%, all samples considered)
```

**Interpretation:**
- Pass@1: Accuracy (first attempt success rate)
- Pass@k: Probability of success with k tries
- Higher k → higher pass rate (more chances)

---

## Class Hierarchy

### Evaluators

```
BaseEvaluator
  ├── TestAllCloseEvaluatorTBG (TritonBench-G)
  │     └── Methods:
  │           - get_ground_truth_fpath()
  │           - get_tests_code()
  │           - format_gen_code()
  │           - _call_file()
  │           - _check_match()
  │           - execute()
  └── TestAllCloseEvaluatorROCm (ROCm kernels)
        └── Methods: (inherits + overrides)
              - get_ground_truth_fpath()
              - get_tests_code()
              - _check_match()
              - execute()
```

**Key Differences (TBG vs ROCm):**
- **Test separator**: TBG uses line-based, ROCm uses block-based
- **Timeout**: TBG 2min, ROCm 40min (complex kernels)
- **Custom tests**: ROCm supports autotuning test variants
- **Error extraction**: ROCm includes pytest error parsing

### Metrics

```
BaseMetric
  ├── Accuracy
  │     └── compute(y_pred) → mean(y_pred)
  └── PassK
        └── compute(n, c, k) → pass@k probability
```

### Processors

```
BaseProcessor
  └── LLMOutputProcessor
        └── process(response) → extracted_code
```

### Performance Evaluators

```
BasePerfEval
  ├── PerformanceEvalTBG
  │     └── Methods:
  │           - evaluate(exec_folder)
  │           - parse(perf_data_path)
  └── PerformanceEvalROCm
        └── Methods: (similar to TBG)
              - evaluate(exec_folder)
              - parse(perf_data_path)
```

---

## Directory Structure

```
geak_eval/
├── __init__.py              # Package exports
├── run.py                   # Main CLI entry point
├── constants.py             # Paths and configuration
├── initializations.py       # Setup functions
│
├── evaluators/              # Correctness testing
│   ├── base.py             # BaseEvaluator
│   ├── interface.py        # TBG/ROCm evaluators
│   ├── TB_correctness.py   # TritonBench comparison
│   └── ROCm_correctness.py # ROCm comparison
│
├── metrics/                 # Evaluation metrics
│   ├── base.py
│   ├── accuracy.py         # Accuracy computation
│   └── passk.py            # Pass@k computation
│
├── processors/              # Code processing
│   ├── base.py
│   └── llm.py              # LLM output extraction
│
├── perf/                    # Performance evaluation
│   ├── base.py
│   ├── efficiency.py       # Main perf evaluator
│   ├── 2_efficiency.py     # Efficiency analysis
│   ├── performance_utils.py
│   └── run_bench/          # Benchmark execution
│       ├── write_file.py
│       └── multiprocess_gpu_run.py
│
├── helpers/                 # Utility functions
│   ├── generators.py       # Temp file generation
│   ├── helper.py           # Shell execution, error extraction
│   └── time.py             # Timestamp utilities
│
└── data/                    # Benchmark datasets
    ├── TritonBench/
    │   ├── data/TritonBench_G_v1/          # Ground truth kernels
    │   └── performance_metrics/perf_G/     # Golden metrics
    └── ROCm/
        ├── data/ROCm_v1/                   # Ground truth kernels
        ├── data/ROCm_v1_autotune/          # Autotuning variants
        └── data/performance/               # Performance data
```

---

## Configuration and Constants

### Key Constants (`constants.py`)

```python
# Dataset roots
TBG_DATA_ROOT = "geak_eval/data/TritonBench/data/TritonBench_G_v1"
ROCm_DATA_ROOT = "geak_eval/data/ROCm/data/ROCm_v1"
ROCm_DATA_AUTOTUNE_ROOT = "geak_eval/data/ROCm/data/ROCm_v1_autotune"

# Performance golden data
TBG_PERF_GOLD_DATA_ROOT = f".../{GPU}_golden_results"
ROCM_PERF_GOLD_DATA_ROOT = ".../performance/golden_results"

# File naming conventions
Names.GEN_SUFFIX = "_gen_triton_code"
Names.REF_SUFFIX = "_ref_triton_code"
Names.RET_SEPERATOR = "*#*#"

# GPU detection
Names.GPU = torch.cuda.get_device_name(0).replace(" ", "_")
```

### CLI Arguments

**Main Evaluation:**
```
--folder_or_file, -f   : Input folder/file path
--outfile, -o          : Output results file
--dataset, -ds         : Dataset type [tbg, rocm]
--file_pat, -p         : File pattern (glob)
--k_vals, -k           : Pass@k values (comma-separated)
--run_on_code, -c      : Direct code evaluation mode
--custom_tests_path, -tp : Custom test suite path
--debug, -d            : Debug mode (limit iterations)
```

**Setup:**
```
--dataset, -ds         : Dataset to setup [all, tbg, rocm]
```

---

## Usage Examples

### Example 1: Evaluate JSON Predictions

```bash
# Setup (first time only)
geak-eval setup -ds tbg

# Evaluate predictions
geak-eval -f llm_predictions.json -o results -ds tbg
```

**Output:**
```
results/
├── tmp/                          # Temporary execution files
├── exec/                         # Passed kernels
├── results.txt                   # Detailed logs
├── results_results_0.json        # Correctness results
├── results_perf_0.json          # Performance results
└── results_all_passes.json      # Aggregated results
```

### Example 2: Multi-Pass Evaluation with Pass@k

```bash
# Generate 10 code samples per problem
for i in {0..9}; do
    python generate.py > predictions_$i.json
done

# Evaluate all passes
for i in {0..9}; do
    geak-eval -f predictions_$i.json -o results -ds tbg
done

# Compute Pass@k
geak-eval -f results/ -o final -ds tbg -k 1,5,10
```

**Output:**
```
Call Accuracy for pass@1: 45.2%
Exec Accuracy for pass@1: 38.7%

Call Accuracy for pass@5: 82.3%
Exec Accuracy for pass@5: 71.5%

Call Accuracy for pass@10: 94.1%
Exec Accuracy for pass@10: 88.9%
```

### Example 3: Direct Code Evaluation

```bash
# Evaluate raw Python files
geak-eval -f generated_kernels/ -o results -ds rocm -c

# With custom tests (autotuning)
geak-eval -f kernels/ -o results -ds rocm -c \
    -tp geak-eval/data/ROCm/data/ROCm_v1_autotune
```

### Example 4: Programmatic Usage

```python
from geak_eval.evaluators.interface import get_evaluators
from geak_eval.processors.llm import LLMOutputProcessor

# Initialize
evaluator = get_evaluators["tbg"]()
processor = LLMOutputProcessor()

# Process LLM output
llm_response = "```python\nimport triton\n...\n```"
code = processor(llm_response)

# Run evaluation
call_status, exec_status, stdout, stderr = evaluator(
    code=code,
    log_root="./tmp",
    exec_root="./exec",
    file_name="softmax.py",
    atol=1e-3,
    rtol=1e-1
)

print(f"Call: {call_status}, Exec: {exec_status}")
print(f"Speedup: {extract_speedup(stdout)}")
```

---

## Detailed Component Descriptions

### 1. Test Separator Logic

**Purpose:** Split ground truth file into kernel code and test code

**TritonBench (Line-based):**
```python
# Ground truth file structure
# ... kernel code ...
##############################################################################
# ... test code ...

# Extraction
with open(file, 'r') as f:
    lines = f.readlines()
    sep_idx = lines.index("#" * 146)
    test_code = lines[sep_idx+1:]
```

**ROCm (Block-based):**
```python
# Ground truth file structure
# ... kernel code ...
##############################################################################
# ... test code ...

# Extraction
with open(file, 'r') as f:
    content = f.read()
    snippets = content.split("#" * 146)
    test_code = snippets[1]
```

### 2. Seeding for Reproducibility

**Applied at import time:**
```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
```

**Ensures:**
- Consistent random tensor generation
- Repeatable test inputs
- Deterministic GPU operations
- Same execution path across runs

### 3. Error Extraction

**Purpose:** Extract meaningful error messages from pytest output

```python
def extract_errors(stderr_string):
    # 1. Check for collection errors (syntax, import)
    collection_error = extract_collection_error(stderr_string)
    if collection_error:
        return collection_error
    
    # 2. Extract first test failure
    first_failure = extract_first_pytest_failure(stderr_string)
    return first_failure
```

**Patterns:**
- `ERRORS` block for collection failures
- `test_*` failure blocks for runtime errors
- Formatted for readability

### 4. Module Import Mechanism

**Purpose:** Execute code and retrieve result variable

```python
def import_variable_from_file(file_path, variable_name="result_gold"):
    set_seed(42)  # Consistent randomness
    
    # Create module spec
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    
    # Execute module
    spec.loader.exec_module(module)
    
    # Retrieve variable
    return getattr(module, variable_name, None)
```

**Advantages:**
- Direct Python execution (no subprocess overhead)
- Access to return values
- Proper exception handling
- Debugging capabilities

### 5. Performance Benchmark Generation

**Purpose:** Create standardized benchmark scripts

```python
def create_benchmark(kernel_code, configs):
    benchmark = f"""
import torch
import triton

{kernel_code}

# Warmup
for _ in range(10):
    result = kernel_wrapper(...)

# Benchmark
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    result = kernel_wrapper(...)
torch.cuda.synchronize()
end = time.time()

latency = (end - start) / 100 * 1000  # ms
print(f"Latency: {{latency:.3f}} ms")
"""
    return benchmark
```

---

## Performance Metrics Explained

### Speedup

**Definition:** Ratio of golden reference latency to generated kernel latency

```
Speedup = Golden_Latency / Generated_Latency
```

**Interpretation:**
- `Speedup > 1.0`: Generated kernel is faster (good!)
- `Speedup = 1.0`: Same performance as reference
- `Speedup < 1.0`: Generated kernel is slower

**Example:**
```
Golden latency: 2.5 ms
Generated latency: 1.0 ms
Speedup: 2.5 / 1.0 = 2.5x (150% faster)
```

### GPU Efficiency (TFLOPS)

**Definition:** Percentage of theoretical peak TFLOPS achieved

```
Efficiency = (Actual_TFLOPS / Peak_TFLOPS) × 100%

where:
Actual_TFLOPS = Operations / (Latency_sec × 10^12)
```

**Example:**
```
Matrix multiplication: M=N=K=4096
Operations: 2 × M × N × K = 137 billion FLOPs
Latency: 10 ms = 0.01 sec
Actual: 137e9 / 0.01 = 13.7 TFLOPS

Peak (MI250): ~47.9 TFLOPS (FP32)
Efficiency: 13.7 / 47.9 = 28.6%
```

### Memory Bandwidth (GB/s)

**Definition:** Data transfer rate between global memory and compute units

```
Bandwidth = (Bytes_Read + Bytes_Written) / Latency_sec / 10^9
```

**Example:**
```
Element-wise operation: N=1M elements, dtype=float32
Bytes: 1M × 4 (read) + 1M × 4 (write) = 8 MB
Latency: 0.1 ms = 0.0001 sec
Bandwidth: 8e6 / 0.0001 / 1e9 = 80 GB/s

Peak (MI250): ~1638 GB/s
Efficiency: 80 / 1638 = 4.9%
```

---

## Error Handling and Edge Cases

### Timeout Handling

```python
# Different timeouts for different stages
CALL_TIMEOUT = 2 * 60      # 2 minutes for call test
EXEC_TIMEOUT = 2 * 60      # 2 minutes for exec test (TBG)
EXEC_TIMEOUT_ROCM = 40 * 60 # 40 minutes for exec test (ROCm)
PERF_TIMEOUT = None        # No timeout for performance (varies)
```

### Empty Code Handling

```python
if code is None or code.strip() == "":
    return False, False, "", "Code is empty"
```

### Type Mismatch Handling

```python
if type(gen_result) != type(ref_result):
    return False, False, None, f"Type mismatch: gen={type(gen_result)}, ref={type(ref_result)}"
```

### Import Failure Handling

```python
try:
    gen_result = import_variable_from_file(gen_file, "result_gold")
except Exception as e:
    return False, False, None, f"Import failed: {str(e)}"
```

---

## Integration with GEAK-Agent

### Agent → Eval Workflow

```python
# In GEAK-Agent (agents/GaAgent.py)
from dataloaders.TritonBench import TritonBench

dataset = TritonBench(...)

# Evaluate generated code
pass_call, pass_exe, speedup, stdout, stderr = \
    dataset.test_opt_correctness(
        code=generated_code,
        filename=mem.ps.filename,
        tmp_dir=tmp_dir,
        exe_dir=exe_dir,
        gpu_id=gpu_id
    )

# Internally calls GEAK-Eval
evaluator = get_evaluators['tbg']()
call_status, exec_status, stdout, stderr = evaluator(
    code, tmp_dir, exe_dir, filename, atol=1e-3, rtol=1e-3
)
```

### Data Flow

```
GEAK-Agent generates code
    ↓
TritonBench.test_opt_correctness()
    ↓
GEAK-Eval evaluator()
    ↓
Returns: pass_call, pass_exe, speedup
    ↓
Agent uses results for:
  - Reflection generation
  - Performance candidate selection
  - Next iteration planning
```

---

## Best Practices

### 1. Tolerance Configuration

**Default values are conservative:**
```python
atol=1e-3  # Absolute tolerance
rtol=1e-1  # Relative tolerance (10%)
```

**Adjust based on kernel type:**
```python
# High-precision kernels (e.g., matrix multiplication)
atol=1e-4, rtol=1e-2

# Lower-precision kernels (e.g., approximate functions)
atol=1e-2, rtol=1e-1

# Mixed precision (FP16)
atol=1e-2, rtol=5e-2
```

### 2. Setup Before Evaluation

**Always run setup for each GPU:**
```bash
# When switching GPUs or first time
geak-eval setup -ds tbg
geak-eval setup -ds rocm
```

**Setup creates GPU-specific baselines:**
- Performance varies across GPU models
- Golden metrics tied to specific hardware
- Re-run after driver updates

### 3. Debugging Failed Evaluations

**Check temporary files:**
```bash
# Generated code
cat tmp/gen/kernel.py_gen_triton_code

# Call output
cat tmp/gen/kernel.py_gen_triton_code.stdout

# Error messages
cat tmp/gen/kernel.py_gen_triton_code.stderr
```

**Common issues:**
- Missing imports
- Function signature mismatch
- Incorrect tensor shapes
- Type incompatibilities

### 4. Performance Evaluation Tips

**Only evaluate correct kernels:**
```python
if exec_status:
    perf_data = perf_evaluator(exec_root)
```

**Check for anomalies:**
```python
if speedup < 0.1 or speedup > 100:
    # Likely measurement error or timeout
    investigate()
```

---

## Extension Guide

### Adding New Dataset

1. **Create ground truth folder:**
```
geak_eval/data/MyDataset/
├── data/
│   └── MyDataset_v1/
│       ├── kernel1.py
│       ├── kernel2.py
│       └── ...
└── performance/
    └── golden_results/
```

2. **Add constants:**
```python
# In constants.py
MyDataset_DATA_ROOT = os.path.join(REPO_ROOT, "data", "MyDataset", "data", "MyDataset_v1")
MyDataset_PERF_ROOT = os.path.join(REPO_ROOT, "data", "MyDataset", "performance", "golden_results")
```

3. **Create evaluator:**
```python
# In evaluators/interface.py
class TestAllCloseEvaluatorMyDataset(TestAllCloseEvaluatorTBG):
    def __init__(self):
        super().__init__(ground_truth_root=MyDataset_DATA_ROOT)
    
    # Override methods as needed
    def get_tests_code(self, fname):
        # Custom test extraction logic
        pass
```

4. **Register evaluator:**
```python
# In evaluators/interface.py
get_evaluators = {
    'tbg': TestAllCloseEvaluatorTBG,
    'rocm': TestAllCloseEvaluatorROCm,
    'mydataset': TestAllCloseEvaluatorMyDataset
}
```

5. **Add performance evaluator:**
```python
# In perf/efficiency.py
class PerformanceEvalMyDataset(BasePerfEval):
    ref_folder = MyDataset_PERF_ROOT
    # Implement evaluate() and parse()
```

### Adding New Metric

```python
# In metrics/my_metric.py
from .base import BaseMetric

class MyMetric(BaseMetric):
    def __init__(self):
        super().__init__(name="MyMetric")
    
    def compute(self, predictions, references):
        # Your metric logic
        return score
```

---

## Troubleshooting

### Issue: "No files found in folder"

**Cause:** Incorrect file pattern or path

**Solution:**
```bash
# Check actual files
ls -la your_folder/

# Adjust pattern
geak-eval -f your_folder -o results -ds tbg -p "kernel_*.json"
```

### Issue: "Import failed: No module named 'triton'"

**Cause:** Missing dependencies in test environment

**Solution:**
```bash
pip install -r requirements.txt
# or
pip install triton torch
```

### Issue: "Timeout expired"

**Cause:** Kernel takes too long to execute

**Solution:**
```python
# Increase timeout in evaluator
evaluator.execute(..., timeout=600)  # 10 minutes
```

### Issue: "Type mismatch: gen=list, ref=tuple"

**Cause:** Different return types

**Solution:**
```python
# In ground truth test
return tuple(results)  # Instead of list

# Or handle in comparison
def compare(ref, gen, ...):
    if type(ref) == tuple and type(gen) == list:
        gen = tuple(gen)
    ...
```

### Issue: Performance metrics missing

**Cause:** Setup not run or GPU changed

**Solution:**
```bash
# Re-run setup
geak-eval setup -ds tbg
```

---

## Performance Considerations

### Evaluation Speed

**Bottlenecks:**
1. **Module imports**: ~0.5-2s per kernel
2. **Kernel compilation**: ~1-5s per kernel
3. **Performance benchmarking**: ~10-60s per kernel

**Optimization strategies:**
- Parallel evaluation (use multiprocessing)
- Skip performance for failed kernels
- Cache compiled kernels
- Batch similar kernels

### Resource Usage

**Memory:**
- Peak: ~4-8 GB GPU memory per kernel
- Temporary files: ~100 MB per kernel
- Performance logs: ~10 MB per run

**Disk:**
- Clean temporary files regularly
- Archive old results
- Compress performance logs

---

## Key Differences: TritonBench vs ROCm

| Aspect | TritonBench-G (tbg) | ROCm |
|--------|---------------------|------|
| **Kernels** | 170+ Triton benchmarks | AMD-specific kernels |
| **Complexity** | Simple to moderate | Moderate to complex |
| **Timeout** | 2 minutes | 40 minutes |
| **Test format** | Line-based separator | Block-based separator |
| **Autotuning** | Standard configs | Advanced configs |
| **Custom tests** | Not supported | Supported (`-tp` flag) |
| **Error parsing** | Basic | Pytest-aware |

---

## Citation

If you use GEAK-Eval in your research, please cite:

```bibtex
@misc{wang2025geakintroducingtritonkernel,
    title={Geak: Introducing Triton Kernel AI Agent & Evaluation Benchmarks},
    author={Jianghui Wang and Vinay Joshi and Saptarshi Majumder and others},
    year={2025},
    eprint={2507.23194},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2507.23194}
}
```

---

## Contributing

Contributions welcome! See areas for improvement:

1. **New kernels**: Add more benchmark kernels
2. **Metrics**: Implement new evaluation metrics
3. **Performance**: Optimize evaluation speed
4. **Documentation**: Improve examples and guides
5. **Testing**: Add unit tests and CI/CD

**Process:**
1. Fork repository
2. Create feature branch
3. Implement changes
4. Add tests
5. Submit pull request

---

## License

Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

See LICENSE.txt in the repository root.

---

## Acknowledgments

GEAK-Eval builds upon:
- [TritonBench](https://github.com/thunlp/TritonBench) - Original benchmark suite
- [ROCm AITER](https://github.com/ROCm/aiter) - AMD AI tools
- [ROCm Triton](https://github.com/ROCm/triton) - AMD Triton fork

Special thanks to the open-source community for tools and feedback.
