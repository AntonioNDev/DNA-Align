# GPU-Accelerated DNA Sequence Alignment Benchmark

**Author:** Antonio Nikoloski  
**Location:** North Macedonia  
**Project:** [PiDP-Project](https://github.com/AntonioNDev/PiDP-Project)

![Python](https://img.shields.io/badge/Python-3.x-blue)
![C++](https://img.shields.io/badge/C%2B%2B-CUDA-76B900)
![Domain](https://img.shields.io/badge/Domain-Bioinformatics-green)
![Focus](https://img.shields.io/badge/Focus-Parallel%20Computing-orange)

## 🧬 Overview

PiDP is a bioinformatics and performance-engineering project that investigates how different computing architectures perform when processing DNA sequence alignments.

The project implements the **Smith–Waterman local alignment algorithm** and compares three execution approaches:

- **Serial CPU:** alignment computation using a single process.
- **Multiprocessing CPU:** alignment computation distributed across multiple CPU processes.
- **CUDA GPU:** alignment computation parallelized across NVIDIA GPU threads.

Reference-derived DNA sequences are processed, synthetic mutations are introduced into query sequences, and the resulting reference-query pairs are aligned and benchmarked. The project also examines simulated mutation patterns, including transitions and transversions.

The primary objective is to compare execution time and validate score consistency across implementations under controlled workloads.

> **Scope:** PiDP is a DNA sequence alignment benchmark. It does not currently perform clinical cancer diagnosis, identify pathogenic variants, or establish whether a mutation is cancer-associated. The mutations used for benchmarking are simulated.

## 🚀 Key Features

- Smith–Waterman local sequence alignment.
- DNA encoding for efficient sequence processing.
- Synthetic mutation generation with configurable mutation rates.
- Transition-biased mutation simulation.
- Serial CPU, multiprocessing CPU, and CUDA GPU implementations.
- CPU/GPU alignment-score validation.
- Benchmarking across different numbers of sequence pairs.
- Mutation-rate and transition/transversion analysis.
- Performance measurements suitable for further visualization and analysis.

## 🏗️ Architecture and Workflow

The project follows a pipeline from reference-sequence processing to alignment and performance analysis.

1. **Reference loading:** Read DNA sequences from a FASTA reference file.
2. **Sequence encoding:** Convert A, C, G, and T bases into integer representations.
3. **Mutation simulation:** Generate query sequences with controlled substitutions.
4. **Alignment:** Calculate local-alignment scores for reference-query pairs.
5. **Parallel execution:** Execute the workload using serial CPU, multiprocessing CPU, and CUDA GPU approaches.
6. **Validation:** Compare alignment scores between implementations.
7. **Benchmark analysis:** Aggregate timing results and analyze simulated mutation patterns.

### Main components

| Component | Responsibility |
|---|---|
| `DataProcessing` | Reference-sequence loading, encoding, and data preparation |
| `BenchmarkData` | Benchmark configuration, synthetic sequence generation, and analysis |
| `PlotData` | Benchmark-result visualization |
| `sw_kernel.cu` | CUDA implementation of the Smith–Waterman alignment kernel |
| `main.py` | Application entry point, depending on the selected run configuration |

Exact responsibilities and available entry points depend on the current implementation in the repository.

## 🧠 The Smith–Waterman Algorithm

Smith–Waterman is a dynamic-programming algorithm for finding the highest-scoring local alignment between two sequences.

For sequence lengths \(m\) and \(n\), its conventional time complexity is \(O(mn)\). For equal-length sequences of length \(L\), the computational work scales quadratically with \(L\).

The current CUDA kernel uses the following scoring parameters:

| Event | Score |
|---|---:|
| Matching bases | +2 |
| Mismatching bases | −1 |
| Gap | −1 |

Where:

- \(H_{i,j}\) is the alignment score at position \((i,j)\).
- \(Q\) is the query sequence.
- \(R\) is the reference sequence.
- \(s(Q_i,R_j)\) is +2 for a match and −1 for a mismatch.

The maximum value in the dynamic-programming matrix represents the best local-alignment score.

### CUDA implementation

The current kernel assigns one GPU thread to each independent reference-query pair. Each thread computes the alignment using two dynamic-programming rows rather than storing the entire matrix.

The kernel declares two local arrays of 1,025 integers, which supports sequence lengths up to 1,024 with the current implementation. Actual performance depends on compiler decisions, register allocation, memory traffic, GPU architecture, and workload size.

## 🧰 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application logic and benchmark orchestration |
| NumPy | Numerical arrays and sequence processing |
| CuPy | GPU arrays and CUDA integration |
| C++ / CUDA | GPU alignment kernel |
| Python multiprocessing | Parallel CPU execution |
| Biopython | FASTA parsing, where used |
| pandas | Tabular benchmark analysis, where used |
| Matplotlib | Benchmark visualization, where used |

## 🧬 Reference Genome

The reference sequence used in the recorded experiments is **Homo sapiens chromosome 1, GRCh38**, identified by NCBI accession `CM000663.2`.

**Reference:** [NCBI Nucleotide — CM000663.2](https://www.ncbi.nlm.nih.gov/nuccore/CM000663.2/)

Expected reference properties:

| Property | Value |
|---|---|
| Organism | *Homo sapiens* |
| Chromosome | 1 |
| Assembly | GRCh38 |
| Accession | `CM000663.2` |
| Sequence length | 248,956,422 bases |
| Normalized sequence MD5 | `6aef897c3d6ff0c78aff06ac189178dd` |

The checksum applies to the normalized sequence, not necessarily to the downloaded FASTA file's raw bytes.

### Download the reference FASTA

The reference genome is not included in the repository. Download it separately and save it as `sequence.fasta` in the project directory.

**Linux or macOS**

```bash
curl -L \
  "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=CM000663.2&rettype=fasta&retmode=text" \
  -o sequence.fasta
```

**Windows PowerShell**

```powershell
Invoke-WebRequest `
  -Uri "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=CM000663.2&rettype=fasta&retmode=text" `
  -OutFile "sequence.fasta"
```

After downloading, verify that the file contains the expected accession and sequence. The loader must be able to find `sequence.fasta` at the configured path.

## 💻 Installation

### Prerequisites

- Git
- [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) or [Anaconda](https://www.anaconda.com/download)
- An NVIDIA GPU compatible with the project's CUDA environment for GPU execution
- The reference FASTA file for reference-based experiments

### 1. Clone the repository

```bash
git clone https://github.com/AntonioNDev/PiDP-Project.git
cd PiDP-Project
```

### 2. Create the Conda environment

The repository includes a `.yml` environment file containing the project's Python dependencies and environment configuration. Use it to create the environment instead of installing packages individually.

```bash
conda env create -f environment.yml
```

### 3. Activate the environment

The environment name is defined inside the `.yml` file under the `name` field. Activate it using that name:

```bash
conda activate <environment-name>
```

### 4. Verify GPU availability

With the environment activated, run:

```bash
python -c "import cupy as cp; print('CuPy:', cp.__version__); print('GPU count:', cp.cuda.runtime.getDeviceCount())"
```

A GPU count of at least one indicates that CuPy can detect a CUDA device. This does not guarantee that every kernel will compile or execute successfully.

**Note:** CUDA execution requires a compatible NVIDIA GPU, driver, and software environment. CPU-only systems can run supported CPU tests but cannot execute the CUDA kernel.

### 4. Verify GPU availability

Run:

```bash
python -c "import cupy as cp; print('CuPy:', cp.__version__); print('GPU count:', cp.cuda.runtime.getDeviceCount())"
```

A GPU count of at least one confirms that CuPy can detect a CUDA device. It does not by itself guarantee that the project's kernel compiles or executes successfully.

## ▶️ Running the Project

### Quick correctness test

The lightweight test configuration should generate a small set of synthetic DNA sequence pairs, run the CPU and GPU implementations, and compare their alignment scores.

```bash
python main.py
```

This command is appropriate for the quick test only if the current `main.py` is configured to run it. The exact behavior depends on the active entry point.

Expected behavior:

1. Generate or load a small sequence workload.
2. Calculate CPU alignment scores.
3. Calculate CUDA alignment scores.
4. Compare the results.
5. Report success if the scores match.

A small synthetic test can verify basic correctness without downloading the full reference genome, provided the selected entry point does not invoke the FASTA loader. CUDA testing still requires a compatible NVIDIA environment.

### Full benchmark

For the complete experiment, configure the benchmark to use the reference FASTA and compare serial CPU, multiprocessing CPU, and CUDA GPU execution.

Record the following parameters for each run:

- Number of reference-query pairs.
- Sequence length.
- Mutation rate and transition bias.
- Random seed.
- Number of repetitions and warm-up runs.
- CPU, GPU, driver, and CUDA versions.
- Whether initialization, memory allocation, and host/device transfers are included in timing.

The exact full-benchmark command should match the current runner in the repository. The quick correctness test should not be assumed to execute the complete benchmark.

## 📊 Benchmark Results

The results below summarize selected validation checks and a recorded GPU timing experiment. They are provided as measurements from specific runs, not as universal performance guarantees.

### Correctness and mutation validation

| Validation | Configuration | Result |
|---|---|---|
| CPU/GPU alignment-score parity | 64 sequence pairs | 0 mismatches |
| Boundary sequence lengths | 1, 128, and 1,024 | Passed |
| Identical-sequence scoring | Matching reference and query | Expected scores confirmed |
| Zero mutation rate | Mutation-generation boundary test | Passed |
| Mutation-rate validation | Target rate: 10% | Observed rate: 9.9323% |
| Transition-bias validation | Target transition fraction: 70% | Observed fraction: 69.6854% |
| Mutation classification | 19,070 changed positions in the recorded test | All classified in that test |

These checks support the tested implementation behavior. They do not establish clinical sensitivity, specificity, or cancer-detection accuracy.

### GPU execution time

The following measurements were recorded using sequence length \(L = 128\), with seven repetitions per workload. Timing covered the `smith_waterman_gpu()` wrapper, including allocation and data-transfer overhead.

| Number of pairs (\(N\)) | Median time (s) | Minimum (s) | Maximum (s) |
|---:|---:|---:|---:|
| 250 | 0.024505 | 0.021917 | 0.035402 |
| 500 | 0.005195 | 0.004706 | 0.005395 |
| 1,000 | 0.010047 | 0.009653 | 0.011516 |
| 1,500 | 0.016776 | 0.016717 | 0.018439 |

**Interpretation:** The 250-pair measurement is an outlier compared with the neighboring workloads. Initialization overhead, measurement variability, or other runtime effects may influence small workloads. The values should therefore be treated as preliminary measurements rather than a definitive scaling curve.

These GPU measurements must not be directly compared with CPU timings from a different run. Reliable speedup calculations require CPU and GPU measurements from the same controlled experiment, using equivalent workloads and clearly defined timing boundaries.

### Performance visualization

![img1](images/mutation_impact.jpg)

*Figure 1: compare serial CPU, multiprocessing CPU, and CUDA GPU execution time across workload sizes. Use CPU and GPU results collected under the same experimental conditions.*

### Mutation-rate visualization

![img1](images/mutation_impact.jpg)

*Figure 2: compare target and observed mutation rates across multiple configurations. If available, show transition and transversion frequencies separately.*

### Speedup calculation

Once comparable CPU and GPU timings are available, calculate speedup as:

\[
\text{Speedup} =
\frac{\text{CPU execution time}}
{\text{GPU execution time}}
\]

Report speedup separately for serial CPU and multiprocessing CPU. Include the workload size, sequence length, hardware, repetition count, and timing methodology so readers can reproduce the comparison.

## ⚠️ Limitations

PiDP currently focuses on performance benchmarking and synthetic mutation experiments.

- Simulated substitutions do not capture all biological mechanisms or the full complexity of cancer genomes.
- A transition-biased mutation model is a configurable simulation assumption, not evidence of cancer-specific behavior.
- Smith–Waterman alignment scores measure local sequence similarity; they do not independently establish whether a mutation is pathogenic or cancer-associated.
- The CUDA kernel uses fixed-size arrays and currently supports sequence lengths up to 1,024.
- CUDA compatibility and performance depend on the hardware and software environment.
- Successful tests on selected inputs do not prove correctness for every possible sequence or workload.
- The recorded benchmark results do not establish clinical diagnostic performance.


## 🔭 Future Work

Potential improvements include:

- A dedicated command-line interface for quick tests and full benchmarks.
- Automated CPU/GPU parity and boundary-condition tests.
- Reproducible benchmark scripts with configurable workload sizes.
- Consistent timing and profiling across CPU and GPU implementations.
- More detailed reporting of measurement variability.
- CUDA optimization guided by profiling results.
- A separate future project focused on biological variant analysis and genome visualization.

## 👤 Author

**Antonio Nikoloski**  
North Macedonia

- GitHub profile: [AntonioNDev](https://github.com/AntonioNDev)
- Repository: [PiDP-Project](https://github.com/AntonioNDev/PiDP-Project)

---

*PiDP investigates the performance trade-offs between serial CPU, multiprocessing CPU, and CUDA GPU execution for DNA sequence alignment. All conclusions should be interpreted within the tested workloads and hardware environment.*
