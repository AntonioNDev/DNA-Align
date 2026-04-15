<div align="center">
  <h1>CUDA-Driven Parallel Detection of Cancer-Associated Mutations in the Human Genome</h1>
  <p><strong>Antonio Nikoloski, 226059</strong></p>
  <p><em>Skopje, North Macedonia</em></p>

  <img src="https://img.shields.io/badge/Language-Python%20%2F%20CUDA%20C%2B%2B-blue" alt="Language">
  <img src="https://img.shields.io/badge/Field-Bioinformatics-green" alt="Field">
  <img src="https://img.shields.io/badge/Status-Research%20in%20Progress-orange" alt="Status">
</div>

<hr />

## 🧬 Project Overview
Detection of cancer-associated mutations in whole-genome sequencing (WGS) data is computationally dominated by local sequence alignment[cite: 6]. This project introduces a <strong>CUDA-driven framework</strong> that parallelizes the <strong>Smith-Waterman algorithm</strong> to address the performance bottlenecks caused by its quadratic complexity[cite: 7].

By utilizing a heterogeneous CPU-GPU architecture, the system accelerates mutation detection while explicitly targeting transition-biased mutations (such as $C \rightarrow T$) characteristic of cancer genomes[cite: 9, 112].

## 🚀 Key Performance Results
Experimental results on real human reference sequences demonstrate a transformative leap in computational throughput[cite: 10, 180]:

<ul>
  <li><strong>Extreme Acceleration:</strong> Achieved speedups of up to <strong>2,228x</strong> over serial CPU execution[cite: 142, 180].</li>
  <li><strong>Multi-threaded Advantage:</strong> Outperformed multi-threaded CPU implementations by over <strong>1,300x</strong>[cite: 10, 142].</li>
  <li><strong>High Occupancy:</strong> The GPU efficiency peaks and reaches a stable high-throughput state beyond $N = 3,000$ alignments[cite: 181, 182].</li>
  <li><strong>Biological Fidelity:</strong> Maintained high accuracy, with alignment scores correctly penalizing mismatches as mutation rates increased[cite: 147, 183].</li>
</ul>

## 🛠️ Architecture and Implementation
The system maps independent alignments to individual CUDA threads using a memory-efficient strategy[cite: 8].

### 💻 System Workflow
<ol>
  <li><strong>Genome Encoding:</strong> DNA sequences are converted into integer representations[cite: 54].</li>
  <li><strong>Mutation Injection:</strong> Query sequences are generated with transition-biased mutations to simulate cancer-like variants[cite: 56].</li>
  <li><strong>Parallel Alignment:</strong> Thousands of reference-query pairs are aligned concurrently on the GPU[cite: 57, 85].</li>
  <li><strong>Post-processing:</strong> Results are transferred back to the CPU for statistical aggregation and mutation impact analysis[cite: 58, 113].</li>
</ol>

### 🧠 Memory Mapping
<ul>
  <li><strong>Global Memory:</strong> Stores encoded reference and query sequences[cite: 88].</li>
  <li><strong>Registers:</strong> Each thread maintains only two rows of the dynamic programming matrix to reduce the memory footprint[cite: 89].</li>
  <li><strong>Shared Memory:</strong> Stores constant scoring parameters (match, mismatch, gap penalties) for threads within a block[cite: 90].</li>
</ul>

## 📊 Experimental Data
The following table summarizes the runtime performance (in seconds) for $N$ alignments of length $L=128$[cite: 120, 138]:

<table align="center">
  <thead>
    <tr>
      <th>Alignments (N)</th>
      <th>CPU Serial (s)</th>
      <th>CPU Parallel (s)</th>
      <th>GPU CUDA (s)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>500</td>
      <td>20.39</td>
      <td>9.35</td>
      <td><strong>0.03</strong></td>
    </tr>
    <tr>
      <td>4,000</td>
      <td>83.15</td>
      <td>51.09</td>
      <td><strong>0.29</strong></td>
    </tr>
    <tr>
      <td>10,000</td>
      <td>228.83</td>
      <td>119.91</td>
      <td><strong>0.29</strong></td>
    </tr>
  </tbody>
</table>

## 📚 Research Context
This work is part of an ongoing research effort to optimize genomic pipelines[cite: 47]. The methodology ensures that biological sensitivity is preserved, allowing for the distinction between high-homology sequences and those containing significant somatic mutations[cite: 183].

---
<div align="center">
  <p>© 2026 Antonio Nikoloski. All rights reserved.</p>
</div>
