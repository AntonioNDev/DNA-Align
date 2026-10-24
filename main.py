import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, cpu_count
from Bio import SeqIO
from functools import wraps

# CONSTANTS
MATCH = 2
MISMATCH = -1
GAP = -1


ENCODE = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': -1}

TRANSITIONS = {
   0: [2],  # A → G
   2: [0],  # G → A
   1: [3],  # C → T
   3: [1],  # T → C
}

ALL_BASES = [0, 1, 2, 3]

class DataProcessing:
   """
      Loads and encodes genome data and generates reproducible,
      transition-biased synthetic mutation pairs.
   """

   def __init__(self, seq_path, window_length, num_pairs, mutation_rates, transition_bias=0.7, random_seed=42):
      """
      Parameters:
         seq_path (str): Path to FASTA genome file
         window_length (int): Length of extracted genomic windows
         num_pairs (int): Number of reference/query pairs
         mutation_rates (list): Mutation rates for experiments
         transition_bias (float): Probability of transition mutation
      """

      # Validate the inputs
      if not isinstance(window_length, (int, np.integer)) or window_length <= 0:
         raise ValueError("window_length must be a positive integer.")
      if not isinstance(num_pairs, (int, np.integer)) or num_pairs < 0:
         raise ValueError("num_pairs must be a non-negative integer.")
      if not 0.0 <= transition_bias <= 1.0:
         raise ValueError("transition_bias must be between 0 and 1.")

      self.seq_path = seq_path
      self.window_length = int(window_length)
      self.num_pairs = int(num_pairs)
      self.mutation_rates = list(mutation_rates)
      self.transition_bias = float(transition_bias)

      # Seed generator that makes experiments reproducible.
      self.rng = np.random.default_rng(random_seed)
   
   def load_fasta_and_encode(self):
      """
         Load FASTA records and return contiguous A/C/G/T segments.

         Ambiguous bases split segments. FASTA records are never joined.
         Returns a list of 1D int8 NumPy arrays.
      """
      base_map = np.full(256, -1, dtype=np.int8)
      base_map[ord('A')] = 0
      base_map[ord('C')] = 1
      base_map[ord('G')] = 2
      base_map[ord('T')] = 3

      segments = []

      for record in SeqIO.parse(self.seq_path, "fasta"):
         sequence = str(record.seq).upper().encode("ascii")
         encoded = base_map[np.frombuffer(sequence, dtype=np.uint8)]

         valid = encoded >= 0
         boundaries = np.diff(
            np.concatenate(([False], valid, [False])).astype(np.int8)
         )
         starts = np.flatnonzero(boundaries == 1)
         ends = np.flatnonzero(boundaries == -1)

         for start, end in zip(starts, ends):
            if end > start:
               segments.append(encoded[start:end].copy())

      if not segments:
         raise ValueError(
            "No valid A/C/G/T sequence segments were found in the FASTA file."
         )

      return segments

   def mutate_base_cancer(self, base):
      """
         Applies a cancer-like mutation to a nucleotide.

         Parameters:
         base (int): Encoded nucleotide (0–3)

         Returns:
         int: Mutated nucleotide
      """
      base = int(base)
      
      if base not in ALL_BASES:
         raise ValueError(f"Invalid encoded DNA base: {base}")

      if self.rng.random() < self.transition_bias:
         return int(self.rng.choice(TRANSITIONS[base]))

      transversions = [
         candidate for candidate in ALL_BASES if candidate not in TRANSITIONS[base]
      ]

      return int(self.rng.choice(transversions))

   def generate_cancer_mutated_pairs(self, genome, mutation_rate):
      """
      Generate reference/query pairs from valid contiguous DNA segments.

      Windows are sampled uniformly from all possible valid window
      positions across all segments.
      
      Parameters:
      genome (np.ndarray): Encoded genome
      mutation_rate (float): Probability of mutation per base

      Returns:
      (refs, queries): int8 arrays of shape (num_pairs, window_length)
      """

      if not np.isfinite(mutation_rate) or not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("mutation_rate must be between 0 and 1.")

      L = self.window_length
      segments = []

      for segment in genome:
         segment = np.asarray(segment)

         if segment.ndim != 1:
               raise ValueError("Each genome segment must be one-dimensional.")

         if not np.isin(segment, [0, 1, 2, 3]).all():
               raise ValueError("Segments must contain only encoded A/C/G/T bases.")

         if len(segment) >= L:
               segments.append(segment.astype(np.int8, copy=False))

      if self.num_pairs == 0:
         empty = np.empty((0, L), dtype=np.int8)
         return empty.copy(), empty.copy()

      if not segments:
         raise ValueError(
            f"No valid contiguous DNA segment is at least {L} bases long."
         )

      # Number of valid window starts per segment.
      counts = np.array(
         [len(segment) - L + 1 for segment in segments],
         dtype=np.int64,
      )
      cumulative = np.cumsum(counts)
      total_windows = int(cumulative[-1])

      # Sample uniformly from every possible valid window position.
      choices = self.rng.integers(
         0, total_windows, size=self.num_pairs
      )

      segment_indices = np.searchsorted(cumulative, choices, side="right")
      previous_cumulative = np.concatenate(([0], cumulative[:-1]))
      local_starts = choices - previous_cumulative[segment_indices]

      refs = np.empty((self.num_pairs, L), dtype=np.int8)
      queries = np.empty_like(refs)

      for i, (segment_idx, start) in enumerate(zip(segment_indices, local_starts)):
         reference = segments[segment_idx][start : start + L].copy()
         query = reference.copy()

         mutation_mask = self.rng.random(L) < mutation_rate

         for position in np.flatnonzero(mutation_mask):
            query[position] = self.mutate_base_cancer(query[position])

         refs[i] = reference
         queries[i] = query

      return refs, queries

class BenchmarkData(DataProcessing):
   """
   Implements CPU Serial/Parallel and GPU CUDA Smith–Waterman alignment and benchmarking.
   """
   def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)

      # Load CUDA kernel from file
      with open("sw_kernel.cu", "r") as f:
         kernel_code = f.read()

      self.sw_kernel = cp.RawKernel(kernel_code, "smith_waterman_kernel")
   
   def measure_time(func):
      """
      A decorator that measures the execution time of a function.
      Returns only the time (seconds) as a float.

      """
      @wraps(func)
      def wrapper(self, *args, **kwargs):
         cp.cuda.Stream.null.synchronize()

         start = time.perf_counter()

         func(self, *args, **kwargs)

         cp.cuda.Stream.null.synchronize()

         return time.perf_counter() - start
      return wrapper

   def smith_waterman_gpu(self, refs, queries):
      """
      Runs Smith–Waterman alignment on GPU.

      Parameters:
      refs (np.ndarray): Reference sequences (N × L)
      queries (np.ndarray): Query sequences (N × L)

      Returns:
      np.ndarray: Alignment scores
      """
      N, L = refs.shape

      d_refs = cp.asarray(refs)
      d_queries = cp.asarray(queries)
      d_scores = cp.zeros(N, dtype=cp.int32)

      if N < 512:
         threads = 64
      elif N < 2048:
         threads = 128
      else:
         threads = 256

      blocks = (N + threads - 1) // threads

      self.sw_kernel((blocks,), (threads,), (d_refs, d_queries, d_scores, L, N))

      cp.cuda.Stream.null.synchronize()
      return d_scores.get()

   def smith_waterman_cpu(self, query, reference):
      """
         CPU implementation of Smith–Waterman local alignment.

         Parameters:
         query (np.ndarray): Query sequence
         reference (np.ndarray): Reference sequence

         Returns:
         int: Maximum alignment score
      """
      prev = np.zeros(len(reference) + 1, dtype=np.int32)
      curr = np.zeros(len(reference) + 1, dtype=np.int32)
      best = 0

      for i in range(1, len(query) + 1):
         curr[0] = 0
         for j in range(1, len(reference) + 1):
               score = max(
                  0,
                  prev[j-1] + (MATCH if query[i-1] == reference[j-1] else MISMATCH),
                  prev[j] + GAP,
                  curr[j-1] + GAP
               )
               curr[j] = score
               best = max(best, score)
         prev, curr = curr, prev

      return best

   def benchmark_with_repeats(self, func, *args, repeats=0):
      import gc

      times = []
      for _ in range(repeats):
         # Чистење на меморија пред секој обид
         cp.get_default_memory_pool().free_all_blocks()
         gc.collect() 
         times.append(func(*args))
      return np.min(times)

   def warmup_gpu(self):
      """GPU initialization for CUDA (here the time is not calculated)"""
      print("Initializing GPU (Warmup)...")
      dummy_data = np.zeros((10, 64), dtype=np.int8)
      self.smith_waterman_gpu(dummy_data, dummy_data)
      print("GPU Ready.")

   def batch_smith_waterman_cpu(self, queries, references):
      """
         Runs Smith–Waterman on a batch of alignments (CPU serial).

         Parameters:
         queries (np.ndarray): Query sequences (N × L)
         references (np.ndarray): Reference sequences (N × L)

         Returns:
         np.ndarray: Alignment scores
      """
      scores = np.zeros(len(queries), dtype=np.int32)

      for i in range(len(queries)):
         scores[i] = self.smith_waterman_cpu(queries[i], references[i])

      return scores
   
   def _sw_worker(self, args):
      """
         Worker function for multiprocessing Smith–Waterman.
      """
      query, reference = args
      return self.smith_waterman_cpu(query, reference)

   def batch_smith_waterman_cpu_parallel(self, queries, references, num_workers=None):
      """
         Parallel CPU Smith–Waterman using multiprocessing.

         Parameters:
         queries (np.ndarray): Query sequences
         references (np.ndarray): Reference sequences
         num_workers (int): Number of CPU processes

         Returns:
         np.ndarray: Alignment scores
      """
      if num_workers is None:
         num_workers = cpu_count()

      tasks = [(queries[i], references[i]) for i in range(len(queries))]

      with Pool(processes=num_workers) as pool:
         scores = pool.map(self._sw_worker, tasks)

      return np.array(scores, dtype=np.int32)
   @measure_time
   def benchmark_gpu(self, refs, queries):
      """Runs GPU Smith-Waterman (Timed). Note: Warmup must be called separately."""
      return self.smith_waterman_gpu(refs, queries)
   @measure_time
   def benchmark_cpu(self, refs, queries):
      """Runs Serial CPU Smith-Waterman (Timed)."""
      return self.batch_smith_waterman_cpu(queries, refs)
   @measure_time
   def benchmark_cpu_parallel(self, refs, queries):
      """Runs Parallel CPU Smith-Waterman (Timed)."""
      return self.batch_smith_waterman_cpu_parallel(queries, refs)

   def mutation_impact_experiment(self, genome):
      """
         Measures alignment score drop under increasing mutation rates.

         Parameters:
         genome (np.ndarray): Encoded genome

         Returns:
         (rates, mean_drops, std_drops)
      """
      mean_drops = []
      std_drops = []

      for rate in self.mutation_rates:
         print(rate)
         refs, queries = self.generate_cancer_mutated_pairs(genome, rate)

         clean = self.smith_waterman_gpu(refs, refs)
         mutated = self.smith_waterman_gpu(queries, refs)

         drops = clean - mutated
         mean_drops.append(np.mean(drops))
         std_drops.append(np.std(drops))

      return self.mutation_rates, mean_drops, std_drops
   
   def analyze_mutation_spectrum(self, refs, queries):
      """
         Counts transition vs transversion mutations.

         Parameters:
         refs (np.ndarray): Reference sequences
         queries (np.ndarray): Mutated sequences

         Returns:
         (transition_count, transversion_count)
      """
      transition_count = 0
      transversion_count = 0

      for i in range(len(refs)):
         for j in range(len(refs[i])):
            if refs[i][j] != queries[i][j]:
               if queries[i][j] in TRANSITIONS[refs[i][j]]:
                  transition_count += 1
               else:
                  transversion_count += 1

      return transition_count, transversion_count

class PlotData:
   """
   Visualization utilities for benchmarking and mutation experiments.
   """

   plt.rcParams.update({
      "font.size": 10,
      "axes.labelsize": 12,
      "axes.titlesize": 13,
      "legend.fontsize": 10,
      "xtick.labelsize": 10,
      "ytick.labelsize": 10,
      "axes.grid": False,
    })
  
   @staticmethod
   def get_trendline(x, y):   
      coeffs = np.polyfit(x, y, 1)
      return np.poly1d(coeffs)(x)

   @staticmethod
   def print_summary_tables(results, rates, means, stds):
      print("\n" + "="*95)
      print(f"{'TABELA 1 & 2: VREME NA IZVRSUVANJE I ZABRZUVANJE':^95}")
      print("="*95)
      header = f"{'N (Aligns)':^12} | {'CPU Serial(s)':^15} | {'CPU Par(s)':^15} | {'GPU CUDA(s)':^13} | {'SpdUp(vsSer)':^12} | {'SpdUp(vsPar)':^12}"
      print(header)
      print("-" * 95)
      
      for r in results:
         speedup_ser = r['cpu_serial'] / r['gpu'] if r['gpu'] > 0 else 0
         speedup_par = r['cpu_parallel'] / r['gpu'] if r['gpu'] > 0 else 0
         
         print(f"{r['N']:^12} | {r['cpu_serial']:^15.4f} | {r['cpu_parallel']:^15.4f} | {r['gpu']:^13.4f} | {speedup_ser:>10.2f}x  | {speedup_par:>10.2f}x")
      
      print("="*95)

      if rates and means:
         print("\n\n" + "="*65)
         print(f"{'TABELA 3: MUTATION IMPACT ANALIZA':^65}")
         print("="*65)
         mut_header = f"{'Mutation Rate':^18} | {'Mean Score Drop':^22} | {'Std Dev':^15}"
         print(mut_header)
         print("-" * 65)
         
         for rate, mean, std in zip(rates, means, stds):
            print(f"{rate:^18.2f} | {mean:^22.2f} | {std:^15.2f}")
         
         print("="*65 + "\n")

   @staticmethod
   def plot_all(results, mut_rates, mut_means, mut_stds):
      Ns = np.array([r["N"] for r in results])
      cpu_s = np.array([r["cpu_serial"] for r in results])
      cpu_p = np.array([r["cpu_parallel"] for r in results])
      gpu = np.array([r["gpu"] for r in results])

      # (Trendlines)
      """ t_cpu_s = PlotData.get_trendline(Ns, cpu_s)
      t_cpu_p = PlotData.get_trendline(Ns, cpu_p)
      t_gpu = PlotData.get_trendline(Ns, gpu) """

      # RUNTIME SCALING
      plt.figure(figsize=(9, 6))

      plt.scatter(Ns, cpu_s, color='C0', alpha=0.2, s=30, label='_nolegend_')
      plt.scatter(Ns, cpu_p, color='C1', alpha=0.2, s=30, label='_nolegend_')
      plt.scatter(Ns, gpu, color='C2', alpha=0.2, s=30, label='_nolegend_')

      plt.plot(Ns, cpu_s, label="CPU Serial", color='C0', linewidth=2)
      plt.plot(Ns, cpu_p, label="CPU Parallel", color='C1', linewidth=2)
      plt.plot(Ns, gpu, label="GPU CUDA", color='C2', linewidth=2)

      plt.title("Smith-Waterman Runtime Scaling")
      plt.xlabel("Number of Alignments (N)")
      plt.ylabel("Runtime (seconds)")
      plt.xticks(Ns, rotation=45)
      plt.legend()
      plt.tight_layout()
      plt.savefig("1_runtime.pdf", dpi=300)

      # GPU SPEEDUP
      plt.figure(figsize=(9, 6))

      """ plt.scatter(Ns, cpu_s / gpu, color='C0', alpha=0.15, s=25, label='_nolegend_')
      plt.scatter(Ns, cpu_p / gpu, color='C1', alpha=0.15, s=25, label='_nolegend_') """

      plt.plot(Ns, cpu_s / gpu, label="GPU vs CPU Serial", color='C0', marker='o', markersize=4)
      plt.plot(Ns, cpu_p / gpu, label="GPU vs CPU Parallel", color='C1', marker='s', markersize=4)
      
      plt.title("GPU Speedup Factor")
      plt.xlabel("Number of Alignments (N)")
      plt.ylabel("Speedup (x)")
      plt.xticks(Ns, rotation=45)
      plt.legend()
      plt.tight_layout()
      plt.savefig("2_speedup.pdf", dpi=300)

      # THROUGHPUT
      plt.figure(figsize=(9, 6))

      """ plt.scatter(Ns, Ns / cpu_s, color='C0', alpha=0.2, s=25, label='_nolegend_')
      plt.scatter(Ns, Ns / cpu_p, color='C1', alpha=0.2, s=25, label='_nolegend_')
      plt.scatter(Ns, Ns / gpu, color='C2', alpha=0.2, s=25, label='_nolegend_') """

      plt.plot(Ns, Ns / cpu_s, label="CPU Serial", color='C0', marker='o', markersize=4)
      plt.plot(Ns, Ns / cpu_p, label="CPU Parallel", color='C1', marker='s', markersize=4)
      plt.plot(Ns, Ns / gpu, label="GPU CUDA", color='C2', marker='^', markersize=4)
      
      plt.title("System Throughput")
      plt.xlabel("Number of Alignments (N)")
      plt.ylabel("Alignments per Second")
      plt.xticks(Ns, rotation=45)
      plt.legend()
      plt.tight_layout()
      plt.savefig("3_throughput.pdf", dpi=300)

      # MUTATION IMPACT 
      plt.figure(figsize=(9, 6))
      plt.errorbar(mut_rates, mut_means, yerr=mut_stds, fmt='-o', color='C2', 
                  ecolor='black', capsize=4, label="Score Drop")
      plt.title("Impact of Mutation Rate on Alignment Score")
      plt.xlabel("Mutation Rate")
      plt.ylabel("Mean Score Drop")
      plt.xticks(mut_rates, rotation=45)
      plt.tight_layout()
      plt.savefig("4_mutation_impact.pdf", dpi=300)

      plt.show()

if __name__ == "__main__":
   # Real testing data results.
   results_for_plotting = [
      {"N": 500,  "cpu_serial": 20.3907,  "cpu_parallel": 9.3538,  "gpu": 0.0319},
      {"N": 1000, "cpu_serial": 25.9704,  "cpu_parallel": 17.0595, "gpu": 0.0606},
      {"N": 1500, "cpu_serial": 29.9612,  "cpu_parallel": 24.2830, "gpu": 0.2003},
      {"N": 2000, "cpu_serial": 41.5980,  "cpu_parallel": 29.7689, "gpu": 0.2072},
      {"N": 2500, "cpu_serial": 62.9430,  "cpu_parallel": 31.8414, "gpu": 0.3812},
      {"N": 3000, "cpu_serial": 65.2359,  "cpu_parallel": 37.2655, "gpu": 0.2452},
      {"N": 3500, "cpu_serial": 85.0904,  "cpu_parallel": 44.7147, "gpu": 0.2731},
      {"N": 4000, "cpu_serial": 88.1515,  "cpu_parallel": 51.0994, "gpu": 0.2935},
      {"N": 4500, "cpu_serial": 98.7122,  "cpu_parallel": 55.1631, "gpu": 0.2912},
      {"N": 5000, "cpu_serial": 110.1804, "cpu_parallel": 68.0932, "gpu": 0.2667},
      {"N": 5500, "cpu_serial": 130.2973, "cpu_parallel": 71.2218, "gpu": 0.2324},
      {"N": 6000, "cpu_serial": 131.9491, "cpu_parallel": 73.4982, "gpu": 0.3229},
      {"N": 6500, "cpu_serial": 153.3262, "cpu_parallel": 80.2527, "gpu": 0.2050},
      {"N": 7000, "cpu_serial": 159.0099, "cpu_parallel": 84.0021, "gpu": 0.2894},
      {"N": 7500, "cpu_serial": 178.1431, "cpu_parallel": 90.7757, "gpu": 0.2213},
      {"N": 8000, "cpu_serial": 180.1929, "cpu_parallel": 99.5048, "gpu": 0.2759},
      {"N": 8500, "cpu_serial": 194.2695, "cpu_parallel": 105.7006, "gpu": 0.3679},
      {"N": 9000, "cpu_serial": 207.9494, "cpu_parallel": 123.4033, "gpu": 0.5148},
      {"N": 9500, "cpu_serial": 221.9495, "cpu_parallel": 125.5227, "gpu": 0.4098},
      {"N": 10000, "cpu_serial": 228.8388, "cpu_parallel": 127.9106, "gpu": 0.2992},
   ]

   # DATA FOR MUTATION IMPACT
   rates = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80, 1.00]
   means = [2.98, 18.76, 36.72, 54.19, 70.71, 101.16, 147.49, 174.37, 180.76]
   stds  = [0.13, 7.15, 9.40, 11.01, 11.89, 12.57, 11.31, 12.55, 16.17]
   
   plotter = PlotData()
      
   plotter.plot_all(results_for_plotting, rates, means, stds)
