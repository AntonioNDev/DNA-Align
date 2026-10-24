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
         candidate for candidate in ALL_BASES
         if candidate != base and candidate not in TRANSITIONS[base]
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
   
   @staticmethod
   def measure_time(func):
      """Measure elapsed wall-clock time in seconds."""
      @wraps(func)
      def wrapper(self, *args, **kwargs):
         is_gpu_benchmark = func.__name__ == "benchmark_gpu"

         # Exclude outstanding GPU work from the new measurement.
         if is_gpu_benchmark:
            cp.cuda.Stream.null.synchronize()

         start = time.perf_counter()
         func(self, *args, **kwargs)

         # Ensure GPU work has completed before stopping the clock.
         if is_gpu_benchmark:
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

   def benchmark_with_repeats(self, func, *args, repeats=5):
      """
      Return every measured runtime in seconds.

      Warm-up should be performed separately before calling this method.
      """
      if not isinstance(repeats, int) or repeats < 1:
         raise ValueError("repeats must be an integer >= 1")

      times = []

      for _ in range(repeats):
         elapsed = func(*args)
         times.append(float(elapsed))

      return times

   def warmup_gpu(self):
      """Initialize CUDA and warm up the kernel outside timed runs."""
      print(f"Initializing GPU (warm-up, L={self.window_length})...")

      dummy_data = np.zeros(
         (10, self.window_length),
         dtype=np.int8
      )

      self.smith_waterman_gpu(dummy_data, dummy_data)
      cp.cuda.Stream.null.synchronize()

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
      print(f"{'TABELA 1 & 2: TABLE TIME FOR SPEEDUPs':^95}")
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
         print(f"{'TABELA 3: MUTATION IMPACT':^65}")
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
         candidate for candidate in ALL_BASES
         if candidate != base and candidate not in TRANSITIONS[base]
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
   
   @staticmethod
   def measure_time(func):
      """Measure elapsed wall-clock time in seconds."""
      @wraps(func)
      def wrapper(self, *args, **kwargs):
         is_gpu_benchmark = func.__name__ == "benchmark_gpu"

         # Exclude outstanding GPU work from the new measurement.
         if is_gpu_benchmark:
            cp.cuda.Stream.null.synchronize()

         start = time.perf_counter()
         func(self, *args, **kwargs)

         # Ensure GPU work has completed before stopping the clock.
         if is_gpu_benchmark:
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

   def benchmark_with_repeats(self, func, *args, repeats=5):
      """
      Return every measured runtime in seconds.

      Warm-up should be performed separately before calling this method.
      """
      if not isinstance(repeats, int) or repeats < 1:
         raise ValueError("repeats must be an integer >= 1")

      times = []

      for _ in range(repeats):
         elapsed = func(*args)
         times.append(float(elapsed))

      return times

   def warmup_gpu(self):
      """Initialize CUDA and warm up the kernel outside timed runs."""
      print(f"Initializing GPU (warm-up, L={self.window_length})...")

      dummy_data = np.zeros(
         (10, self.window_length),
         dtype=np.int8
      )

      self.smith_waterman_gpu(dummy_data, dummy_data)
      cp.cuda.Stream.null.synchronize()

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
      print(f"{'TABELA 1 & 2: TABLE TIME FOR SPEEDUPs':^95}")
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
         print(f"{'TABELA 3: MUTATION IMPACT':^65}")
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
   WINDOW_LENGTH = 128
   NUM_PAIRS_LIST = [500, 1000, 1500]
   MUTATION_RATE = 0.10

   benchmark = BenchmarkData(
      seq_path="sequence.fasta",
      window_length=WINDOW_LENGTH,
      num_pairs=max(NUM_PAIRS_LIST),
      mutation_rates=[MUTATION_RATE],
      transition_bias=0.7,
      random_seed=42,
   )

   genome = benchmark.load_fasta_and_encode()
   benchmark.warmup_gpu()

   results = []

   for num_pairs in NUM_PAIRS_LIST:
      benchmark.num_pairs = num_pairs

      refs, queries = benchmark.generate_cancer_mutated_pairs(
         genome, MUTATION_RATE
      )

      # Verify CPU/GPU score parity.
      cpu_scores = benchmark.batch_smith_waterman_cpu(queries, refs)
      gpu_scores = benchmark.smith_waterman_gpu(refs, queries)

      if not np.array_equal(cpu_scores, gpu_scores):
         raise RuntimeError(f"CPU/GPU scores differ for N={num_pairs}!")

      # Measure each implementation once.
      cpu_serial = benchmark.benchmark_cpu(refs, queries)
      cpu_parallel = benchmark.benchmark_cpu_parallel(refs, queries)
      gpu = benchmark.benchmark_gpu(refs, queries)

      results.append({
         "N": num_pairs,
         "cpu_serial": cpu_serial,
         "cpu_parallel": cpu_parallel,
         "gpu": gpu,
      })

      # Observed mutation statistics.
      changed = int(np.count_nonzero(refs != queries))
      total_bases = refs.size
      observed_rate = changed / total_bases if total_bases else 0.0

      transitions, transversions = benchmark.analyze_mutation_spectrum(
         refs, queries
      )

      print(f"\n--- N={num_pairs}, L={WINDOW_LENGTH} ---")
      print("CPU/GPU correctness: PASSED")
      print(f"Target mutation rate:   {MUTATION_RATE:.2%}")
      print(f"Observed mutation rate: {observed_rate:.2%}")
      print(f"Mutated bases: {changed}/{total_bases}")
      print(f"Transitions: {transitions}")
      print(f"Transversions: {transversions}")
      print(f"CPU Serial:       {cpu_serial:.6f} s")
      print(f"CPU Multiprocess: {cpu_parallel:.6f} s")
      print(f"CUDA GPU:         {gpu:.6f} s")

   # Runtime comparison plot.
   Ns = [r["N"] for r in results]

   plt.figure(figsize=(8, 5))
   plt.plot(Ns, [r["cpu_serial"] for r in results],
            marker="o", label="CPU Serial")
   plt.plot(Ns, [r["cpu_parallel"] for r in results],
            marker="s", label="CPU Multiprocessing")
   plt.plot(Ns, [r["gpu"] for r in results],
            marker="^", label="CUDA GPU")

   plt.title(f"Smith–Waterman Runtime (L={WINDOW_LENGTH})")
   plt.xlabel("Number of Alignments (N)")
   plt.ylabel("Execution Time (seconds)")
   plt.xticks(Ns)
   plt.legend()
   plt.tight_layout()
   plt.savefig("benchmark_runtime.pdf", dpi=300)
   plt.show()

