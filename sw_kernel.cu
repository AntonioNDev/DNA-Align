// sw_kernel.cu
// CUDA implementation of Smith–Waterman local alignment
// One GPU thread computes one full alignment

extern "C" __global__
void smith_waterman_kernel(
   const signed char* refs,     // Flattened reference sequences (N × L)
   const signed char* queries,  // Flattened query sequences (N × L)
   int* scores,                 // Output alignment scores (N)
   int L,                       // Sequence length
   int N                        // Number of alignments
) {
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   if (tid >= N) return;

   const signed char* ref = refs + tid * L;
   const signed char* qry = queries + tid * L;

   // Thread-local DP rows (L ≤ 1024)
   int prev[1025];
   int curr[1025];

   for (int j = 0; j <= L; j++) prev[j] = 0;

   int max_score = 0;

   for (int i = 1; i <= L; i++) {
      curr[0] = 0;
      int qi = qry[i - 1];

      for (int j = 1; j <= L; j++) {
         int rj = ref[j - 1];

         int score_diag = prev[j - 1] + (qi == rj ? 2 : -1);
         int score_up   = prev[j] - 1;
         int score_left = curr[j - 1] - 1;

         int score = max(0, max(score_diag, max(score_up, score_left)));
         curr[j] = score;

         if (score > max_score) max_score = score;
      }

      for (int j = 0; j <= L; j++) prev[j] = curr[j];
   }

   scores[tid] = max_score;
}