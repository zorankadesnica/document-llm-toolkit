"""
Computational Complexity Analysis for Document-Level Models

Provides detailed complexity analysis comparing different approaches
to document-level context modeling.

Key Results:
- Full concatenation: O((NL)² d) = O(N²L²d) — quadratic in document length
- Hierarchical: O(NkL²d + N²d) — linear in N for fixed k
- Sparse attention: O(NL·s·d) where s << NL

For N=50 sentences, L=30 tokens:
- Concatenation: 2,250,000 operations
- Hierarchical (k=3): 137,500 operations (16x faster!)

Reference: Vaswani et al. (2017) "Attention Is All You Need"
           Stojanovski & Fraser (2019) "Combining Local and Document-Level Context"
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import time


class AttentionType(Enum):
    """Types of attention mechanisms."""
    FULL = "full"                    # Standard O(n²) attention
    LOCAL = "local"                  # Fixed window
    SPARSE = "sparse"                # Learned sparsity
    HIERARCHICAL = "hierarchical"    # Sentence + token level


@dataclass
class ComplexityResult:
    """Result of complexity analysis."""
    theoretical_ops: int
    memory_bytes: int
    estimated_time_ms: float
    bottleneck: str
    recommendations: List[str]


@dataclass
class DocumentConfig:
    """Configuration for document processing."""
    num_sentences: int      # N
    avg_sentence_length: int  # L
    model_dim: int          # d
    num_heads: int          # h
    local_context_size: int  # k (for hierarchical)
    sparsity_factor: float  # s (for sparse attention)
    dtype_bytes: int = 4    # float32 = 4 bytes


class AttentionComplexityAnalyzer:
    """
    Analyzes computational complexity of different attention mechanisms.
    
    Theoretical Background:
    
    Standard self-attention:
        Attention(Q, K, V) = softmax(QK^T / √d_k) V
        
        Operations breakdown:
        1. QK^T: O(n² d_k) multiplications
        2. Softmax: O(n²) operations
        3. Attention × V: O(n² d_v) multiplications
        
        Total: O(n² d) where d = d_k = d_v
        
    Memory:
        - Q, K, V matrices: O(n d)
        - Attention weights: O(n²)
        - Output: O(n d)
        Total: O(n² + nd) ≈ O(n²) for large n
    """
    
    def __init__(self, config: DocumentConfig):
        self.config = config
        
        # Estimated FLOPS per operation (rough estimate for modern GPUs)
        self.flops_per_second = 10e12  # 10 TFLOPS
    
    def analyze_full_attention(self) -> ComplexityResult:
        """
        Analyze full concatenation attention.
        
        Sequence length = N × L (all sentences concatenated)
        
        Complexity: O((NL)² d) = O(N²L²d)
        
        This is the BOTTLENECK for document-level processing.
        """
        N = self.config.num_sentences
        L = self.config.avg_sentence_length
        d = self.config.model_dim
        
        seq_len = N * L
        
        # QK^T: seq_len² × d multiplications
        qk_ops = seq_len * seq_len * d
        
        # Softmax: seq_len² operations (exp, sum, divide)
        softmax_ops = 3 * seq_len * seq_len
        
        # Attention × V: seq_len² × d
        av_ops = seq_len * seq_len * d
        
        total_ops = qk_ops + softmax_ops + av_ops
        
        # Memory: attention weights matrix
        attn_memory = seq_len * seq_len * self.config.dtype_bytes
        
        # Q, K, V matrices
        qkv_memory = 3 * seq_len * d * self.config.dtype_bytes
        
        total_memory = attn_memory + qkv_memory
        
        # Estimated time
        est_time = (total_ops / self.flops_per_second) * 1000  # ms
        
        return ComplexityResult(
            theoretical_ops=total_ops,
            memory_bytes=total_memory,
            estimated_time_ms=est_time,
            bottleneck="O(N²L²) attention computation",
            recommendations=[
                "Use hierarchical attention for documents > 10 sentences",
                "Consider sparse attention patterns",
                "Batch sentences independently if context not needed"
            ]
        )
    
    def analyze_hierarchical_attention(self) -> ComplexityResult:
        """
        Analyze hierarchical (local + global) attention.
        
        Local: Token-level attention over k previous sentences
            Complexity per sentence: O(kL² d)
            Total: O(NkL² d)
            
        Global: Sentence-level attention over N sentences
            Complexity: O(N² d)
            
        Total: O(NkL²d + N²d)
        
        For fixed k (e.g., k=3), this is O(NL²d + N²d) ≈ O(NL²d) when L >> N
        """
        N = self.config.num_sentences
        L = self.config.avg_sentence_length
        d = self.config.model_dim
        k = self.config.local_context_size
        
        # Local attention: each sentence attends to k previous
        local_seq_len = (k + 1) * L  # Current + k previous
        local_ops_per_sent = local_seq_len * local_seq_len * d * 2  # QK^T + AV
        local_total_ops = N * local_ops_per_sent
        
        # Global attention: N sentence embeddings
        global_ops = N * N * d * 2  # Much smaller!
        
        total_ops = local_total_ops + global_ops
        
        # Memory: much smaller attention matrices
        local_memory = local_seq_len * local_seq_len * self.config.dtype_bytes
        global_memory = N * N * self.config.dtype_bytes
        qkv_memory = 3 * L * d * self.config.dtype_bytes  # Per sentence
        
        total_memory = local_memory + global_memory + qkv_memory
        
        # Estimated time
        est_time = (total_ops / self.flops_per_second) * 1000
        
        return ComplexityResult(
            theoretical_ops=total_ops,
            memory_bytes=total_memory,
            estimated_time_ms=est_time,
            bottleneck="O(kL²) local attention per sentence",
            recommendations=[
                f"Optimal for documents with N > {k * 2} sentences",
                "Tune k based on anaphora distance distribution",
                "Use efficient sentence pooling for global context"
            ]
        )
    
    def analyze_sparse_attention(self) -> ComplexityResult:
        """
        Analyze sparse attention (e.g., with sparsemax).
        
        Instead of full N×L attention, only attend to s << NL positions.
        
        Complexity: O(NL × s × d)
        
        Sparsity achieved via:
        - Sparsemax (Martins & Astudillo, 2016)
        - Top-k attention
        - Learned sparsity patterns
        """
        N = self.config.num_sentences
        L = self.config.avg_sentence_length
        d = self.config.model_dim
        s = int(self.config.sparsity_factor * N * L)  # Attended positions
        
        seq_len = N * L
        
        # Sparse attention: only s connections per query
        # Still need full QK^T to determine which positions, but can optimize
        selection_ops = seq_len * seq_len  # Finding top-s
        sparse_attn_ops = seq_len * s * d * 2  # Actual attention
        
        total_ops = selection_ops + sparse_attn_ops
        
        # Memory: sparse attention matrix
        sparse_memory = seq_len * s * self.config.dtype_bytes
        qkv_memory = 3 * seq_len * d * self.config.dtype_bytes
        
        total_memory = sparse_memory + qkv_memory
        
        # Estimated time
        est_time = (total_ops / self.flops_per_second) * 1000
        
        return ComplexityResult(
            theoretical_ops=total_ops,
            memory_bytes=total_memory,
            estimated_time_ms=est_time,
            bottleneck="Selection of top-s positions",
            recommendations=[
                "Use when document structure allows sparsity",
                "Combine with hierarchical for best results",
                f"Current sparsity: {self.config.sparsity_factor:.1%}"
            ]
        )
    
    def compare_all(self) -> Dict[str, ComplexityResult]:
        """Compare all attention mechanisms."""
        return {
            "full_concatenation": self.analyze_full_attention(),
            "hierarchical": self.analyze_hierarchical_attention(),
            "sparse": self.analyze_sparse_attention()
        }
    
    def generate_report(self) -> str:
        """Generate human-readable complexity report."""
        results = self.compare_all()
        
        N = self.config.num_sentences
        L = self.config.avg_sentence_length
        k = self.config.local_context_size
        
        report = []
        report.append("=" * 70)
        report.append("COMPUTATIONAL COMPLEXITY ANALYSIS")
        report.append("=" * 70)
        report.append(f"\nDocument Configuration:")
        report.append(f"  Sentences (N): {N}")
        report.append(f"  Avg sentence length (L): {L}")
        report.append(f"  Total tokens: {N * L}")
        report.append(f"  Model dimension (d): {self.config.model_dim}")
        report.append(f"  Local context size (k): {k}")
        
        report.append(f"\n{'Method':<25} {'Operations':>15} {'Memory (MB)':>12} {'Time (ms)':>10}")
        report.append("-" * 65)
        
        for method, result in results.items():
            mem_mb = result.memory_bytes / (1024 * 1024)
            report.append(
                f"{method:<25} {result.theoretical_ops:>15,} {mem_mb:>12.2f} {result.estimated_time_ms:>10.3f}"
            )
        
        # Speedup analysis
        full_ops = results["full_concatenation"].theoretical_ops
        hier_ops = results["hierarchical"].theoretical_ops
        speedup = full_ops / hier_ops
        
        report.append(f"\n{'=' * 70}")
        report.append("ANALYSIS")
        report.append("=" * 70)
        report.append(f"\nSpeedup (Hierarchical vs Full): {speedup:.1f}x")
        report.append(f"\nComplexity Formulas:")
        report.append(f"  Full concatenation: O((NL)²d) = O({N}×{L})²×d = O({(N*L)**2}d)")
        report.append(f"  Hierarchical: O(NkL²d + N²d) = O({N}×{k}×{L}²d + {N}²d)")
        report.append(f"                = O({N*k*L*L}d + {N*N}d)")
        
        report.append(f"\nMemory Scaling:")
        report.append(f"  Full: Stores {N*L}×{N*L} = {(N*L)**2:,} attention weights")
        report.append(f"  Hierarchical: Stores {(k+1)*L}×{(k+1)*L} + {N}×{N} = {((k+1)*L)**2 + N*N:,} weights")
        
        report.append(f"\nRecommendations:")
        for method, result in results.items():
            report.append(f"\n  {method}:")
            for rec in result.recommendations:
                report.append(f"    • {rec}")
        
        return "\n".join(report)


class MemoryEstimator:
    """
    Estimates memory requirements for document-level models.
    
    Critical for deployment: GPUs have limited memory!
    
    Memory components:
    1. Model parameters: O(L × d²) per layer
    2. Activations: O(batch × seq × d)
    3. Attention weights: O(batch × heads × seq²)
    4. Gradients (training): 2× parameters + activations
    """
    
    def __init__(self, config: DocumentConfig):
        self.config = config
    
    def estimate_inference_memory(
        self,
        batch_size: int = 1,
        num_layers: int = 6
    ) -> Dict[str, int]:
        """Estimate memory for inference."""
        N = self.config.num_sentences
        L = self.config.avg_sentence_length
        d = self.config.model_dim
        h = self.config.num_heads
        seq_len = N * L
        
        # Model parameters (shared across batch)
        # Each layer: 4 attention projections + 2 FFN
        params_per_layer = 4 * d * d + 2 * d * 4 * d  # Attention + FFN
        total_params = num_layers * params_per_layer
        param_memory = total_params * self.config.dtype_bytes
        
        # Activations (per batch item)
        act_per_layer = seq_len * d * 3  # Input, attention output, FFN output
        act_memory = batch_size * num_layers * act_per_layer * self.config.dtype_bytes
        
        # Attention weights (the big one!)
        attn_memory = batch_size * h * seq_len * seq_len * self.config.dtype_bytes
        
        return {
            "parameters_mb": param_memory / (1024 ** 2),
            "activations_mb": act_memory / (1024 ** 2),
            "attention_mb": attn_memory / (1024 ** 2),
            "total_mb": (param_memory + act_memory + attn_memory) / (1024 ** 2)
        }
    
    def estimate_training_memory(
        self,
        batch_size: int = 8,
        num_layers: int = 6
    ) -> Dict[str, int]:
        """Estimate memory for training (includes gradients)."""
        inference = self.estimate_inference_memory(batch_size, num_layers)
        
        # Gradients: same size as parameters
        grad_memory = inference["parameters_mb"]
        
        # Optimizer states (Adam: 2× parameters for momentum and variance)
        optimizer_memory = 2 * inference["parameters_mb"]
        
        # Activation checkpointing can reduce activation memory
        total = (
            inference["total_mb"] + 
            grad_memory + 
            optimizer_memory
        )
        
        return {
            "inference_mb": inference["total_mb"],
            "gradients_mb": grad_memory,
            "optimizer_mb": optimizer_memory,
            "total_mb": total
        }
    
    def find_max_batch_size(
        self,
        gpu_memory_gb: float = 16.0,
        num_layers: int = 6
    ) -> int:
        """Find maximum batch size for given GPU memory."""
        memory_budget = gpu_memory_gb * 1024  # Convert to MB
        
        for batch_size in range(1, 128):
            mem = self.estimate_training_memory(batch_size, num_layers)
            if mem["total_mb"] > memory_budget * 0.9:  # 90% budget
                return batch_size - 1
        
        return 128  # Max tested


class RuntimeProfiler:
    """
    Profile actual runtime of attention operations.
    
    Useful for validating theoretical analysis against real performance.
    """
    
    @staticmethod
    def profile_attention(
        seq_len: int,
        hidden_dim: int,
        num_heads: int,
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """Profile attention operation."""
        import torch
        
        # Create dummy tensors
        Q = torch.randn(1, num_heads, seq_len, hidden_dim // num_heads)
        K = torch.randn(1, num_heads, seq_len, hidden_dim // num_heads)
        V = torch.randn(1, num_heads, seq_len, hidden_dim // num_heads)
        
        # Warmup
        for _ in range(10):
            _ = torch.matmul(Q, K.transpose(-2, -1))
        
        # Profile
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            
            scores = torch.matmul(Q, K.transpose(-2, -1))
            scores = scores / np.sqrt(hidden_dim // num_heads)
            attn = torch.softmax(scores, dim=-1)
            output = torch.matmul(attn, V)
            
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
        
        return {
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times)
        }


def scaling_analysis(
    max_sentences: int = 100,
    sentence_length: int = 30,
    model_dim: int = 512
) -> List[Dict]:
    """
    Analyze how complexity scales with document length.
    
    Returns data for plotting complexity curves.
    """
    results = []
    
    for N in range(5, max_sentences + 1, 5):
        config = DocumentConfig(
            num_sentences=N,
            avg_sentence_length=sentence_length,
            model_dim=model_dim,
            num_heads=8,
            local_context_size=3,
            sparsity_factor=0.1
        )
        
        analyzer = AttentionComplexityAnalyzer(config)
        comparison = analyzer.compare_all()
        
        results.append({
            "num_sentences": N,
            "total_tokens": N * sentence_length,
            "full_ops": comparison["full_concatenation"].theoretical_ops,
            "hierarchical_ops": comparison["hierarchical"].theoretical_ops,
            "sparse_ops": comparison["sparse"].theoretical_ops,
            "speedup": (
                comparison["full_concatenation"].theoretical_ops /
                comparison["hierarchical"].theoretical_ops
            )
        })
    
    return results


# Example usage and demonstration
if __name__ == "__main__":
    # Configuration matching the example in the question
    config = DocumentConfig(
        num_sentences=50,
        avg_sentence_length=30,
        model_dim=512,
        num_heads=8,
        local_context_size=3,
        sparsity_factor=0.1
    )
    
    # Analyze complexity
    analyzer = AttentionComplexityAnalyzer(config)
    print(analyzer.generate_report())
    
    # Memory estimation
    print("\n" + "=" * 70)
    print("MEMORY ESTIMATION")
    print("=" * 70)
    
    mem_estimator = MemoryEstimator(config)
    
    inference_mem = mem_estimator.estimate_inference_memory(batch_size=1)
    print(f"\nInference Memory (batch=1):")
    for key, value in inference_mem.items():
        print(f"  {key}: {value:.2f} MB")
    
    training_mem = mem_estimator.estimate_training_memory(batch_size=8)
    print(f"\nTraining Memory (batch=8):")
    for key, value in training_mem.items():
        print(f"  {key}: {value:.2f} MB")
    
    max_batch = mem_estimator.find_max_batch_size(gpu_memory_gb=16.0)
    print(f"\nMax batch size for 16GB GPU: {max_batch}")
    
    # Scaling analysis
    print("\n" + "=" * 70)
    print("SCALING ANALYSIS")
    print("=" * 70)
    
    scaling = scaling_analysis(max_sentences=50, sentence_length=30)
    
    print(f"\n{'Sentences':>10} {'Tokens':>10} {'Full Ops':>15} {'Hier Ops':>15} {'Speedup':>10}")
    print("-" * 65)
    for row in scaling[::2]:  # Every other row
        print(
            f"{row['num_sentences']:>10} "
            f"{row['total_tokens']:>10} "
            f"{row['full_ops']:>15,} "
            f"{row['hierarchical_ops']:>15,} "
            f"{row['speedup']:>10.1f}x"
        )
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. Full concatenation scales QUADRATICALLY with document length
       O((NL)²) becomes prohibitive for N > 20 sentences
       
    2. Hierarchical approach scales LINEARLY for fixed local context k
       O(NkL² + N²) ≈ O(NL²) when L >> √N
       
    3. Memory is often the bottleneck, not compute
       Attention weights: O(N²L²) floats = 4N²L² bytes
       For N=50, L=30: 4×2500×900 = 9MB just for attention!
       
    4. Practical recommendations:
       - Documents < 10 sentences: Full concatenation OK
       - Documents 10-50 sentences: Use hierarchical
       - Documents > 50 sentences: Sparse + hierarchical essential
    """)
