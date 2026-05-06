"""Spatial benchmark harness and evaluation for GeneVector."""
from .eval import (
    evaluate_ablation,
    evaluate_gene_pair_recovery,
    evaluate_phenotype_typing,
)
from .harness import (
    BenchmarkConfig,
    DEFAULT_TARGET_VARIANTS,
    run_benchmark,
)
from .report import write_report

__all__ = [
    "BenchmarkConfig",
    "DEFAULT_TARGET_VARIANTS",
    "run_benchmark",
    "evaluate_gene_pair_recovery",
    "evaluate_phenotype_typing",
    "evaluate_ablation",
    "write_report",
]
