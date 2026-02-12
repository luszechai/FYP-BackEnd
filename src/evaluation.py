"""Evaluation dashboard generation module"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
import os
import json


def get_available_evaluation_methods() -> Dict[str, str]:
    """
    Get a dictionary of available evaluation methods with descriptions.
    
    Returns:
        Dictionary mapping method names to descriptions
    """
    return {
        'max_similarity': 'Hit if maximum similarity score >= threshold (recommended)',
        'avg_similarity': 'Hit if average similarity score >= threshold',
        'top_k_relevance': 'Hit if at least one retrieved document has similarity >= threshold',
        'composite': 'Hit if (max_similarity >= threshold) AND (documents retrieved)',
        'strict': 'Hit if max_similarity >= 0.7 (high relevance requirement)',
        'lenient': 'Hit if max_similarity >= 0.3 (low relevance requirement)'
    }


def calculate_hit_rate(metrics: List[Dict], method: str = 'max_similarity', threshold: float = 0.5) -> float:
    """
    Calculate hit rate using different evaluation standards.
    
    Args:
        metrics: List of metric dictionaries
        method: Evaluation method to use:
            - 'max_similarity': Hit if max_similarity >= threshold (default)
            - 'avg_similarity': Hit if avg_similarity >= threshold
            - 'top_k_relevance': Hit if at least one doc has similarity >= threshold
            - 'composite': Hit if (max_similarity >= threshold) AND (num_docs > 0)
            - 'strict': Hit if max_similarity >= 0.7 (high relevance)
            - 'lenient': Hit if max_similarity >= 0.3 (low relevance)
        threshold: Similarity threshold (default: 0.5)
    
    Returns:
        Hit rate as a float between 0 and 1
    """
    if not metrics:
        return 0.0
    
    hits = 0
    for metric in metrics:
        max_sim = metric.get('max_similarity', 0)
        avg_sim = metric.get('avg_similarity', 0)
        num_docs = metric.get('num_docs', 0)
        
        if method == 'max_similarity':
            # Hit if maximum similarity exceeds threshold
            hits += 1 if max_sim >= threshold else 0
        elif method == 'avg_similarity':
            # Hit if average similarity exceeds threshold
            hits += 1 if avg_sim >= threshold else 0
        elif method == 'top_k_relevance':
            # Hit if at least one document is relevant (same as max_similarity but clearer intent)
            hits += 1 if max_sim >= threshold else 0
        elif method == 'composite':
            # Hit if both max similarity is good AND documents were retrieved
            hits += 1 if (max_sim >= threshold and num_docs > 0) else 0
        elif method == 'strict':
            # High relevance threshold (0.7)
            hits += 1 if max_sim >= 0.7 else 0
        elif method == 'lenient':
            # Low relevance threshold (0.3)
            hits += 1 if max_sim >= 0.3 else 0
        else:
            # Default to max_similarity
            hits += 1 if max_sim >= threshold else 0
    
    return hits / len(metrics)


def _load_ragas_results(ragas_results_path: str = "eval_results.json") -> Optional[Dict]:
    """
    Attempt to load Ragas evaluation results from disk.
    Returns None if file does not exist or is invalid.
    """
    if not os.path.exists(ragas_results_path):
        return None
    try:
        with open(ragas_results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Basic validation: expect 'aggregate' key
        if "aggregate" in data:
            return data
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def generate_evaluation_dashboard(metrics: List[Dict], hit_rate_method: str = 'max_similarity', 
                                  hit_rate_threshold: float = 0.5,
                                  ragas_results_path: str = "eval_results.json"):
    """
    Generate the evaluation dashboard with response time tracking.
    
    When Ragas evaluation results are available (eval_results.json), a 5th panel
    is added showing context_precision, context_recall, faithfulness, and
    answer_relevancy scores.
    
    Args:
        metrics: List of metric dictionaries
        hit_rate_method: Method for calculating hit rate ('max_similarity', 'avg_similarity', 
                        'top_k_relevance', 'composite', 'strict', 'lenient')
        hit_rate_threshold: Similarity threshold for hit rate calculation (default: 0.5)
        ragas_results_path: Path to Ragas evaluation results JSON (default: eval_results.json)
    """
    if not metrics:
        print("\n⚠️ No metrics to evaluate")
        return

    print("\n📊 Generating Evaluation Dashboard...")
    print(f"📋 Using hit rate method: {hit_rate_method} (threshold: {hit_rate_threshold})")

    # Check for Ragas results
    ragas_data = _load_ragas_results(ragas_results_path)
    has_ragas = ragas_data is not None
    if has_ragas:
        print("📊 Ragas evaluation results detected – adding Ragas metrics panel")

    # Prepare data
    df = pd.DataFrame(metrics)

    # Calculate hit rate using the specified method
    response_rate = calculate_hit_rate(metrics, method=hit_rate_method, threshold=hit_rate_threshold)

    # Create figure – use 3 rows when Ragas data is present
    if has_ragas:
        fig = plt.figure(figsize=(16, 15))
        fig.patch.set_facecolor('#b8c9d9')
        fig.suptitle('SFU Chatbot RAG Evaluation Results', fontsize=18, fontweight='bold', y=0.98)
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25, left=0.08, right=0.95, top=0.94, bottom=0.05)
    else:
        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor('#b8c9d9')
        fig.suptitle('SFU Chatbot RAG Evaluation Results', fontsize=18, fontweight='bold', y=0.98)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25, left=0.08, right=0.95, top=0.93, bottom=0.07)

    # ==================== TOP LEFT: Retrieval Performance Metrics ====================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#e8f0f7')
    avg_sim = df['avg_similarity'].mean()
    max_sim = df['max_similarity'].mean()
    min_sim = df['min_similarity'].mean()

    metrics_data = [response_rate, avg_sim, max_sim, min_sim]
    metrics_labels = ['Hit Rate', 'Avg Similarity', 'Max Similarity', 'Min Similarity']
    colors = ['#5dade2', '#af7ac5', '#f39c12', '#e74c3c']

    bars = ax1.bar(metrics_labels, metrics_data, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, value in zip(bars, metrics_data):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax1.set_title('Retrieval Performance Metrics', fontsize=13, fontweight='bold', pad=10)
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3, color='white', linewidth=1.5)
    ax1.tick_params(axis='x', rotation=15)

    # ==================== TOP RIGHT: Average Similarity by Query ====================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#e8f0f7')

    query_nums = range(1, len(df) + 1)
    bars = ax2.bar(query_nums, df['avg_similarity'], color='#82c77e', edgecolor='black', linewidth=1.5)

    # Add threshold line (use the hit rate threshold)
    threshold = hit_rate_threshold
    ax2.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Relevance Threshold ({threshold})')

    ax2.set_title('Average Similarity by Query', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel('Query Number', fontsize=11)
    ax2.set_ylabel('Similarity Score', fontsize=11)
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', alpha=0.3, color='white', linewidth=1.5)

    # ==================== BOTTOM LEFT: Performance by Query Category ====================
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#e8f0f7')

    category_performance = df.groupby('category')['avg_similarity'].mean().sort_values(ascending=True)

    bars = ax3.barh(category_performance.index, category_performance.values,
                    color='#5dade2', edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, (cat, val) in enumerate(category_performance.items()):
        ax3.text(val + 0.01, i, f'{val:.3f}', va='center', fontweight='bold', fontsize=10)

    # Add evaluation summary text box
    avg_response_time = df['response_time'].mean()
    summary_text = f"""Evaluation Summary:
• Total Queries: {len(df)}
• Hit Rate ({hit_rate_method}): {response_rate*100:.1f}%
• Strong Similarity: {avg_sim:.3f}
• Avg Response Time: {avg_response_time:.2f}s
• Threshold: {hit_rate_threshold}"""

    ax3.text(0.02, 0.15, summary_text, transform=ax3.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add test queries annotation
    test_queries = df['category'].value_counts()
    queries_text = "Test Queries Included:\n" + "\n".join([f"• {cat}" for cat in test_queries.index[:5]])
    ax3.text(0.02, 0.95, queries_text, transform=ax3.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    ax3.set_title('Performance by Query Category', fontsize=13, fontweight='bold', pad=10)
    ax3.set_xlabel('Average Similarity Score', fontsize=11)
    ax3.set_xlim(0, max(category_performance.values) * 1.15)
    ax3.grid(axis='x', alpha=0.3, color='white', linewidth=1.5)

    # ==================== BOTTOM RIGHT: Response Time for Each Query ====================
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#e8f0f7')

    query_nums = range(1, len(df) + 1)

    # Plot response times with color gradient based on speed
    colors_gradient = []
    for rt in df['response_time']:
        if rt < 2.0:
            colors_gradient.append('#82c77e')  # Green for fast
        elif rt < 5.0:
            colors_gradient.append('#f39c12')  # Orange for medium
        else:
            colors_gradient.append('#e74c3c')  # Red for slow

    bars = ax4.bar(query_nums, df['response_time'], color=colors_gradient,
                   edgecolor='black', linewidth=1.5)

    # Add average line
    avg_time = df['response_time'].mean()
    ax4.axhline(y=avg_time, color='blue', linestyle='--', linewidth=2,
                label=f'Avg: {avg_time:.2f}s')

    # Add performance thresholds
    ax4.axhline(y=2.0, color='green', linestyle=':', linewidth=1.5, alpha=0.5,
                label='Fast (<2s)')
    ax4.axhline(y=5.0, color='orange', linestyle=':', linewidth=1.5, alpha=0.5,
                label='Medium (<5s)')

    ax4.set_title('Response Time for Each Query', fontsize=13, fontweight='bold', pad=10)
    ax4.set_xlabel('Query Number', fontsize=11)
    ax4.set_ylabel('Response Time (seconds)', fontsize=11)
    ax4.set_ylim(0, max(df['response_time']) * 1.2)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(axis='y', alpha=0.3, color='white', linewidth=1.5)

    # Add performance statistics annotation
    fast_queries = sum(1 for rt in df['response_time'] if rt < 2.0)
    medium_queries = sum(1 for rt in df['response_time'] if 2.0 <= rt < 5.0)
    slow_queries = sum(1 for rt in df['response_time'] if rt >= 5.0)

    stats_text = f"""Performance Distribution:
Fast (<2s): {fast_queries} queries
Medium (2-5s): {medium_queries} queries
Slow (≥5s): {slow_queries} queries

Min: {df['response_time'].min():.2f}s
Max: {df['response_time'].max():.2f}s
Median: {df['response_time'].median():.2f}s"""

    ax4.text(0.98, 0.97, stats_text, transform=ax4.transAxes,
            fontsize=8, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    # ==================== ROW 3: Ragas Evaluation Metrics (if available) ====================
    if has_ragas:
        aggregate = ragas_data.get("aggregate", {})
        per_question = ragas_data.get("per_question", [])

        # --- Left panel: Ragas aggregate scores bar chart ---
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.set_facecolor('#e8f0f7')

        ragas_metric_names = list(aggregate.keys())
        ragas_metric_values = [aggregate[k] for k in ragas_metric_names]
        ragas_colors = ['#2ecc71', '#3498db', '#9b59b6', '#e67e22']
        # Extend colours if there are more metrics
        while len(ragas_colors) < len(ragas_metric_names):
            ragas_colors.append('#95a5a6')

        ragas_bars = ax5.bar(
            ragas_metric_names,
            ragas_metric_values,
            color=ragas_colors[:len(ragas_metric_names)],
            edgecolor='black',
            linewidth=1.5,
        )
        for bar, value in zip(ragas_bars, ragas_metric_values):
            height = bar.get_height()
            ax5.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f'{value:.4f}',
                ha='center',
                va='bottom',
                fontweight='bold',
                fontsize=10,
            )

        ax5.set_title('Ragas Evaluation Metrics (Aggregate)', fontsize=13, fontweight='bold', pad=10)
        ax5.set_ylabel('Score', fontsize=11)
        ax5.set_ylim(0, 1.1)
        ax5.grid(axis='y', alpha=0.3, color='white', linewidth=1.5)
        ax5.tick_params(axis='x', rotation=15)

        # --- Right panel: per-question Ragas score distribution ---
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.set_facecolor('#e8f0f7')

        if per_question:
            pq_df = pd.DataFrame(per_question)
            # Pick numeric metric columns
            score_cols = [c for c in pq_df.columns if c in aggregate]
            if score_cols:
                pq_df_scores = pq_df[score_cols].apply(pd.to_numeric, errors='coerce')
                bp = ax6.boxplot(
                    [pq_df_scores[col].dropna().values for col in score_cols],
                    labels=score_cols,
                    patch_artist=True,
                    medianprops=dict(color='black', linewidth=2),
                )
                for patch, color in zip(bp['boxes'], ragas_colors[:len(score_cols)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)

        # Add summary text
        ragas_summary_lines = [f"Ragas Results ({len(per_question)} questions):"]
        for name, val in aggregate.items():
            ragas_summary_lines.append(f"  {name}: {val:.4f}")
        ragas_summary = "\n".join(ragas_summary_lines)
        ax6.text(
            0.98, 0.97, ragas_summary, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
        )

        ax6.set_title('Ragas Score Distribution (Per Question)', fontsize=13, fontweight='bold', pad=10)
        ax6.set_ylabel('Score', fontsize=11)
        ax6.set_ylim(0, 1.1)
        ax6.grid(axis='y', alpha=0.3, color='white', linewidth=1.5)

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"rag_evaluation_{timestamp}.png"
    plt.savefig(filename, facecolor=fig.get_facecolor(), dpi=150)
    print(f"📈 Evaluation dashboard saved to: {filename}")

    # Print performance summary to console
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Total Queries Processed: {len(df)}")
    print(f"Hit Rate Method: {hit_rate_method}")
    print(f"Hit Rate Threshold: {hit_rate_threshold}")
    print(f"Hit Rate: {response_rate*100:.1f}%")
    print(f"\nSimilarity Metrics:")
    print(f"  Average Similarity: {avg_sim:.3f}")
    print(f"  Max Similarity (avg): {max_sim:.3f}")
    print(f"  Min Similarity (avg): {min_sim:.3f}")
    print(f"\nResponse Time Metrics:")
    print(f"  Average Response Time: {avg_time:.3f}s")
    print(f"  Fastest Response: {df['response_time'].min():.3f}s")
    print(f"  Slowest Response: {df['response_time'].max():.3f}s")
    print(f"  Median Response Time: {df['response_time'].median():.3f}s")
    print(f"\nPerformance Distribution:")
    print(f"  Fast Queries (<2s): {fast_queries} ({fast_queries/len(df)*100:.1f}%)")
    print(f"  Medium Queries (2-5s): {medium_queries} ({medium_queries/len(df)*100:.1f}%)")
    print(f"  Slow Queries (≥5s): {slow_queries} ({slow_queries/len(df)*100:.1f}%)")

    if has_ragas:
        print(f"\nRagas Evaluation Metrics:")
        for name, val in ragas_data.get("aggregate", {}).items():
            print(f"  {name}: {val:.4f}")
        print(f"  Questions Evaluated: {len(ragas_data.get('per_question', []))}")

    print("="*60)

    plt.show()

