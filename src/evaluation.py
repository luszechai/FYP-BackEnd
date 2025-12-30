"""Evaluation dashboard generation module"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime


def generate_evaluation_dashboard(metrics: List[Dict]):
    """Generate the 4-panel evaluation dashboard with response time tracking"""
    if not metrics:
        print("\n⚠️ No metrics to evaluate")
        return

    print("\n📊 Generating Evaluation Dashboard...")

    # Prepare data
    df = pd.DataFrame(metrics)

    # Create figure with grey background
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#b8c9d9')

    # Main title
    fig.suptitle('SFU Chatbot RAG Evaluation Results', fontsize=18, fontweight='bold', y=0.98)

    # Create 2x2 grid with specific spacing
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25, left=0.08, right=0.95, top=0.93, bottom=0.07)

    # ==================== TOP LEFT: Retrieval Performance Metrics ====================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#e8f0f7')

    response_rate = df['hit'].sum() / len(df)
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

    # Add threshold line
    threshold = 0.5
    ax2.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label='Relevance Threshold')

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
• Perfect Hit Rate: {response_rate*100:.1f}%
• Strong Similarity: {avg_sim:.3f}
• Avg Response Time: {avg_response_time:.2f}s"""

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
    print(f"Average Response Time: {avg_time:.3f}s")
    print(f"Fastest Response: {df['response_time'].min():.3f}s")
    print(f"Slowest Response: {df['response_time'].max():.3f}s")
    print(f"Median Response Time: {df['response_time'].median():.3f}s")
    print(f"\nPerformance Distribution:")
    print(f"  Fast Queries (<2s): {fast_queries} ({fast_queries/len(df)*100:.1f}%)")
    print(f"  Medium Queries (2-5s): {medium_queries} ({medium_queries/len(df)*100:.1f}%)")
    print(f"  Slow Queries (≥5s): {slow_queries} ({slow_queries/len(df)*100:.1f}%)")
    print("="*60)

    plt.show()

