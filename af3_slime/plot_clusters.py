from typing import Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Change to absolute import
from contact_clustering import ClusterAnalysisResults
from scipy.cluster.hierarchy import dendrogram


def plot_feature_heatmap(results: ClusterAnalysisResults, output_path: str):
    """Plot heatmap of feature vectors"""
    plt.figure(figsize=(12, 8))

    # Create heatmap
    sns.heatmap(
        results.features,
        xticklabels=results.feature_names,
        yticklabels=results.positions,
        cmap="YlOrRd",
    )

    # Modify y-axis ticks to show every 10th position
    ax = plt.gca()
    positions = results.positions
    tick_positions = range(0, len(positions), 10)  # Every 10th index
    tick_labels = [positions[i] for i in tick_positions]  # Get position numbers
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    plt.title("Contact Features by Position")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_cluster_assignments(results: ClusterAnalysisResults, output_path: str):
    """Plot cluster assignments along sequence"""
    plt.figure(figsize=(12, 6))

    # Plot mean contact probability for all positions
    plt.plot(
        results.positions,
        results.features[:, 1],  # Mean contact probability
        "gray",
        alpha=0.3,
        label="Mean Contact",
    )

    # Plot clusters with different colors
    unique_clusters = set(results.dbscan_clusters)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))

    for cluster_id, color in zip(unique_clusters, colors):
        if cluster_id == -1:  # Noise points
            continue

        mask = results.dbscan_clusters == cluster_id
        plt.scatter(
            np.array(results.positions)[mask],
            results.features[mask, 1],
            c=[color],
            label=f"Cluster {cluster_id}",
        )

    plt.xlabel("Position")
    plt.ylabel("Mean Contact Probability")
    plt.title("Contact Clusters by Position")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_dendrogram(results: ClusterAnalysisResults, output_path: str):
    """Plot hierarchical clustering dendrogram"""
    plt.figure(figsize=(12, 8))
    dendrogram(results.linkage_matrix)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Position Index")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_cluster_summary(results: ClusterAnalysisResults, output_path: str):
    """Plot comprehensive summary of clustering results"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), height_ratios=[2, 1])

    # Upper plot: Contact patterns and clusters
    ax1.plot(
        results.positions, results.features[:, 1], "b-", alpha=0.5, label="Mean Contact"
    )
    ax1.plot(
        results.positions, results.features[:, 0], "r--", alpha=0.5, label="Max Contact"
    )

    # Plot cluster regions
    ymin, ymax = ax1.get_ylim()
    for cluster_id, stats in results.cluster_stats.items():
        if cluster_id == -1:  # Skip noise
            continue

        positions = stats["positions"]
        ax1.axvspan(
            min(positions),
            max(positions),
            alpha=0.2,
            color=plt.cm.Set1(cluster_id / len(results.cluster_stats)),
        )

        # Annotate cluster properties
        ax1.text(
            np.mean(positions),
            ymax * 0.9,
            f"C{cluster_id}\n{stats['mean_contact']:.2f}",
            horizontalalignment="center",
        )

    ax1.set_xlabel("Position")
    ax1.set_ylabel("Contact Probability")
    ax1.set_title("Contact Patterns and Clusters")
    ax1.legend()
    ax1.grid(True)

    # Lower plot: Feature importance
    feature_means = np.mean(results.features, axis=0)
    feature_stds = np.std(results.features, axis=0)

    ax2.bar(results.feature_names, feature_means, yerr=feature_stds)
    ax2.set_xlabel("Features")
    ax2.set_ylabel("Mean Value")
    ax2.set_title("Feature Importance")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def visualize_contact_clusters(results: ClusterAnalysisResults, output_prefix: Union[str, Dict[str, str]]):
    """
    Generate all visualizations
    
    Args:
        results: Cluster analysis results
        output_prefix: Either a string prefix for all output files or a dictionary mapping
                      output types to file paths
    """
    # Handle dictionary of output paths
    if isinstance(output_prefix, dict):
        # Use provided paths or generate defaults
        heatmap_path = output_prefix.get('heatmap', 'cluster_features_heatmap.png')
        assignments_path = output_prefix.get('assignments', 'cluster_assignments.png')
        dendrogram_path = output_prefix.get('dendrogram', 'cluster_dendrogram.png')
        summary_path = output_prefix.get('summary', 'cluster_summary.png')
        
        # Generate cluster statistics JSON if path provided
        if 'stats' in output_prefix:
            stats_path = output_prefix['stats']
            save_cluster_stats(results, stats_path)
    else:
        # Use string prefix for all outputs
        heatmap_path = f"{output_prefix}_features_heatmap.png"
        assignments_path = f"{output_prefix}_cluster_assignments.png"
        dendrogram_path = f"{output_prefix}_dendrogram.png"
        summary_path = f"{output_prefix}_summary.png"
    
    # Generate visualizations
    plot_feature_heatmap(results, heatmap_path)
    plot_cluster_assignments(results, assignments_path)
    plot_dendrogram(results, dendrogram_path)
    plot_cluster_summary(results, summary_path)


def save_cluster_stats(results: ClusterAnalysisResults, output_path: str):
    """Save cluster statistics to JSON file"""
    import json
    
    # Convert cluster stats to serializable format
    serializable_stats = {}
    for cluster_id, stats in results.cluster_stats.items():
        serializable_stats[str(cluster_id)] = {
            'positions': stats['positions'],
            'size': stats['size'],
            'mean_contact': stats['mean_contact'],
            'max_contact': stats['max_contact'],
            'is_continuous': stats['is_continuous'],
            'span': stats['span']
        }
    
    # Add overall statistics
    serializable_stats['overall'] = {
        'num_clusters': len([k for k in results.cluster_stats.keys() if k != -1]),
        'noise_points': len([p for c, p in zip(results.dbscan_clusters, results.positions) if c == -1]),
        'total_positions': len(results.positions)
    }
    
    # Write to file
    with open(output_path, 'w') as f:
        json.dump(serializable_stats, f, indent=2)
