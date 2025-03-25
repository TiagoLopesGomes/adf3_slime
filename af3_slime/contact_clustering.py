from typing import Dict, List, Tuple, NamedTuple
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import logging

@dataclass
class ClusterAnalysisResults:
    positions: List[int]
    features: np.ndarray
    scaled_features: np.ndarray
    dbscan_clusters: np.ndarray
    linkage_matrix: np.ndarray
    feature_names: List[str]
    cluster_stats: Dict[int, Dict]

def create_feature_vectors(contact_probs: Dict[int, List[float]], 
                         threshold: float = 0.3,
                         window_size: int = 3) -> Tuple[np.ndarray, List[str]]:
    """Create feature vectors for each position"""
    positions = sorted(contact_probs.keys())
    features = []
    
    for pos in positions:
        values = np.array(contact_probs[pos])
        
        # Get local neighborhood for density calculation
        neighborhood = []
        for offset in range(-window_size, window_size + 1):
            if pos + offset in contact_probs:
                neighborhood.extend(contact_probs[pos + offset])
        neighborhood = np.array(neighborhood)
        
        feature_vector = [
            np.max(values),                              # Maximum contact
            np.mean(values),                             # Mean contact
            np.sum(values > threshold) / len(values),    # Frequency of strong contacts
            np.std(values),                             # Variability
            np.percentile(values, 75),                  # Upper quartile
            np.mean(neighborhood),                      # Local density
            np.sum(neighborhood > threshold) / len(neighborhood),  # Local strong contacts
            np.median(values),                          # Median contact
            np.percentile(values, 90)                   # 90th percentile
        ]
        features.append(feature_vector)
    
    feature_names = [
        'Max', 'Mean', 'Frequency', 'Std', 'Q75',
        'LocalDensity', 'LocalStrong', 'Median', 'P90'
    ]
    
    return np.array(features), feature_names

def analyze_clusters(positions: List[int], 
                    clusters: np.ndarray, 
                    features: np.ndarray) -> Dict[int, Dict]:
    """Analyze properties of each cluster"""
    cluster_stats = {}
    
    for cluster_id in set(clusters):
        mask = clusters == cluster_id
        cluster_positions = np.array(positions)[mask]
        cluster_features = features[mask]
        
        # Skip noise points (cluster_id = -1) if empty
        if len(cluster_positions) == 0:
            continue
            
        # Calculate cluster statistics
        stats = {
            'positions': cluster_positions.tolist(),
            'size': len(cluster_positions),
            'mean_contact': np.mean(cluster_features[:, 1]),  # Mean of mean contacts
            'max_contact': np.max(cluster_features[:, 0]),    # Max of max contacts
            'is_continuous': all(b-a == 1 for a, b in zip(sorted(cluster_positions)[:-1], 
                                                         sorted(cluster_positions)[1:])),
            'span': max(cluster_positions) - min(cluster_positions) + 1
        }
        
        cluster_stats[cluster_id] = stats
    
    return cluster_stats

def analyze_contact_patterns(contact_probs: Dict[int, List[float]], 
                           threshold: float = 0.3,
                           eps: float = 0.3,
                           min_samples: int = 3) -> ClusterAnalysisResults:
    """Perform complete clustering analysis of contact patterns"""
    logger = logging.getLogger(__name__)
    
    # Create feature vectors
    features, feature_names = create_feature_vectors(contact_probs, threshold)
    positions = sorted(contact_probs.keys())
    
    logger.debug(f"Analyzing contact patterns for {len(positions)} positions")
    logger.debug(f"Position range: {min(positions)} to {max(positions)}")
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Perform clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_clusters = dbscan.fit_predict(scaled_features)
    
    # Log cluster information
    unique_clusters = np.unique(dbscan_clusters)
    logger.debug(f"DBSCAN found {len(unique_clusters)} unique clusters (including noise)")
    logger.debug(f"Cluster IDs: {unique_clusters}")
    
    # Count positions in each cluster
    for cluster_id in unique_clusters:
        count = np.sum(dbscan_clusters == cluster_id)
        if cluster_id == -1:
            logger.debug(f"Noise cluster: {count} positions")
        else:
            logger.debug(f"Cluster {cluster_id}: {count} positions")
    
    # Calculate hierarchical clustering
    linkage_matrix = linkage(scaled_features, method='ward')
    
    # Analyze clusters
    cluster_stats = analyze_clusters(positions, dbscan_clusters, features)
    
    return ClusterAnalysisResults(
        positions=positions,
        features=features,
        scaled_features=scaled_features,
        dbscan_clusters=dbscan_clusters,
        linkage_matrix=linkage_matrix,
        feature_names=feature_names,
        cluster_stats=cluster_stats
    )
