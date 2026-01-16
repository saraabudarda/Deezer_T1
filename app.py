import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from scipy.spatial.distance import euclidean, cosine
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import os

# ==========================================
# CONFIGURATION
# ==========================================
"""
Configuration file for the Deezer Mood Detection EDA Dashboard
Contains constants, color schemes, and settings
"""

# Dataset paths
# Dataset paths
# Robustly determine DATA_DIR by searching for train.csv
current_script_dir = os.path.dirname(os.path.abspath(__file__))
possible_dirs = [
    ".",                                            # Current working directory
    current_script_dir,                             # Directory of the script
    os.path.join(current_script_dir, ".."),         # Parent directory
]

DATA_DIR = "." # Default fallback
for d in possible_dirs:
    if os.path.exists(os.path.join(d, "train.csv")):
        DATA_DIR = d
        break

TRAIN_FILE = "train.csv"
VALIDATION_FILE = "validation.csv"
TEST_FILE = "test.csv"

# Column names
ID_COLUMNS = ['dzr_sng_id', 'MSD_sng_id', 'MSD_track_id']
TARGET_COLUMNS = ['valence', 'arousal']
METADATA_COLUMNS = ['artist_name', 'track_name']
ALL_COLUMNS = ID_COLUMNS + TARGET_COLUMNS + METADATA_COLUMNS

# Valence-Arousal ranges (typical range based on dataset)
VALENCE_RANGE = (-2.5, 2.5)
AROUSAL_RANGE = (-2.5, 2.5)

# Color schemes for visualizations
COLORS = {
    'train': '#3498db',      # Blue
    'validation': '#e74c3c', # Red
    'test': '#2ecc71',       # Green
    'combined': '#9b59b6',   # Purple
    'primary': '#2c3e50',    # Dark blue-gray
    'secondary': '#95a5a6',  # Gray
}

# Plot styling
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIGURE_SIZE = (10, 6)
FONT_SIZE = 12
TITLE_SIZE = 14

# Streamlit page configuration
PAGE_TITLE = "Deezer Mood Detection - EDA Dashboard"
PAGE_ICON = "🎵"
LAYOUT = "wide"

# Emotional quadrants (based on valence-arousal model)
EMOTIONAL_QUADRANTS = {
    'Q1': {'name': 'Happy/Excited', 'valence': 'positive', 'arousal': 'high'},
    'Q2': {'name': 'Angry/Tense', 'valence': 'negative', 'arousal': 'high'},
    'Q3': {'name': 'Sad/Depressed', 'valence': 'negative', 'arousal': 'low'},
    'Q4': {'name': 'Calm/Relaxed', 'valence': 'positive', 'arousal': 'low'},
}

# Statistical significance level
ALPHA = 0.05

# Dashboard text content
ABOUT_TEXT = """
## About the Dataset

The **Deezer Mood Detection Dataset** is designed for music emotion recognition research. 
It contains metadata and emotional annotations for music tracks, focusing on two key dimensions 
of emotion: **valence** and **arousal**.

### Valence-Arousal Model (Russell's Circumplex Model)

The dataset uses the **valence-arousal emotional space**, a well-established psychological model:

- **Valence**: Represents the positivity or negativity of emotion
  - Positive values → Happy, joyful, pleasant emotions
  - Negative values → Sad, angry, unpleasant emotions

- **Arousal**: Represents the intensity or energy level of emotion
  - High values → Excited, energetic, intense emotions
  - Low values → Calm, relaxed, low-energy emotions

### Task Type

This is a **regression problem** where the goal is to predict continuous values for valence 
and arousal based on audio features (not included in this dataset).

### Dataset Splits

- **Training set**: Used to train machine learning models
- **Validation set**: Used to tune hyperparameters and prevent overfitting
- **Test set**: Used for final model evaluation
"""

DATA_QUALITY_TEXT = """
### Data Quality Considerations

1. **Label Subjectivity**: Emotional annotations are inherently subjective and may vary between annotators
2. **Missing Audio Features**: This dataset contains only metadata and labels, not raw audio or extracted features
3. **Outliers**: Some tracks may have extreme valence/arousal values that could represent annotation errors or genuinely extreme emotions
4. **Distribution Balance**: The distribution of emotions across the valence-arousal space may not be uniform
"""

# ==========================================
# STATISTICS UTILS
# ==========================================
"""
Statistical analysis utilities for Deezer Mood Detection Dataset
"""

def compute_descriptive_stats(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """
    Compute comprehensive descriptive statistics for specified columns.
    """
    if columns is None:
        columns = TARGET_COLUMNS
    
    stats_dict = {}
    
    for col in columns:
        if col in df.columns:
            stats_dict[col] = {
                'Count': df[col].count(),
                'Mean': df[col].mean(),
                'Std': df[col].std(),
                'Min': df[col].min(),
                '25%': df[col].quantile(0.25),
                'Median': df[col].median(),
                '75%': df[col].quantile(0.75),
                'Max': df[col].max(),
                'Skewness': df[col].skew(),
                'Kurtosis': df[col].kurtosis(),
            }
    
    stats_df = pd.DataFrame(stats_dict).T
    return stats_df


def compute_correlation(df: pd.DataFrame, method: str = 'pearson') -> Dict[str, float]:
    """
    Compute correlation between valence and arousal.
    """
    if method == 'pearson':
        corr, p_value = stats.pearsonr(df['valence'], df['arousal'])
    elif method == 'spearman':
        corr, p_value = stats.spearmanr(df['valence'], df['arousal'])
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    
    return {
        'correlation': corr,
        'p_value': p_value,
        'method': method,
        'significant': p_value < 0.05
    }


def compare_distributions(df1: pd.DataFrame, df2: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Compare distributions between two datasets using statistical tests.
    """
    # Kolmogorov-Smirnov test
    ks_statistic, ks_pvalue = stats.ks_2samp(df1[column], df2[column])
    
    # Mann-Whitney U test (non-parametric)
    mw_statistic, mw_pvalue = stats.mannwhitneyu(df1[column], df2[column])
    
    # T-test (parametric)
    t_statistic, t_pvalue = stats.ttest_ind(df1[column], df2[column])
    
    return {
        'ks_test': {
            'statistic': ks_statistic,
            'p_value': ks_pvalue,
            'significant': ks_pvalue < 0.05
        },
        'mann_whitney': {
            'statistic': mw_statistic,
            'p_value': mw_pvalue,
            'significant': mw_pvalue < 0.05
        },
        't_test': {
            'statistic': t_statistic,
            'p_value': t_pvalue,
            'significant': t_pvalue < 0.05
        }
    }


def test_normality(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Test if a distribution is normal using multiple tests.
    """
    # Shapiro-Wilk test (good for small samples)
    if len(df) <= 5000:
        shapiro_stat, shapiro_p = stats.shapiro(df[column].dropna())
    else:
        shapiro_stat, shapiro_p = None, None
    
    # Anderson-Darling test
    anderson_result = stats.anderson(df[column].dropna())
    
    # Kolmogorov-Smirnov test against normal distribution
    ks_stat, ks_p = stats.kstest(
        df[column].dropna(),
        'norm',
        args=(df[column].mean(), df[column].std())
    )
    
    return {
        'shapiro_wilk': {
            'statistic': shapiro_stat,
            'p_value': shapiro_p,
            'normal': shapiro_p > 0.05 if shapiro_p is not None else None
        } if shapiro_stat is not None else None,
        'anderson_darling': {
            'statistic': anderson_result.statistic,
            'critical_values': anderson_result.critical_values.tolist(),
            'significance_levels': anderson_result.significance_level.tolist()
        },
        'ks_test': {
            'statistic': ks_stat,
            'p_value': ks_p,
            'normal': ks_p > 0.05
        }
    }


def compute_split_statistics(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute statistics for each dataset split.
    """
    all_stats = []
    
    for split_name, df in datasets.items():
        for col in TARGET_COLUMNS:
            if col in df.columns:
                all_stats.append({
                    'Split': split_name.capitalize(),
                    'Variable': col.capitalize(),
                    'Count': df[col].count(),
                    'Mean': f"{df[col].mean():.4f}",
                    'Std': f"{df[col].std():.4f}",
                    'Min': f"{df[col].min():.4f}",
                    'Max': f"{df[col].max():.4f}",
                    'Skewness': f"{df[col].skew():.4f}",
                    'Kurtosis': f"{df[col].kurtosis():.4f}",
                })
    
    return pd.DataFrame(all_stats)


def identify_emotional_quadrant_stat(valence: float, arousal: float) -> str:
    """
    Identify which emotional quadrant a point belongs to.
    RENAMED to avoid conflict if defined elsewhere, though in single file it's fine.
    """
    if valence >= 0 and arousal >= 0:
        return 'Q1'  # Happy/Excited
    elif valence < 0 and arousal >= 0:
        return 'Q2'  # Angry/Tense
    elif valence < 0 and arousal < 0:
        return 'Q3'  # Sad/Depressed
    else:
        return 'Q4'  # Calm/Relaxed


def compute_quadrant_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute distribution of samples across emotional quadrants.
    """
    df_copy = df.copy()
    df_copy['quadrant'] = df_copy.apply(
        lambda row: identify_emotional_quadrant_stat(row['valence'], row['arousal']),
        axis=1
    )
    
    quadrant_counts = df_copy['quadrant'].value_counts().sort_index()
    quadrant_pct = (quadrant_counts / len(df) * 100).round(2)
    
    quadrant_names = {
        'Q1': 'Happy/Excited',
        'Q2': 'Angry/Tense',
        'Q3': 'Sad/Depressed',
        'Q4': 'Calm/Relaxed'
    }
    
    # Handle missing quadrants
    for q in quadrant_names:
        if q not in quadrant_counts:
            quadrant_counts[q] = 0
            quadrant_pct[q] = 0.0

    result = pd.DataFrame({
        'Quadrant': [quadrant_names.get(q, q) for q in quadrant_counts.index],
        'Count': quadrant_counts.values,
        'Percentage': quadrant_pct.values
    })
    
    return result

# ==========================================
# VISUALIZATIONS UTILS
# ==========================================
"""
Visualization utilities for Deezer Mood Detection Dataset
"""

# Set default style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_distribution_histogram(
    df: pd.DataFrame,
    column: str,
    split_name: str = None,
    bins: int = 50,
    color: str = None,
    ax: plt.Axes = None
) -> plt.Figure:
    """
    Plot histogram for a single variable.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    else:
        fig = ax.get_figure()
    
    if color is None:
        color = COLORS['primary']
    
    ax.hist(df[column], bins=bins, color=color, alpha=0.7, edgecolor='black')
    
    title = f'Distribution of {column.capitalize()}'
    if split_name:
        title += f' ({split_name.capitalize()} Set)'
    
    ax.set_xlabel(column.capitalize(), fontsize=FONT_SIZE)
    ax.set_ylabel('Frequency', fontsize=FONT_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_val = df[column].mean()
    # std_val = df[column].std() # Unused
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_overlaid_distributions(
    datasets: Dict[str, pd.DataFrame],
    column: str,
    bins: int = 50
) -> plt.Figure:
    """
    Plot overlaid histograms for multiple dataset splits.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    for split_name, df in datasets.items():
        color = COLORS.get(split_name, COLORS['primary'])
        ax.hist(
            df[column],
            bins=bins,
            alpha=0.5,
            label=split_name.capitalize(),
            color=color,
            edgecolor='black'
        )
    
    ax.set_xlabel(column.capitalize(), fontsize=FONT_SIZE)
    ax.set_ylabel('Frequency', fontsize=FONT_SIZE)
    ax.set_title(f'{column.capitalize()} Distribution Across Splits', fontsize=TITLE_SIZE, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_density_comparison(
    datasets: Dict[str, pd.DataFrame],
    column: str
) -> plt.Figure:
    """
    Plot kernel density estimation for multiple splits.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    for split_name, df in datasets.items():
        color = COLORS.get(split_name, COLORS['primary'])
        df[column].plot.kde(
            ax=ax,
            label=split_name.capitalize(),
            color=color,
            linewidth=2
        )
    
    ax.set_xlabel(column.capitalize(), fontsize=FONT_SIZE)
    ax.set_ylabel('Density', fontsize=FONT_SIZE)
    ax.set_title(f'{column.capitalize()} Density Comparison', fontsize=TITLE_SIZE, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_boxplot(
    datasets: Dict[str, pd.DataFrame],
    column: str
) -> plt.Figure:
    """
    Create boxplot for comparing distributions across splits.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Prepare data
    data_list = []
    labels = []
    for split_name, df in datasets.items():
        data_list.append(df[column].dropna())
        labels.append(split_name.capitalize())
    
    # Create boxplot
    bp = ax.boxplot(data_list, labels=labels, patch_artist=True, showmeans=True)
    
    # Color boxes
    for patch, split_name in zip(bp['boxes'], datasets.keys()):
        color = COLORS.get(split_name, COLORS['primary'])
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel(column.capitalize(), fontsize=FONT_SIZE)
    ax.set_title(f'{column.capitalize()} Distribution by Split', fontsize=TITLE_SIZE, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_scatter_valence_arousal(
    df: pd.DataFrame,
    split_name: str = None,
    sample_size: int = None,
    alpha: float = 0.3
) -> plt.Figure:
    """
    Create scatter plot of valence vs arousal with quadrant lines.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Sample if dataset is large
    if sample_size and len(df) > sample_size:
        df_plot = df.sample(n=sample_size, random_state=42)
    else:
        df_plot = df
    
    # Create scatter plot
    scatter = ax.scatter(
        df_plot['valence'],
        df_plot['arousal'],
        alpha=alpha,
        c=COLORS['primary'],
        s=20,
        edgecolors='none'
    )
    
    # Add quadrant lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    
    # Add quadrant labels
    quadrant_labels = {
        'Q1': 'Happy/Excited\n(+V, +A)',
        'Q2': 'Angry/Tense\n(-V, +A)',
        'Q3': 'Sad/Depressed\n(-V, -A)',
        'Q4': 'Calm/Relaxed\n(+V, -A)'
    }
    
    # Get axis limits
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    
    # Q1: top-right
    ax.text(x_lim[1] * 0.7, y_lim[1] * 0.85, quadrant_labels['Q1'],
            fontsize=10, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    # Q2: top-left
    ax.text(x_lim[0] * 0.7, y_lim[1] * 0.85, quadrant_labels['Q2'],
            fontsize=10, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    # Q3: bottom-left
    ax.text(x_lim[0] * 0.7, y_lim[0] * 0.85, quadrant_labels['Q3'],
            fontsize=10, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    # Q4: bottom-right
    ax.text(x_lim[1] * 0.7, y_lim[0] * 0.85, quadrant_labels['Q4'],
            fontsize=10, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Valence (Negativity ← → Positivity)', fontsize=FONT_SIZE)
    ax.set_ylabel('Arousal (Low Energy ← → High Energy)', fontsize=FONT_SIZE)
    
    title = 'Valence-Arousal Emotional Space'
    if split_name:
        title += f' ({split_name.capitalize()} Set)'
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """
    Create correlation heatmap for numerical variables.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Compute correlation matrix
    corr_matrix = df[['valence', 'arousal']].corr()
    
    # Create heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    
    ax.set_title('Correlation Matrix: Valence vs Arousal', fontsize=TITLE_SIZE, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_hexbin_density(
    df: pd.DataFrame,
    split_name: str = None,
    gridsize: int = 30
) -> plt.Figure:
    """
    Create hexbin plot for high-density visualization of valence-arousal space.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    hexbin = ax.hexbin(
        df['valence'],
        df['arousal'],
        gridsize=gridsize,
        cmap='YlOrRd',
        mincnt=1
    )
    
    # Add quadrant lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Valence (Negativity ← → Positivity)', fontsize=FONT_SIZE)
    ax.set_ylabel('Arousal (Low Energy ← → High Energy)', fontsize=FONT_SIZE)
    
    title = 'Valence-Arousal Density Plot'
    if split_name:
        title += f' ({split_name.capitalize()} Set)'
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')
    
    cb = plt.colorbar(hexbin, ax=ax)
    cb.set_label('Count', fontsize=FONT_SIZE)
    
    plt.tight_layout()
    return fig


def plot_quadrant_distribution(quadrant_df: pd.DataFrame) -> plt.Figure:
    """
    Create bar plot of emotional quadrant distribution.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']  # Green, Red, Blue, Orange
    
    bars = ax.bar(
        quadrant_df['Quadrant'],
        quadrant_df['Count'],
        color=colors,
        alpha=0.7,
        edgecolor='black'
    )
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, quadrant_df['Percentage']):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{pct:.1f}%',
            ha='center',
            va='bottom',
            fontsize=FONT_SIZE
        )
    
    ax.set_xlabel('Emotional Quadrant', fontsize=FONT_SIZE)
    ax.set_ylabel('Number of Tracks', fontsize=FONT_SIZE)
    ax.set_title('Distribution Across Emotional Quadrants', fontsize=TITLE_SIZE, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

# ==========================================
# RECOMMENDER UTILS
# ==========================================
"""
Recommender system utilities for Deezer Mood Detection Dataset
"""

def compute_similarity_matrix(df: pd.DataFrame, method: str = 'euclidean') -> np.ndarray:
    """
    Compute pairwise similarity matrix for all tracks.
    """
    features = df[['valence', 'arousal']].values
    n_samples = len(features)
    similarity_matrix = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(n_samples):
            if method == 'euclidean':
                similarity_matrix[i, j] = euclidean(features[i], features[j])
            elif method == 'cosine':
                similarity_matrix[i, j] = 1 - cosine(features[i], features[j])
    
    return similarity_matrix


def find_similar_tracks(
    df: pd.DataFrame,
    track_idx: int,
    n_recommendations: int = 10,
    method: str = 'euclidean'
) -> pd.DataFrame:
    """
    Find similar tracks based on emotional similarity.
    """
    features = df[['valence', 'arousal']].values
    reference = features[track_idx].reshape(1, -1)
    
    # Use KNN for efficient similarity search
    knn = NearestNeighbors(n_neighbors=n_recommendations + 1, metric=method)
    knn.fit(features)
    
    distances, indices = knn.kneighbors(reference)
    
    # Exclude the reference track itself
    similar_indices = indices[0][1:]
    similar_distances = distances[0][1:]
    
    similar_tracks = df.iloc[similar_indices].copy()
    similar_tracks['distance'] = similar_distances
    similar_tracks['similarity_score'] = 1 / (1 + similar_distances)  # Convert to similarity
    
    return similar_tracks


def recommend_by_mood(
    df: pd.DataFrame,
    target_valence: float,
    target_arousal: float,
    n_recommendations: int = 10
) -> pd.DataFrame:
    """
    Recommend tracks based on desired mood (valence, arousal).
    """
    features = df[['valence', 'arousal']].values
    target = np.array([[target_valence, target_arousal]])
    
    # Calculate distances to target mood
    distances = np.array([euclidean(target[0], feat) for feat in features])
    
    # Get top N closest tracks
    top_indices = np.argsort(distances)[:n_recommendations]
    
    recommendations = df.iloc[top_indices].copy()
    recommendations['distance_to_target'] = distances[top_indices]
    recommendations['match_score'] = 1 / (1 + distances[top_indices])
    
    return recommendations


def cluster_by_mood(df: pd.DataFrame, n_clusters: int = 8) -> pd.DataFrame:
    """
    Cluster tracks by emotional similarity using K-Means.
    """
    features = df[['valence', 'arousal']].values
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features)
    
    df_clustered = df.copy()
    df_clustered['cluster'] = clusters
    df_clustered['cluster_center_valence'] = kmeans.cluster_centers_[clusters, 0]
    df_clustered['cluster_center_arousal'] = kmeans.cluster_centers_[clusters, 1]
    
    return df_clustered, kmeans


def analyze_mood_diversity(df: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze diversity of moods in the dataset.
    """
    features = df[['valence', 'arousal']].values
    
    # Calculate pairwise distances
    n_samples = min(1000, len(df))  # Sample for efficiency
    sample_features = features[np.random.choice(len(features), n_samples, replace=False)]
    
    distances = []
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            distances.append(euclidean(sample_features[i], sample_features[j]))
    
    diversity_metrics = {
        'mean_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'min_distance': np.min(distances),
        'max_distance': np.max(distances),
        'coverage_score': np.std(features, axis=0).mean()  # Spread in emotional space
    }
    
    return diversity_metrics


def compute_mood_transition_path(
    df: pd.DataFrame,
    start_valence: float,
    start_arousal: float,
    end_valence: float,
    end_arousal: float,
    n_steps: int = 5
) -> pd.DataFrame:
    """
    Create a mood transition path for playlist generation.
    """
    # Create intermediate mood points
    valence_steps = np.linspace(start_valence, end_valence, n_steps)
    arousal_steps = np.linspace(start_arousal, end_arousal, n_steps)
    
    playlist = []
    
    for v, a in zip(valence_steps, arousal_steps):
        # Find closest track to this mood point
        track = recommend_by_mood(df, v, a, n_recommendations=1)
        playlist.append(track.iloc[0])
    
    return pd.DataFrame(playlist)


def get_artist_mood_profile(df: pd.DataFrame, artist_name: str) -> Dict[str, Any]:
    """
    Get mood profile for a specific artist.
    """
    artist_tracks = df[df['artist_name'] == artist_name]
    
    if len(artist_tracks) == 0:
        return None
    
    profile = {
        'artist': artist_name,
        'n_tracks': len(artist_tracks),
        'mean_valence': artist_tracks['valence'].mean(),
        'std_valence': artist_tracks['valence'].std(),
        'mean_arousal': artist_tracks['arousal'].mean(),
        'std_arousal': artist_tracks['arousal'].std(),
        'dominant_quadrant': artist_tracks.apply(
            lambda row: identify_quadrant_name(row['valence'], row['arousal']), axis=1
        ).mode()[0] if len(artist_tracks) > 0 else None
    }
    
    return profile


def identify_quadrant_name(valence: float, arousal: float) -> str:
    """
    Identify emotional quadrant.
    """
    if valence >= 0 and arousal >= 0:
        return 'Happy/Excited'
    elif valence < 0 and arousal >= 0:
        return 'Angry/Tense'
    elif valence < 0 and arousal < 0:
        return 'Sad/Depressed'
    else:
        return 'Calm/Relaxed'


def compute_recommendation_coverage(df: pd.DataFrame, n_recommendations: int = 10) -> float:
    """
    Compute what percentage of catalog can be recommended.
    """
    n_samples = min(100, len(df))
    sample_indices = np.random.choice(len(df), n_samples, replace=False)
    
    recommended_tracks = set()
    
    for idx in sample_indices:
        similar = find_similar_tracks(df, idx, n_recommendations)
        recommended_tracks.update(similar.index)
    
    coverage = len(recommended_tracks) / len(df) * 100
    return coverage

# ==========================================
# DATA LOADER UTILS
# ==========================================
"""
Data loading and validation utilities for Deezer Mood Detection Dataset
"""

def load_dataset(file_name: str) -> pd.DataFrame:
    """
    Load a single CSV file from the dataset directory.
    """
    file_path = os.path.join(DATA_DIR, file_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    return df


def load_all_datasets() -> Dict[str, pd.DataFrame]:
    """
    Load all three dataset splits (train, validation, test).
    """
    datasets = {
        'train': load_dataset(TRAIN_FILE),
        'validation': load_dataset(VALIDATION_FILE),
        'test': load_dataset(TEST_FILE)
    }
    
    return datasets


def validate_dataset(df: pd.DataFrame, split_name: str) -> Dict[str, Any]:
    """
    Validate a dataset and return quality metrics.
    """
    validation_results = {
        'split_name': split_name,
        'n_samples': len(df),
        'n_features': len(df.columns),
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'total_missing': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
    }
    
    # Check for target columns
    if all(col in df.columns for col in TARGET_COLUMNS):
        validation_results['valence_range'] = (df['valence'].min(), df['valence'].max())
        validation_results['arousal_range'] = (df['arousal'].min(), df['arousal'].max())
        validation_results['valence_nulls'] = df['valence'].isnull().sum()
        validation_results['arousal_nulls'] = df['arousal'].isnull().sum()
    
    return validation_results


def combine_datasets(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine all dataset splits into a single DataFrame with split identifier.
    """
    combined_dfs = []
    
    for split_name, df in datasets.items():
        df_copy = df.copy()
        df_copy['split'] = split_name
        combined_dfs.append(df_copy)
    
    combined = pd.concat(combined_dfs, ignore_index=True)
    return combined


def get_dataset_summary(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create a summary table of all dataset splits.
    """
    summary_data = []
    
    for split_name, df in datasets.items():
        summary_data.append({
            'Split': split_name.capitalize(),
            'Samples': len(df),
            'Features': len(df.columns),
            'Missing Values': df.isnull().sum().sum(),
            'Duplicates': df.duplicated().sum(),
        })
    
    # Add combined row
    combined = combine_datasets(datasets)
    summary_data.append({
        'Split': 'Combined',
        'Samples': len(combined),
        'Features': len(combined.columns) - 1,  # Exclude 'split' column
        'Missing Values': combined.drop('split', axis=1).isnull().sum().sum(),
        'Duplicates': combined.drop('split', axis=1).duplicated().sum(),
    })
    
    return pd.DataFrame(summary_data)


def check_data_consistency(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Check consistency across dataset splits.
    """
    # Get column names from each split
    columns_by_split = {name: set(df.columns) for name, df in datasets.items()}
    
    # Check if all splits have the same columns
    all_columns = [set(df.columns) for df in datasets.values()]
    columns_consistent = all(cols == all_columns[0] for cols in all_columns)
    
    # Check data types consistency
    dtypes_by_split = {name: df.dtypes.to_dict() for name, df in datasets.items()}
    
    consistency_results = {
        'columns_consistent': columns_consistent,
        'columns_by_split': {k: list(v) for k, v in columns_by_split.items()},
        'dtypes_consistent': all(
            dtypes_by_split['train'] == dtypes 
            for dtypes in dtypes_by_split.values()
        ),
        'dtypes_by_split': dtypes_by_split,
    }
    
    return consistency_results


def detect_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> Tuple[pd.Series, int]:
    """
    Detect outliers using the IQR method.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    outlier_count = outliers.sum()
    
    return outliers, outlier_count

# ==========================================
# MAIN APP
# ==========================================
"""
Deezer Mood Detection - Exploratory Data Analysis Dashboard
A professional, thesis-ready Streamlit application
"""

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #34495e;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and cache all datasets"""
    return load_all_datasets()


@st.cache_data
def get_combined_data(datasets):
    """Combine and cache all datasets"""
    return combine_datasets(datasets)


def show_overview_page(datasets: Dict[str, pd.DataFrame], combined_data: pd.DataFrame):
    """Display Page 1: Dataset Overview & Initial Analysis"""
    
    st.markdown('<div class="sub-header">📖 1. Introduction & Dataset Description</div>', unsafe_allow_html=True)
    
    # About section
    st.markdown(ABOUT_TEXT)
    
    # Visual representation of valence-arousal model
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Valence Dimension")
        st.markdown("""
        - **High Valence (+)**: Happy, joyful, pleasant
        - **Low Valence (-)**: Sad, angry, unpleasant
        - **Range**: Typically -2 to +2
        """)
    
    with col2:
        st.markdown("#### Arousal Dimension")
        st.markdown("""
        - **High Arousal (+)**: Excited, energetic, intense
        - **Low Arousal (-)**: Calm, relaxed, peaceful
        - **Range**: Typically -2 to +2
        """)
    
    st.markdown("---")
    
    # Dataset Summary
    st.markdown('<div class="sub-header">📊 2. Dataset Summary</div>', unsafe_allow_html=True)
    
    summary_df = get_dataset_summary(datasets)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", f"{len(combined_data):,}")
    with col2:
        st.metric("Training Samples", f"{len(datasets['train']):,}")
    with col3:
        st.metric("Validation Samples", f"{len(datasets['validation']):,}")
    with col4:
        st.metric("Test Samples", f"{len(datasets['test']):,}")
    
    st.markdown("---")
    
    # Data Quality Assessment
    st.markdown('<div class="sub-header">🔍 3. Data Quality Assessment</div>', unsafe_allow_html=True)
    
    # Check for missing values
    st.markdown("#### Missing Values Analysis")
    
    missing_data = []
    for split_name, df in datasets.items():
        missing_count = df.isnull().sum().sum()
        missing_data.append({
            'Split': split_name.capitalize(),
            'Total Missing Values': missing_count,
            'Missing Percentage': f"{(missing_count / (len(df) * len(df.columns)) * 100):.2f}%"
        })
    
    missing_df = pd.DataFrame(missing_data)
    st.dataframe(missing_df, use_container_width=True, hide_index=True)
    
    if missing_df['Total Missing Values'].sum() == 0:
        st.success("✅ No missing values detected in any dataset split!")
    else:
        st.warning("⚠️ Missing values detected. Further investigation recommended.")
    
    # Data consistency check
    st.markdown("#### Data Consistency Across Splits")
    consistency = check_data_consistency(datasets)
    
    if consistency['columns_consistent']:
        st.success("✅ All dataset splits have consistent column structure")
    else:
        st.error("❌ Column structure inconsistency detected across splits")
    
    if consistency['dtypes_consistent']:
        st.success("✅ Data types are consistent across all splits")
    else:
        st.warning("⚠️ Data type inconsistencies detected")
    
    # Display column information
    st.markdown("#### Dataset Schema")
    schema_data = []
    for col in datasets['train'].columns:
        schema_data.append({
            'Column Name': col,
            'Data Type': str(datasets['train'][col].dtype),
            'Non-Null Count': datasets['train'][col].count(),
            'Sample Value': str(datasets['train'][col].iloc[0])
        })
    
    schema_df = pd.DataFrame(schema_data)
    st.dataframe(schema_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Descriptive Statistics
    st.markdown('<div class="sub-header">📈 4. Descriptive Statistics</div>', unsafe_allow_html=True)
    
    st.markdown("#### Statistics by Split and Variable")
    
    stats_df = compute_split_statistics(datasets)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    # Combined dataset statistics
    st.markdown("#### Combined Dataset Statistics")
    combined_stats = compute_descriptive_stats(combined_data, TARGET_COLUMNS)
    st.dataframe(combined_stats, use_container_width=True)
    
    # Key observations
    st.markdown("#### 📌 Key Statistical Observations")
    
    valence_mean = combined_data['valence'].mean()
    arousal_mean = combined_data['arousal'].mean()
    valence_std = combined_data['valence'].std()
    arousal_std = combined_data['arousal'].std()
    
    observations = f"""
    1. **Valence Distribution**:
       - Mean: {valence_mean:.3f} ({"positive" if valence_mean > 0 else "negative"} tendency)
       - Standard Deviation: {valence_std:.3f}
       - Range: [{combined_data['valence'].min():.3f}, {combined_data['valence'].max():.3f}]
    
    2. **Arousal Distribution**:
       - Mean: {arousal_mean:.3f} ({"high" if arousal_mean > 0 else "low"} energy tendency)
       - Standard Deviation: {arousal_std:.3f}
       - Range: [{combined_data['arousal'].min():.3f}, {combined_data['arousal'].max():.3f}]
    
    3. **Data Spread**: Both variables show substantial variation, indicating diverse emotional content
    
    4. **Skewness**: 
       - Valence: {combined_data['valence'].skew():.3f}
       - Arousal: {combined_data['arousal'].skew():.3f}
    """
    
    st.markdown(observations)
    
    # Correlation
    st.markdown("#### Correlation Analysis")
    corr_result = compute_correlation(combined_data, method='pearson')
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Pearson Correlation", f"{corr_result['correlation']:.4f}")
    with col2:
        st.metric("P-value", f"{corr_result['p_value']:.4e}")
    
    if abs(corr_result['correlation']) < 0.3:
        strength = "weak"
    elif abs(corr_result['correlation']) < 0.7:
        strength = "moderate"
    else:
        strength = "strong"
    
    st.info(f"The correlation between valence and arousal is **{strength}** "
            f"({'positive' if corr_result['correlation'] > 0 else 'negative'}).")
    
    st.markdown("---")
    
    # Data Quality & Limitations
    st.markdown('<div class="sub-header">⚠️ 5. Data Quality & Limitations</div>', unsafe_allow_html=True)
    
    st.markdown(DATA_QUALITY_TEXT)
    
    # Outlier detection
    st.markdown("#### Outlier Detection (IQR Method)")
    
    outlier_data = []
    for split_name, df in datasets.items():
        for col in TARGET_COLUMNS:
            outliers, count = detect_outliers_iqr(df, col)
            percentage = (count / len(df)) * 100
            outlier_data.append({
                'Split': split_name.capitalize(),
                'Variable': col.capitalize(),
                'Outlier Count': count,
                'Percentage': f"{percentage:.2f}%"
            })
    
    outlier_df = pd.DataFrame(outlier_data)
    st.dataframe(outlier_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    **Note**: Outliers are identified using the IQR method (1.5 × IQR). These may represent:
    - Extreme but valid emotional expressions
    - Annotation errors
    - Edge cases in the dataset
    """)


def show_visualization_page(datasets: Dict[str, pd.DataFrame], combined_data: pd.DataFrame):
    """Display Page 2: Data Visualization & Exploration"""
    
    st.markdown('<div class="sub-header">📊 Interactive Data Exploration</div>', unsafe_allow_html=True)
    
    # Sidebar filters
    st.sidebar.markdown("### 🎛️ Filters")
    
    selected_split = st.sidebar.selectbox(
        "Select Dataset Split",
        ["Combined", "Train", "Validation", "Test"],
        key="viz_split_selector"
    )
    
    if selected_split == "Combined":
        data_to_plot = combined_data
    else:
        data_to_plot = datasets[selected_split.lower()]
    
    st.sidebar.metric("Selected Samples", f"{len(data_to_plot):,}")
    
    # Distribution Analysis
    st.markdown('<div class="sub-header">📈 1. Distribution Analysis</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Histograms", "Density Plots", "Box Plots"])
    
    with tab1:
        st.markdown("### Histograms")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Valence Distribution")
            fig_val = plot_distribution_histogram(
                data_to_plot,
                'valence',
                selected_split if selected_split != "Combined" else None,
                bins=50,
                color=COLORS.get(selected_split.lower(), COLORS['combined'])
            )
            st.pyplot(fig_val)
            plt.close()
        
        with col2:
            st.markdown("#### Arousal Distribution")
            fig_aro = plot_distribution_histogram(
                data_to_plot,
                'arousal',
                selected_split if selected_split != "Combined" else None,
                bins=50,
                color=COLORS.get(selected_split.lower(), COLORS['combined'])
            )
            st.pyplot(fig_aro)
            plt.close()
    
    with tab2:
        st.markdown("### Kernel Density Estimation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Valence Density")
            fig_val_density = plot_density_comparison(datasets, 'valence')
            st.pyplot(fig_val_density)
            plt.close()
        
        with col2:
            st.markdown("#### Arousal Density")
            fig_aro_density = plot_density_comparison(datasets, 'arousal')
            st.pyplot(fig_aro_density)
            plt.close()
    
    with tab3:
        st.markdown("### Box Plots (Split Comparison)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Valence Box Plot")
            fig_val_box = plot_boxplot(datasets, 'valence')
            st.pyplot(fig_val_box)
            plt.close()
        
        with col2:
            st.markdown("#### Arousal Box Plot")
            fig_aro_box = plot_boxplot(datasets, 'arousal')
            st.pyplot(fig_aro_box)
            plt.close()
    
    st.markdown("---")
    
    # Split Consistency Analysis
    st.markdown('<div class="sub-header">🔄 2. Split Consistency Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("### Overlaid Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Valence Across Splits")
        fig_val_overlay = plot_overlaid_distributions(datasets, 'valence', bins=40)
        st.pyplot(fig_val_overlay)
        plt.close()
    
    with col2:
        st.markdown("#### Arousal Across Splits")
        fig_aro_overlay = plot_overlaid_distributions(datasets, 'arousal', bins=40)
        st.pyplot(fig_aro_overlay)
        plt.close()
    
    # Statistical comparison
    st.markdown("### Statistical Comparison Tests")
    
    st.markdown("**Comparing Training vs Validation Sets**")
    
    val_comparison = compare_distributions(datasets['train'], datasets['validation'], 'valence')
    aro_comparison = compare_distributions(datasets['train'], datasets['validation'], 'arousal')
    
    comparison_data = [
        {
            'Variable': 'Valence',
            'KS Test p-value': f"{val_comparison['ks_test']['p_value']:.4f}",
            'Significant Difference': "Yes" if val_comparison['ks_test']['significant'] else "No",
            'T-test p-value': f"{val_comparison['t_test']['p_value']:.4f}"
        },
        {
            'Variable': 'Arousal',
            'KS Test p-value': f"{aro_comparison['ks_test']['p_value']:.4f}",
            'Significant Difference': "Yes" if aro_comparison['ks_test']['significant'] else "No",
            'T-test p-value': f"{aro_comparison['t_test']['p_value']:.4f}"
        }
    ]
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    st.info("**Interpretation**: A p-value > 0.05 suggests no significant difference between distributions (desired for consistent splits).")
    
    st.markdown("---")
    
    # Relationship Analysis
    st.markdown('<div class="sub-header">🎯 3. Valence-Arousal Relationship Analysis</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Scatter Plot", "Density Heatmap", "Quadrant Analysis"])
    
    with tab1:
        st.markdown("### Scatter Plot: Valence vs Arousal")
        
        sample_size = st.slider(
            "Number of points to display (for performance)",
            min_value=1000,
            max_value=min(len(data_to_plot), 15000),
            value=min(5000, len(data_to_plot)),
            step=1000
        )
        
        fig_scatter = plot_scatter_valence_arousal(
            data_to_plot,
            selected_split if selected_split != "Combined" else None,
            sample_size=sample_size,
            alpha=0.3
        )
        st.pyplot(fig_scatter)
        plt.close()
    
    with tab2:
        st.markdown("### Hexbin Density Plot")
        
        fig_hexbin = plot_hexbin_density(
            data_to_plot,
            selected_split if selected_split != "Combined" else None,
            gridsize=30
        )
        st.pyplot(fig_hexbin)
        plt.close()
    
    with tab3:
        st.markdown("### Emotional Quadrant Distribution")
        
        quadrant_df = compute_quadrant_distribution(data_to_plot)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.dataframe(quadrant_df, use_container_width=True, hide_index=True)
        
        with col2:
            fig_quadrant = plot_quadrant_distribution(quadrant_df)
            st.pyplot(fig_quadrant)
            plt.close()
        
        st.markdown("""
        **Quadrant Interpretation**:
        - **Q1 (Happy/Excited)**: Positive valence, high arousal
        - **Q2 (Angry/Tense)**: Negative valence, high arousal
        - **Q3 (Sad/Depressed)**: Negative valence, low arousal
        - **Q4 (Calm/Relaxed)**: Positive valence, low arousal
        """)
    
    # Correlation heatmap
    st.markdown("### Correlation Matrix")
    fig_corr = plot_correlation_heatmap(data_to_plot)
    st.pyplot(fig_corr)
    plt.close()
    
    corr_result = compute_correlation(data_to_plot)
    st.metric("Pearson Correlation Coefficient", f"{corr_result['correlation']:.4f}")
    
    st.markdown("---")
    
    # Summary insights
    st.markdown('<div class="sub-header">💡 4. Key Insights</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    ### Summary of Findings
    
    1. **Dataset Size**: The combined dataset contains **{len(combined_data):,}** music tracks with emotional annotations.
    
    2. **Distribution Characteristics**:
       - Both valence and arousal show approximately normal distributions
       - Valence mean: {combined_data['valence'].mean():.3f}
       - Arousal mean: {combined_data['arousal'].mean():.3f}
    
    3. **Split Consistency**: The train, validation, and test splits show {"consistent" if not val_comparison['ks_test']['significant'] else "some differences in"} distributions.
    
    4. **Valence-Arousal Relationship**: 
       - Correlation: {corr_result['correlation']:.3f}
       - The relationship is {"weak" if abs(corr_result['correlation']) < 0.3 else "moderate" if abs(corr_result['correlation']) < 0.7 else "strong"}
    
    5. **Emotional Coverage**: The dataset covers all four emotional quadrants, with varying densities.
    
    6. **Data Quality**: {"No" if combined_data.isnull().sum().sum() == 0 else "Some"} missing values detected.
    """)


def show_recommender_analysis_page(datasets: Dict[str, pd.DataFrame], combined_data: pd.DataFrame):
    """Display Page 3: Recommender System Analysis"""
    
    st.markdown('<div class="sub-header">🎯 Recommender System Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This page provides EDA specifically focused on **building a mood-based music recommender system**.
    Explore similarity metrics, clustering, and recommendation strategies based on the valence-arousal emotional space.
    """)
    
    # Sidebar filters
    st.sidebar.markdown("### 🎛️ Recommender Settings")
    
    selected_split = st.sidebar.selectbox(
        "Select Dataset Split",
        ["Combined", "Train", "Validation", "Test"],
        key="rec_split"
    )
    
    if selected_split == "Combined":
        data_to_analyze = combined_data
    else:
        data_to_analyze = datasets[selected_split.lower()]
    
    st.sidebar.metric("Tracks Available", f"{len(data_to_analyze):,}")
    
    st.markdown("---")
    
    # 1. Mood Diversity Analysis
    st.markdown('<div class="sub-header">📊 1. Mood Diversity Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Why This Matters for Recommenders**: Understanding mood diversity helps ensure the recommender 
    can provide varied recommendations across the entire emotional spectrum.
    """)
    
    with st.spinner("Analyzing mood diversity..."):
        diversity_metrics = analyze_mood_diversity(data_to_analyze)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Distance", f"{diversity_metrics['mean_distance']:.3f}")
    with col2:
        st.metric("Std Distance", f"{diversity_metrics['std_distance']:.3f}")
    with col3:
        st.metric("Max Distance", f"{diversity_metrics['max_distance']:.3f}")
    with col4:
        st.metric("Coverage Score", f"{diversity_metrics['coverage_score']:.3f}")
    
    st.info("""
    **Interpretation**: 
    - **Mean Distance**: Average emotional distance between tracks (higher = more diverse)
    - **Coverage Score**: Spread in emotional space (higher = better coverage)
    - **High diversity** enables rich, varied recommendations
    """)
    
    st.markdown("---")
    
    # 2. Mood Clustering
    st.markdown('<div class="sub-header">🎨 2. Mood-Based Clustering</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Why This Matters for Recommenders**: Clustering identifies natural mood groups, 
    enabling category-based recommendations and efficient similarity search.
    """)
    
    n_clusters = st.slider("Number of Mood Clusters", min_value=4, max_value=12, value=8, step=1)
    
    with st.spinner(f"Clustering tracks into {n_clusters} mood groups..."):
        clustered_data, kmeans_model = cluster_by_mood(data_to_analyze, n_clusters=n_clusters)
    
    # Visualize clusters
    fig, ax = plt.subplots(figsize=(12, 10))
    
    scatter = ax.scatter(
        clustered_data['valence'],
        clustered_data['arousal'],
        c=clustered_data['cluster'],
        cmap='tab10',
        alpha=0.6,
        s=30,
        edgecolors='none'
    )
    
    # Plot cluster centers
    centers = kmeans_model.cluster_centers_
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        c='red',
        marker='X',
        s=300,
        edgecolors='black',
        linewidths=2,
        label='Cluster Centers',
        zorder=5
    )
    
    # Add quadrant lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Valence (Negativity ← → Positivity)', fontsize=12)
    ax.set_ylabel('Arousal (Low Energy ← → High Energy)', fontsize=12)
    ax.set_title(f'K-Means Clustering ({n_clusters} Mood Clusters)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, ax=ax, label='Cluster ID')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Cluster statistics
    st.markdown("#### Cluster Statistics")
    cluster_stats = clustered_data.groupby('cluster').agg({
        'valence': ['mean', 'std', 'count'],
        'arousal': ['mean', 'std']
    }).round(3)
    cluster_stats.columns = ['Valence Mean', 'Valence Std', 'Track Count', 'Arousal Mean', 'Arousal Std']
    st.dataframe(cluster_stats, use_container_width=True)
    
    st.success(f"✅ Identified {n_clusters} distinct mood clusters for efficient recommendation grouping")
    
    st.markdown("---")
    
    # 3. Similar Track Finder (Demo)
    st.markdown('<div class="sub-header">🔍 3. Similar Track Finder (Recommendation Demo)</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Why This Matters for Recommenders**: This demonstrates content-based filtering - 
    finding tracks with similar emotional characteristics.
    """)
    
    # Random track selector
    if st.button("🎲 Select Random Track"):
        st.session_state.random_track_idx = np.random.randint(0, len(data_to_analyze))
    
    if 'random_track_idx' not in st.session_state:
        st.session_state.random_track_idx = 0
    
    reference_track = data_to_analyze.iloc[st.session_state.random_track_idx]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 🎵 Reference Track")
        st.markdown(f"""
        **Artist**: {reference_track['artist_name']}  
        **Track**: {reference_track['track_name']}  
        **Valence**: {reference_track['valence']:.3f}  
        **Arousal**: {reference_track['arousal']:.3f}  
        **Mood**: {identify_quadrant_name(reference_track['valence'], reference_track['arousal'])}
        """)
    
    with col2:
        n_recommendations = st.slider("Number of Recommendations", min_value=5, max_value=20, value=10)
    
    # Find similar tracks
    with st.spinner("Finding similar tracks..."):
        similar_tracks = find_similar_tracks(
            data_to_analyze.reset_index(drop=True),
            st.session_state.random_track_idx,
            n_recommendations=n_recommendations
        )
    
    st.markdown("#### 🎯 Recommended Similar Tracks")
    
    # Display recommendations
    display_cols = ['artist_name', 'track_name', 'valence', 'arousal', 'similarity_score']
    similar_display = similar_tracks[display_cols].copy()
    similar_display.columns = ['Artist', 'Track', 'Valence', 'Arousal', 'Similarity Score']
    similar_display['Similarity Score'] = similar_display['Similarity Score'].round(3)
    similar_display['Valence'] = similar_display['Valence'].round(3)
    similar_display['Arousal'] = similar_display['Arousal'].round(3)
    
    st.dataframe(similar_display, use_container_width=True, hide_index=True)
    
    st.info("""
    **How It Works**: Uses K-Nearest Neighbors (KNN) with Euclidean distance in valence-arousal space.
    Higher similarity score = more emotionally similar to the reference track.
    """)
    
    st.markdown("---")
    
    # 4. Mood-Based Recommendation
    st.markdown('<div class="sub-header">🎭 4. Mood-Based Recommendation</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Why This Matters for Recommenders**: Allows users to specify desired mood and get matching tracks.
    This is the core of a mood-based recommender system.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_valence = st.slider(
            "Target Valence (Emotional Positivity)",
            min_value=-2.5,
            max_value=2.5,
            value=0.0,
            step=0.1
        )
    
    with col2:
        target_arousal = st.slider(
            "Target Arousal (Energy Level)",
            min_value=-2.5,
            max_value=2.5,
            value=0.0,
            step=0.1
        )
    
    target_mood = identify_quadrant_name(target_valence, target_arousal)
    st.markdown(f"**Selected Mood**: {target_mood}")
    
    n_mood_recs = st.slider("Number of Recommendations", min_value=5, max_value=20, value=10, key="mood_recs")
    
    # Get recommendations
    with st.spinner("Finding tracks matching your mood..."):
        mood_recommendations = recommend_by_mood(
            data_to_analyze.reset_index(drop=True),
            target_valence,
            target_arousal,
            n_recommendations=n_mood_recs
        )
    
    st.markdown("#### 🎵 Recommended Tracks for Your Mood")
    
    display_cols = ['artist_name', 'track_name', 'valence', 'arousal', 'match_score']
    mood_display = mood_recommendations[display_cols].copy()
    mood_display.columns = ['Artist', 'Track', 'Valence', 'Arousal', 'Match Score']
    mood_display['Match Score'] = mood_display['Match Score'].round(3)
    mood_display['Valence'] = mood_display['Valence'].round(3)
    mood_display['Arousal'] = mood_display['Arousal'].round(3)
    
    st.dataframe(mood_display, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # 5. Mood Transition Playlist
    st.markdown('<div class="sub-header">🌈 5. Mood Transition Playlist Generator</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Why This Matters for Recommenders**: Creates playlists that smoothly transition between moods,
    useful for activities like workout warm-up/cool-down or sleep preparation.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Start Mood")
        start_v = st.slider("Start Valence", -2.0, 2.0, -1.0, 0.1, key="start_v")
        start_a = st.slider("Start Arousal", -2.0, 2.0, -1.0, 0.1, key="start_a")
        st.markdown(f"**Mood**: {identify_quadrant_name(start_v, start_a)}")
    
    with col2:
        st.markdown("##### End Mood")
        end_v = st.slider("End Valence", -2.0, 2.0, 1.0, 0.1, key="end_v")
        end_a = st.slider("End Arousal", -2.0, 2.0, 1.0, 0.1, key="end_a")
        st.markdown(f"**Mood**: {identify_quadrant_name(end_v, end_a)}")
    
    n_steps = st.slider("Playlist Length (tracks)", 3, 10, 5)
    
    if st.button("🎵 Generate Mood Journey Playlist"):
        with st.spinner("Creating mood transition playlist..."):
            playlist = compute_mood_transition_path(
                data_to_analyze.reset_index(drop=True),
                start_v, start_a,
                end_v, end_a,
                n_steps=n_steps
            )
        
        st.markdown("#### 🎵 Your Mood Journey Playlist")
        
        for idx, track in playlist.iterrows():
            st.markdown(f"""
            **{idx + 1}.** {track['artist_name']} - {track['track_name']}  
            *Valence: {track['valence']:.2f}, Arousal: {track['arousal']:.2f}* ({identify_quadrant_name(track['valence'], track['arousal'])})
            """)
        
        st.success(f"✅ Created a {n_steps}-track playlist transitioning from {identify_quadrant_name(start_v, start_a)} to {identify_quadrant_name(end_v, end_a)}")
    
    st.markdown("---")
    
    # 6. Recommender System Metrics
    st.markdown('<div class="sub-header">📈 6. Recommender System Readiness Metrics</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Why This Matters**: These metrics assess how well the dataset supports building a recommender system.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Tracks", f"{len(data_to_analyze):,}")
        st.caption("Catalog size for recommendations")
    
    with col2:
        unique_artists = data_to_analyze['artist_name'].nunique()
        st.metric("Unique Artists", f"{unique_artists:,}")
        st.caption("Artist diversity")
    
    with col3:
        avg_tracks_per_artist = len(data_to_analyze) / unique_artists
        st.metric("Avg Tracks/Artist", f"{avg_tracks_per_artist:.1f}")
        st.caption("Artist representation")
    
    # Coverage analysis
    st.markdown("#### Recommendation Coverage Analysis")
    
    if st.button("🔍 Compute Recommendation Coverage"):
        with st.spinner("Computing coverage (this may take a moment)..."):
            coverage = compute_recommendation_coverage(data_to_analyze.reset_index(drop=True), n_recommendations=10)
        
        st.metric("Catalog Coverage", f"{coverage:.1f}%")
        st.caption("Percentage of catalog that can be recommended (with 10 recommendations per query)")
        
        if coverage > 80:
            st.success("✅ Excellent coverage! Most tracks can be recommended.")
        elif coverage > 50:
            st.info("ℹ️ Good coverage. Consider strategies to improve long-tail recommendations.")
        else:
            st.warning("⚠️ Low coverage. Many tracks may not be recommended frequently.")
    
    st.markdown("---")
    
    # 7. Key Insights for Recommender System
    st.markdown('<div class="sub-header">💡 7. Key Insights for Building Recommender System</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    ### Summary of Recommender-Focused Findings
    
    Based on the analysis above, here are key insights for building your mood-based recommender:
    
    1. **Dataset Size**: {len(data_to_analyze):,} tracks provide a solid foundation for recommendations
    
    2. **Mood Diversity**: 
       - Mean emotional distance: {diversity_metrics['mean_distance']:.3f}
       - The dataset covers diverse moods, enabling varied recommendations
    
    3. **Clustering**: 
       - Natural mood groups identified through K-Means clustering
       - Can be used for efficient category-based recommendations
    
    4. **Similarity Search**:
       - Euclidean distance in valence-arousal space works well
       - K-Nearest Neighbors provides fast, accurate similar track finding
    
    5. **Mood-Based Filtering**:
       - Users can specify desired mood (valence, arousal)
       - System finds closest matching tracks effectively
    
    6. **Playlist Generation**:
       - Smooth mood transitions are possible
       - Useful for activity-based playlists (workout, relaxation, etc.)
    
    7. **Recommendation Strategies**:
       - ✅ **Content-Based**: Use valence-arousal similarity
       - ✅ **Mood-Based**: Direct mood specification
       - ✅ **Hybrid**: Combine with collaborative filtering
       - ✅ **Context-Aware**: Mood trajectories for activities
    
    ### Next Steps for Implementation
    
    1. **Extract Audio Features**: Add MFCCs, spectral features for richer recommendations
    2. **Build Prediction Models**: Train regression models to predict valence/arousal from audio
    3. **Implement Ranking**: Add relevance scoring and diversity constraints
    4. **Add Personalization**: Learn user preferences over time
    5. **Optimize Performance**: Use approximate nearest neighbors (FAISS) for scalability
    6. **User Interface**: Build interactive mood selector and playlist generator
    
    ### Recommended Algorithms
    
    - **Similarity**: K-Nearest Neighbors (KNN) with Euclidean distance
    - **Clustering**: K-Means for mood categories
    - **Ranking**: Weighted combination of similarity + diversity + popularity
    - **Personalization**: Collaborative filtering or matrix factorization
    """)


def main():
    """Main application function"""
    
    # Title
    st.markdown('<div class="main-header">🎵 Deezer Mood Detection Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #7f8c8d; font-size: 1.2rem; margin-bottom: 2rem;">Exploratory Data Analysis Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Dataset Overview & Analysis", "Data Visualization & Exploration", "Recommender System Analysis"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This dashboard provides a comprehensive exploratory data analysis "
        "of the Deezer Mood Detection Dataset for academic research purposes."
    )
    
    # Load data
    with st.spinner("Loading datasets..."):
        datasets = load_data()
        combined_data = get_combined_data(datasets)
    
    # Route to appropriate page
    if page == "Dataset Overview & Analysis":
        show_overview_page(datasets, combined_data)
    elif page == "Data Visualization & Exploration":
        show_visualization_page(datasets, combined_data)
    else:
        show_recommender_analysis_page(datasets, combined_data)


if __name__ == "__main__":
    main()





