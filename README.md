# AF-SLIME: AlphaFold3 SLIM discovery pipeline


```
    █████╗ ███████╗              ███████╗██╗     ██╗███╗   ███╗███████╗ 
   ██╔══██╗██╔════╝  ▃▃▃▃▃▃      ██╔════╝██║     ██║████╗ ████║██╔════╝   
   ███████║█████╗     ▃▃▃▃▃▃     ███████╗██║     ██║██╔████╔██║█████╗       
   ██╔══██║██╔══╝      ▃▃▃▃▃▃    ╚════██║██║     ██║██║╚██╔╝██║██╔══╝         
   ██║  ██║██║          ▃▃▃▃▃▃   ███████║███████╗██║██║ ╚═╝ ██║███████╗        
   ╚═╝  ╚═╝╚═╝                   ╚══════╝╚══════╝╚═╝╚═╝     ╚═╝╚══════╝         
```

## 💡 What is AF-SLIME?

**AF-SLIME** is a comprehensive pipeline for analyzing and visualizing Short Linear Motif (SLIM) interactions using AlphaFold3. The tool streamlines the process of generating predictions for peptide fragments against target proteins and provides detailed analysis of binding interactions, confidence metrics, and post-translational modifications (PTMs).

With AF-SLIME, you can:
- Generate sliding window peptide fragments from a protein sequence
- Submit these fragments to AlphaFold3 for structure prediction
- Analyze confidence metrics (e.g pLDDT, PAE, contact probabilities) across all predictions
- Score protein interfaces using PyRosetta
- Identify contact patterns and binding hotspots
- Visualize results through comprehensive plots and contact maps
- Analyze the effects of post-translational modifications on binding

AF-SLIME is particularly useful for studying intrinsically disordered regions (IDRs), usually contaning putative SLIMs,  and their interactions with structured domains, especially when post-translational modifications play a key role in modulating these interactions.

## 🧰 Installation

### Prerequisites

- AlphaFold3
- PyRosetta (for interface scoring)

### Step-by-Step Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/TiagoLopesGomes/af3_slime.git
   cd af3_slime
   ```

2. **Create a conda environment** (recommended)

   ```bash
   conda create -n af3_slime python=3.11
   conda activate af3_slime
   ```

3. **Install required packages**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install PyRosetta**

   PyRosetta is required for interface scoring. Follow the installation instructions on the [PyRosetta website](http://www.pyrosetta.org/) or use:

   ```bash
   pip install pyrosetta-installer
   python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'
   ```

4. **Install AlphaFold3**

   Check instructions [here](https://github.com/Kuhlman-Lab/alphafold3).


## 🔎 How Does AF-SLIME Work?

AF-SLIME operates in two main stages:

1. **Generation Stage** (main.py): Generate peptide fragments and run AlphaFold3 predictions
2. **Analysis Stage** (plot_metrics.py): Analyze results and create visualizations

### Main Workflow

1. **Fragment Generation and Prediction**:
   - Read receptor and ligand sequences from FASTA files
   - Generate overlapping peptide fragments with specified length and offset
   - Validate and incorporate post-translational modifications
   - Create input JSON files for each fragment
   - Submit to AlphaFold3 for structure prediction
   - Collect prediction outputs (structures, confidence scores)

2. **Analysis and Visualization**:
   - Collect pLDDT values across all predictions
   - Extract contact probabilities and PAE data
   - Perform interface scoring using PyRosetta
   - Identify key contact residues and binding hotspots
   - Generate plots for confidence metrics, contact probabilities, and interface metrics
   - Create circular contact diagrams and perform cluster analysis

## 💻 Usage

AF-SLIME is a two-step workflow. First, you generate predictions, then you analyze them.

### Step 1: Generate Peptide Fragments and AlphaFold3 Predictions

Use `main.py` to generate peptide fragments from your proteins and run AlphaFold3 predictions:

```bash
python main.py --receptor receptor.fasta --ligand ligand.fasta --output-dir af3_slide_preds_your_project --peptide-size 10 --offset 1 --start-residue 1 --ptm-config ptm.yaml
```

#### Command-line Arguments for main.py:

| Argument | Description |
|----------|-------------|
| `--receptor` | Path to the FASTA file containing the receptor sequence (required) |
| `--ligand` | Path to the FASTA file containing the ligand sequence (required) |
| `--output-dir` | Directory to store prediction outputs (default: "af3_predictions") |
| `--peptide-size` | Size of each peptide fragment (default: 10 residues) |
| `--offset` | Offset between consecutive fragments (default: 5 residues) |
| `--start-residue` | Starting residue number for the first ligand fragment |
| `--ptm-config` | Path to YAML file containing PTM configurations (optional) |
| `--af3-script-path` | Path to the AlphaFold3 run script (default: "~/software/alphafold3/run/run_af3.py") |

#### PTM Configuration (ptm.yaml):

If you want to include post-translational modifications, create a YAML file with the following format:

```yaml
modifications:
  - ptmType: "PTR"    # Chemical Component Dictionary (CCD) code for e.g phosphotyrosine
    ptmPosition: 454  # Absolute residue position in the ligand sequence
  - ptmType: "PTR"
    ptmPosition: 474
  - ptmType: "SEP"    # Phosphoserine
    ptmPosition: 483
```

The PTM types use the Chemical Component Dictionary (CCD) codes from AlphaFold3's source code. Common codes include:
- `PTR`: Phosphotyrosine
- `SEP`: Phosphoserine
- `TPO`: Phosphothreonine
- `MLY`: Methyllysine
- `ALY`: Acetyllysine

### Step 2: Analyze Predictions and Generate Visualizations

After running the predictions, use `plot_metrics.py` to analyze the results:

```bash
python plot_metrics.py --config your_config.yaml
```

#### Configuration File for Analysis (your_config.yaml):

```yaml
# AF-SLIME Example User Configuration

# Input/Output settings
input:
  dir: "./your_output_dir"                    # Directory containing prediction files
  ptm_config: "ptm.yaml"                      # PTM configuration file (optional)

output:
  metrics: "plddt_pae.png"                    # pLDDT and PAE metrics output
  contacts: "ca_contac_probs.png"             # Contact probabilities output
  interface_metrics: "interface_metrics.png"  # Interface metrics output
  chord_diagram: "circos.png"                 # Circular chord diagram output
  
  # Individual cluster output filenames
  clusters:
    assignments: "cluster_assignments.png"    # Plot showing cluster assignments along sequence
    features: "cluster_features.png"          # Feature importance visualization
    dendrogram: "cluster_dendrogram.png"      # Hierarchical clustering dendrogram
    heatmap: "cluster_heatmap.png"            # Feature heatmap with clustering
    summary: "cluster_summary.png"            # Comprehensive cluster summary

# Processing options
processing:
  atom_type: "CA"       # Atom type to use for analysis
  do_relax: true        # Whether to perform relaxation (set to false to use pre-relaxed structures if available)
  
  # Receptor settings
  receptor:
    start: 1            # Starting residue number for receptor

# Internal parameters
parameters:
  # Contact analysis parameters
  contact_analysis:
    top_n: 3            # Number of top contacts to average
    threshold: 0.3      # Threshold for significant contacts
    window_size: 3      # Window size for neighborhood calculations
  
  # Clustering parameters
  clustering:
    eps: 0.3            # Maximum distance between samples for DBSCAN
    min_samples: 3      # Minimum samples to form a dense region in DBSCAN
    threshold: 0.3      # Threshold for contact significance
  
  # Interface scoring parameters
  interface_scoring:
    distance_cutoff: 8.0     # Distance cutoff for contacts (Angstroms)
    clash_threshold: 2.0     # Threshold for clash detection
    contact_threshold: 0.5   # Threshold for contact probability
  
  # PTM token filtering
  ptm_filtering:
    keep_second_token: true      # Keep the second token (CA) for PTM residues
    average_tokens: false        # If true, average the values across all tokens for PTM residues instead of keeping just one 
    tokens_to_average: ["ALL"]   # Use "ALL" to average all tokens, or specify tokens like ["CA", "CB"]
```

## 📊 Example Results

<div align="center">
<img src="docs/assets/example_plots.png" width="90%" alt="Example Plots">
</div>

### The plot_metrics.py Workflow

The `plot_metrics.py` script performs comprehensive analysis of AlphaFold3 predictions:

1. **Data Collection**:
   - Reads all prediction files in the specified directory
   - Extracts confidence metrics (pLDDT, PAE, contact probabilities)
   - Collects interface metrics through PyRosetta scoring

2. **Data Processing**:
   - Aligns data by absolute residue positions
   - Calculates averages and standard errors across all predictions
   - Identifies binding hotspots based on contact probabilities
   - Handles post-translational modifications appropriately

3. **Visualization**:
   - Generates plots showing average metrics with error bands
   - Creates circular contact diagrams showing key interactions
   - Produces cluster analysis to identify binding patterns

## 🙏 Acknowledgments

- The AlphaFold3 team at Deepmind
- The PyRosetta team 
- The BindCraft team 

## 📚 References

If you use AF-SLIME in your research, please cite:

```
[Citation information]
```