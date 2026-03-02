"""
Automobile Sensor Anomaly Detection using PCA

Analyzes sensor data using Principal Component Analysis (PCA) to detect anomalies
based on reconstruction error. Includes eigenvalue/eigenvector analysis.
"""

import pandas as pd
from ai_functions import ai_function
from ai_functions.types import AIFunctionConfig
from pathlib import Path


# Configuration for ML operations
ML_CONFIG = AIFunctionConfig(
    code_executor_additional_imports=[
        "pandas.*", 
        "numpy.*", 
        "sklearn.*",
        "matplotlib.*",
        "plotly.*"
    ],
    code_execution_mode="local"
)


@ai_function(config=ML_CONFIG)
def load_sensor_data(file_path: str = "data/sensor_data.csv") -> pd.DataFrame:
    """
    Load automobile sensor data from CSV file.
    
    Load the data from: {file_path}
    
    Steps:
    1. Check if the file exists using os.path.exists()
    2. If file doesn't exist, raise FileNotFoundError with message: "Data file not found: {{file_path}}. Please run generate_data.py first to create the dataset."
    3. Load the CSV file using pd.read_csv()
    4. Convert the 'timestamp' column to datetime using pd.to_datetime()
    5. Verify the loaded data has at least 100 rows, if not raise ValueError: "Loaded data has insufficient rows"
    6. Print: "Loaded X samples from {{file_path}}"
    
    Return the DataFrame with timestamp as datetime type.
    """


@ai_function(config=ML_CONFIG)
def perform_pca_analysis(data: pd.DataFrame, n_components: int = 10) -> dict:
    """
    Perform PCA analysis on sensor data and extract eigenvalues/eigenvectors.
    
    Given data with {data.shape[0]} samples and columns: {list(data.columns)}
    
    Steps:
    1. Select numeric sensor features (exclude timestamp, vehicle_id, latitude, longitude)
    2. Standardize the features using sklearn.preprocessing.StandardScaler
    3. Fit PCA with n_components={n_components} using sklearn.decomposition.PCA
    4. Extract eigenvalues (explained_variance_) and eigenvectors (components_)
    5. Calculate cumulative variance explained
    
    Print the following:
    - "PCA Analysis Results"
    - "=" * 70
    - For each component (top {n_components}):
      * "PC{{i}}: Eigenvalue={{eigenvalue:.4f}}, Variance={{variance_pct:.2f}}%, Cumulative={{cumulative:.2f}}%"
    - "=" * 70
    - "\nTop 5 sensor loadings for each component:"
    - For each of top 3 components, show the 5 sensors with highest absolute loadings
    
    Return a dictionary with:
    - 'pca_model': the fitted PCA object
    - 'scaler': the fitted StandardScaler object
    - 'eigenvalues': array of eigenvalues
    - 'eigenvectors': array of eigenvectors (components)
    - 'variance_explained': array of variance explained ratios
    - 'cumulative_variance': array of cumulative variance
    - 'feature_names': list of feature names used
    - 'transformed_data': PCA-transformed data
    """


@ai_function(config=ML_CONFIG)
def detect_anomalies_pca(data: pd.DataFrame, pca_results: dict, contamination: float = 0.05) -> pd.DataFrame:
    """
    Detect anomalies using PCA reconstruction error method.
    
    Given data with {data.shape[0]} samples and PCA results.
    
    Steps:
    1. Select the same numeric features used in PCA
    2. Standardize using the fitted scaler from pca_results
    3. Transform to PCA space using pca_results['pca_model']
    4. Inverse transform back to original space
    5. Calculate reconstruction error (MSE between original and reconstructed)
    6. Determine threshold at {contamination*100}th percentile of reconstruction errors
    7. Mark samples with error > threshold as anomalies
    8. Add columns to data:
       - 'reconstruction_error': the MSE for each sample
       - 'is_anomaly_pca': boolean (True if anomaly)
    9. Print: "Detected X anomalies using PCA (Y% of data)"
    10. Print: "Reconstruction error threshold: Z"
    
    Return the DataFrame with anomaly labels added.
    """


@ai_function(config=ML_CONFIG)
def create_pca_visualizations(data: pd.DataFrame, pca_results: dict) -> dict[str, str]:
    """
    Create visualizations for PCA analysis and anomaly detection.
    
    Using the data with {data.shape[0]} samples and PCA results:
    
    Create 4 Plotly visualizations:
    
    1. 'scree_plot': Scree plot showing eigenvalues
       - X-axis: Principal Component number (1 to n)
       - Y-axis: Eigenvalue (variance explained)
       - Add line for cumulative variance (secondary y-axis)
       - Title: "PCA Scree Plot - Eigenvalues and Cumulative Variance"
       - Mark the "elbow" point
    
    2. 'biplot': Biplot of first 2 principal components
       - X-axis: PC1
       - Y-axis: PC2
       - Color points by is_anomaly_pca (red=anomaly, blue=normal)
       - Add arrows for top 5 sensor loadings (eigenvectors)
       - Title: "PCA Biplot - First Two Components"
    
    3. 'reconstruction_error_dist': Distribution of reconstruction errors
       - Histogram of reconstruction errors
       - Separate colors for normal vs anomalies
       - Add vertical line for threshold
       - Title: "Reconstruction Error Distribution"
    
    4. 'pca_3d': 3D scatter plot of first 3 PCs
       - X, Y, Z: PC1, PC2, PC3
       - Color by is_anomaly_pca
       - Title: "3D PCA Space - Anomaly Detection"
    
    Return dictionary mapping visualization names to HTML strings.
    """


@ai_function(config=ML_CONFIG)
def generate_pca_report(data: pd.DataFrame, pca_results: dict) -> str:
    """
    Generate a comprehensive PCA analysis report.
    
    Analyze the data with {data.shape[0]} samples and {data['is_anomaly_pca'].sum()} anomalies.
    
    Create a detailed text report including:
    
    1. PCA Summary:
       - Total variance explained by top N components
       - Number of components needed for 95% variance
       - Interpretation of top 3 components (which sensors dominate)
    
    2. Eigenvalue Analysis:
       - List all eigenvalues with variance percentages
       - Identify the "elbow" point
    
    3. Anomaly Detection Results:
       - Total anomalies detected
       - Comparison with expected contamination rate
       - Reconstruction error statistics (mean, std, min, max)
    
    4. Top Anomalies:
       - Show 5 samples with highest reconstruction errors
       - Which sensors deviate most from normal patterns
    
    5. Component Interpretation:
       - For top 3 components, explain what they represent based on sensor loadings
       - Example: "PC1 represents driving intensity (high speed, RPM)"
    
    6. Recommendations:
       - Which vehicles have most anomalies
       - Which sensors are most problematic
       - Suggested actions
    
    Return formatted text report.
    """


def main():
    """Main PCA analysis pipeline"""
    print("=" * 70)
    print("AUTOMOBILE SENSOR ANOMALY DETECTION - PCA METHOD")
    print("=" * 70)
    
    # Load data
    print("\nLoading sensor data...")
    data = load_sensor_data()
    
    # Perform PCA analysis
    print("\n1. Performing PCA analysis...")
    pca_results = perform_pca_analysis(data, n_components=10)
    
    # Detect anomalies using PCA
    print("\n2. Detecting anomalies using reconstruction error...")
    results = detect_anomalies_pca(data, pca_results, contamination=0.05)
    anomaly_count = results['is_anomaly_pca'].sum()
    print(f"✓ Detected {anomaly_count} anomalies ({anomaly_count/len(results)*100:.1f}%)")
    
    # Create visualizations
    print("\n3. Creating PCA visualizations...")
    visualizations = create_pca_visualizations(results, pca_results)
    print(f"✓ Generated {len(visualizations)} visualizations")
    
    # Save visualizations
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    for name, html_content in visualizations.items():
        viz_file = output_dir / f"pca_{name}.html"
        viz_file.write_text(html_content)
        print(f"  - Saved {viz_file}")
    
    # Generate report
    print("\n4. Generating PCA analysis report...")
    report = generate_pca_report(results, pca_results)
    
    report_file = output_dir / "pca_analysis_report.txt"
    report_file.write_text(report)
    print(f"✓ Report saved to {report_file}")
    
    # Save results
    results_file = output_dir / "sensor_data_with_pca_anomalies.csv"
    results.to_csv(results_file, index=False)
    print(f"✓ Results saved to {results_file}")
    
    # Display report
    print("\n" + "=" * 70)
    print("PCA ANALYSIS REPORT")
    print("=" * 70)
    print(report)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nView visualizations:")
    for name in visualizations.keys():
        print(f"  - open data/pca_{name}.html")
    
    print("\nKey outputs:")
    print(f"  - Eigenvalues and eigenvectors printed above")
    print(f"  - PCA space visualizations saved")
    print(f"  - Reconstruction error analysis complete")


if __name__ == "__main__":
    main()
