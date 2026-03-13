"""
Automobile Sensor Anomaly Detection

Analyzes sensor data using machine learning to detect anomalies and visualize results.
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
def detect_anomalies(data: pd.DataFrame) -> pd.DataFrame:
    """
    Detect anomalies in automobile sensor data using Isolation Forest.
    
    Given data with {data.shape[0]} samples and columns: {list(data.columns)}
    
    Steps:
    1. Select numeric sensor features (exclude timestamp, vehicle_id, latitude, longitude)
    2. Use sklearn.ensemble.IsolationForest with:
       - contamination=0.05 (expected 5% anomalies)
       - random_state=42 for reproducibility
    3. Fit the model and predict anomalies
    4. Add two new columns to the data:
       - 'anomaly_score': the anomaly score from the model (lower = more anomalous)
       - 'is_anomaly': boolean (True if anomaly detected, False if normal)
    5. Print summary: "Detected X anomalies out of Y samples (Z%)"
    
    Return the DataFrame with the two new columns added.
    """


@ai_function(config=ML_CONFIG)
def create_visualizations(data: pd.DataFrame) -> dict[str, str]:
    """
    Create interactive visualizations for anomaly analysis.
    
    Using the data with {data.shape[0]} samples and 'is_anomaly' column:
    
    Create 3 Plotly visualizations:
    
    1. 'speed_rpm_plot': Scatter plot
       - X-axis: speed_kmh
       - Y-axis: rpm
       - Color: is_anomaly (red for anomalies, blue for normal)
       - Size: engine_temp_celsius
       - Hover: show all sensor values
       - Title: "Speed vs RPM - Anomaly Detection"
    
    2. 'temperature_analysis': Line plot over time
       - X-axis: timestamp
       - Y-axis: engine_temp_celsius, coolant_temp_celsius, transmission_temp_celsius
       - Mark anomalies with red dots
       - Title: "Temperature Sensors Over Time"
    
    3. 'tire_pressure_heatmap': Heatmap
       - Show tire pressure for all 4 tires over time
       - Highlight anomalous readings
       - Title: "Tire Pressure Monitoring"
    
    Return a dictionary mapping visualization names to their HTML content.
    Each should be a complete standalone HTML string using fig.to_html().
    """


@ai_function(config=ML_CONFIG)
def generate_summary_report(data: pd.DataFrame) -> str:
    """
    Generate a summary analysis report.
    
    Analyze the data with {data.shape[0]} samples and {data['is_anomaly'].sum()} anomalies.
    
    Create a text report including:
    1. Overall statistics (total samples, anomaly count, percentage)
    2. Anomaly breakdown by vehicle
    3. Most common anomaly patterns (which sensors show extreme values)
    4. Time-based patterns (which hours have most anomalies)
    5. Recommendations for maintenance or investigation
    
    Return a formatted text report.
    """


def main():
    """Main analysis pipeline"""
    print("=" * 70)
    print("AUTOMOBILE SENSOR ANOMALY DETECTION")
    print("=" * 70)
    
    # Load data using AI function
    print("\nLoading sensor data...")
    data = load_sensor_data()
    
    # Detect anomalies
    print("\n1. Detecting anomalies with Isolation Forest...")
    results = detect_anomalies(data)
    anomaly_count = results['is_anomaly'].sum()
    print(f"✓ Detected {anomaly_count} anomalies ({anomaly_count/len(results)*100:.1f}%)")
    
    # Create visualizations
    print("\n2. Creating visualizations...")
    visualizations = create_visualizations(results)
    print(f"✓ Generated {len(visualizations)} visualizations")
    
    # Save visualizations
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    for name, html_content in visualizations.items():
        viz_file = output_dir / f"{name}.html"
        viz_file.write_text(html_content)
        print(f"  - Saved {viz_file}")
    
    # Generate summary report
    print("\n3. Generating summary report...")
    report = generate_summary_report(results)
    
    report_file = output_dir / "analysis_report.txt"
    report_file.write_text(report)
    print(f"✓ Report saved to {report_file}")
    
    # Save results with anomaly labels
    results_file = output_dir / "sensor_data_with_anomalies.csv"
    results.to_csv(results_file, index=False)
    print(f"✓ Results saved to {results_file}")
    
    # Display report
    print("\n" + "=" * 70)
    print("ANALYSIS REPORT")
    print("=" * 70)
    print(report)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nView visualizations:")
    for name in visualizations.keys():
        print(f"  - open data/{name}.html")


if __name__ == "__main__":
    main()
