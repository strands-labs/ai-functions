"""
Automobile Sensor Data Generator

Generates synthetic automobile sensor data with realistic patterns and anomalies.
"""

import pandas as pd
from ai_functions import ai_function
from ai_functions.types import AIFunctionConfig
from strands import tool
from typing import Dict


# Configuration for data science operations
DATA_CONFIG = AIFunctionConfig(
    code_executor_additional_imports=[
        "pandas.*", 
        "numpy.*",
        "os.*"
    ],
    code_execution_mode="local"
)


@tool
def get_vehicle_specs(vehicle_id: str) -> Dict[str, float]:
    """
    Get vehicle specifications for realistic sensor data generation.
    
    Returns specifications that affect sensor readings for different vehicle types.
    
    Args:
        vehicle_id: Vehicle identifier (e.g., 'V001', 'V002', etc.)
        
    Returns:
        Dictionary with vehicle specifications including:
        - max_speed_kmh: Maximum speed capability
        - max_rpm: Maximum engine RPM (redline)
        - idle_rpm: Idle RPM
        - optimal_rpm_range: [min, max] for optimal driving
        - fuel_efficiency: Liters per 100km at cruise
        - engine_displacement: Engine size in liters
        - weight_kg: Vehicle weight
        - max_brake_pressure: Maximum brake pressure
        - tire_size: Tire size affecting pressure ranges
    """
    # Define different vehicle types with realistic specs
    vehicle_specs = {
        'V001': {  # Compact Sedan
            'vehicle_type': 'Compact Sedan',
            'max_speed_kmh': 180.0,
            'max_rpm': 6500.0,
            'idle_rpm': 700.0,
            'optimal_rpm_min': 1500.0,
            'optimal_rpm_max': 3500.0,
            'fuel_efficiency_l_per_100km': 6.5,
            'engine_displacement_l': 1.6,
            'weight_kg': 1300.0,
            'max_brake_pressure_bar': 80.0,
            'tire_pressure_psi': 32.0,
            'normal_temp_celsius': 90.0
        },
        'V002': {  # SUV
            'vehicle_type': 'SUV',
            'max_speed_kmh': 160.0,
            'max_rpm': 6000.0,
            'idle_rpm': 650.0,
            'optimal_rpm_min': 1800.0,
            'optimal_rpm_max': 3800.0,
            'fuel_efficiency_l_per_100km': 10.5,
            'engine_displacement_l': 3.0,
            'weight_kg': 2100.0,
            'max_brake_pressure_bar': 100.0,
            'tire_pressure_psi': 35.0,
            'normal_temp_celsius': 92.0
        },
        'V003': {  # Sports Car
            'vehicle_type': 'Sports Car',
            'max_speed_kmh': 250.0,
            'max_rpm': 8000.0,
            'idle_rpm': 900.0,
            'optimal_rpm_min': 2000.0,
            'optimal_rpm_max': 5000.0,
            'fuel_efficiency_l_per_100km': 12.0,
            'engine_displacement_l': 4.0,
            'weight_kg': 1500.0,
            'max_brake_pressure_bar': 120.0,
            'tire_pressure_psi': 36.0,
            'normal_temp_celsius': 95.0
        },
        'V004': {  # Electric Vehicle
            'vehicle_type': 'Electric Vehicle',
            'max_speed_kmh': 200.0,
            'max_rpm': 15000.0,  # Electric motors spin faster
            'idle_rpm': 0.0,  # No idle for electric
            'optimal_rpm_min': 3000.0,
            'optimal_rpm_max': 8000.0,
            'fuel_efficiency_l_per_100km': 0.0,  # Uses kWh instead
            'engine_displacement_l': 0.0,  # No engine displacement
            'weight_kg': 1800.0,
            'max_brake_pressure_bar': 90.0,
            'tire_pressure_psi': 42.0,  # Higher for efficiency
            'normal_temp_celsius': 65.0  # Battery temp, not engine
        },
        'V005': {  # Pickup Truck
            'vehicle_type': 'Pickup Truck',
            'max_speed_kmh': 170.0,
            'max_rpm': 5500.0,
            'idle_rpm': 600.0,
            'optimal_rpm_min': 1500.0,
            'optimal_rpm_max': 3200.0,
            'fuel_efficiency_l_per_100km': 13.0,
            'engine_displacement_l': 5.0,
            'weight_kg': 2400.0,
            'max_brake_pressure_bar': 110.0,
            'tire_pressure_psi': 40.0,
            'normal_temp_celsius': 88.0
        }
    }
    
    # Return specs for the requested vehicle, or default to V001 if not found
    return vehicle_specs.get(vehicle_id, vehicle_specs['V001'])


@ai_function(config=DATA_CONFIG)
def validate_sensor_data(result: pd.DataFrame) -> None:
    """
    Validate the generated sensor data for correctness.
    
    The result DataFrame has {result.shape[0]} rows and {result.shape[1]} columns.
    
    Perform these validation checks and raise ValueError if any fail:
    
    1. Check all required columns exist:
       ['timestamp', 'vehicle_id', 'speed_kmh', 'rpm', 'throttle_position',
        'brake_pressure', 'steering_angle', 'fuel_consumption_lph',
        'engine_temp_celsius', 'battery_voltage', 'tire_pressure_fl',
        'tire_pressure_fr', 'tire_pressure_rl', 'tire_pressure_rr',
        'ambient_temp_celsius', 'coolant_temp_celsius', 'oil_pressure_bar',
        'transmission_temp_celsius', 'latitude', 'longitude']
    
    2. Check data types:
       - timestamp must be datetime64 type (use pd.api.types.is_datetime64_any_dtype)
       - vehicle_id must be string/object type (use pd.api.types.is_string_dtype)
       - All other columns must be numeric (use pd.api.types.is_numeric_dtype)
    
    3. Check for NaN values - data should not contain any NaN values
    
    4. Check minimum number of rows - must have at least 100 samples
    """


@ai_function(config=DATA_CONFIG, post_conditions=[validate_sensor_data], tools=[get_vehicle_specs])
def generate_sensor_data(num_samples: int = 1000, anomaly_rate: float = 0.05) -> pd.DataFrame:
    """
    Generate synthetic automobile sensor data with realistic patterns and anomalies.
    
    IMPORTANT: Use the get_vehicle_specs tool to get realistic specifications for each vehicle (V001-V005).
    This will provide vehicle-specific parameters like max_speed, max_rpm, fuel_efficiency, etc.
    Use these specs to generate realistic sensor data for each vehicle type.
    
    Create a dataset with {num_samples} samples including these sensors:
    
    TEMPORAL DATA:
    - timestamp: datetime values spanning 24 hours
    - vehicle_id: 5 different vehicles (V001-V005)
    
    MOTION SENSORS (use vehicle specs for realistic ranges):
    - speed_kmh: vehicle speed (0 to max_speed from specs, normal: 40-120)
    - rpm: engine RPM (idle_rpm to max_rpm from specs, normal: optimal_rpm_min to optimal_rpm_max)
    - throttle_position: 0-100% (normal: 10-70%)
    - brake_pressure: 0 to max_brake_pressure from specs (normal: 0-30)
    - steering_angle: -540 to 540 degrees (normal: -90 to 90)
    
    FUEL & POWER SENSORS (use vehicle specs):
    - fuel_consumption_lph: based on fuel_efficiency from specs
    - battery_voltage: volts (11-15V, normal: 12.5-14.5V)
    
    TEMPERATURE SENSORS (use vehicle specs for normal_temp):
    - engine_temp_celsius: 70-110°C (normal: use normal_temp_celsius from specs)
    - ambient_temp_celsius: -10 to 45°C (normal: 15-30°C)
    - coolant_temp_celsius: 70-110°C (normal: similar to engine temp)
    - transmission_temp_celsius: 70-110°C (normal: 80-95°C)
    
    PRESSURE SENSORS (use vehicle specs):
    - tire_pressure_fl: front left tire PSI (use tire_pressure_psi from specs ± 8)
    - tire_pressure_fr: front right tire PSI (use tire_pressure_psi from specs ± 8)
    - tire_pressure_rl: rear left tire PSI (use tire_pressure_psi from specs ± 8)
    - tire_pressure_rr: rear right tire PSI (use tire_pressure_psi from specs ± 8)
    - oil_pressure_bar: 1-6 bar (normal: 2-4.5)
    
    GPS SENSORS:
    - latitude: GPS coordinate (realistic range: 37.0 to 38.0)
    - longitude: GPS coordinate (realistic range: -122.5 to -121.5)
    
    ANOMALIES (inject approximately {anomaly_rate*100}% anomalies):
    - Sudden speed spikes (speed > 90% of max_speed AND rpm > 85% of max_rpm)
    - Engine overheating (engine_temp > normal_temp + 15°C)
    - Low tire pressure (any tire < tire_pressure_psi - 4)
    - Battery issues (voltage < 12V or > 14.8V)
    - Simultaneous hard braking and acceleration (brake_pressure > 60 AND throttle > 70)
    - Oil pressure problems (oil_pressure < 1.5 or > 5.5 bar)
    - Transmission overheating (transmission_temp > 105°C)
    
    Return the pandas DataFrame with all columns and proper data types.
    """


def save_sensor_data(data: pd.DataFrame, file_path: str = "data/sensor_data.csv") -> None:
    """
    Save the sensor data DataFrame to a CSV file.
    
    Args:
        data: DataFrame containing sensor data
        file_path: Path where to save the CSV file
    """
    import os
    
    # Create directory if it doesn't exist
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    # Save to CSV
    data.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")
    
    # Verify file was saved
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Failed to save data file: {file_path}")


if __name__ == "__main__":
    print("=" * 70)
    print("AUTOMOBILE SENSOR DATA GENERATOR")
    print("=" * 70)
    
    # Generate data
    print(f"\nGenerating 1000 samples with 5% anomaly rate...")
    data = generate_sensor_data(num_samples=1000, anomaly_rate=0.05)
    
    print(f"✓ Generated {len(data)} samples")
    print(f"✓ Vehicles: {data['vehicle_id'].nunique()}")
    print(f"✓ Columns: {len(data.columns)}")
    
    # Save data to file
    print(f"\nSaving data to file...")
    save_sensor_data(data)
    
    # Display sample
    print("\nSample data (first 5 rows):")
    print(data.head())
    
    print("\nData types:")
    print(data.dtypes)
    
    print("\n" + "=" * 70)
    print("DATA GENERATION COMPLETE")
    print("=" * 70)
