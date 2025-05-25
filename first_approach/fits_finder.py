#!/usr/bin/env python
"""
MAST Archive Search for Planet Nine Candidate

This script searches the MAST archives for observations around the predicted
position of the Planet Nine candidate discovered by Terry Long Phan's team.
Results are saved to a text file for later processing and downloading.

Usage:
  python mast_p9_search.py [--radius RADIUS] [--output OUTPUT]

Options:
  --radius RADIUS  Search radius in degrees [default: 0.2]
  --output OUTPUT  Output file name [default: p9_mast_results.txt]
"""

import argparse
import os
import json
from datetime import datetime

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astroquery.mast import Observations
from astroquery.irsa import Irsa

# Safe function to extract values, handling N/A and errors
def safe_get(row, column, default="Unknown"):
    """Safely extract a value from a row/column, handling N/A and exceptions."""
    try:
        value = row[column]
        if value is None or value == 'N/A':
            return default
        return value
    except Exception:
        return default

# Custom JSON encoder to handle non-serializable types
class NumpyEncoder(json.JSONEncoder):
    """Special JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, list, dict)):
            return list(obj)
        return super(NumpyEncoder, self).default(obj)

# Set up argument parser
parser = argparse.ArgumentParser(description='Search MAST archives for Planet Nine candidate')
parser.add_argument('--radius', type=float, default=0.2, help='Search radius in degrees')
parser.add_argument('--output', type=str, default='p9_mast_results.txt', help='Output file name')
args = parser.parse_args()

# Known positions of the Planet Nine candidate
POSITIONS = {
    "iras": {"ra": 35.74075, "dec": -48.5125, "year": 1983},
    "akari": {"ra": 35.18379, "dec": -49.2135, "year": 2006},
}

# Current date and year
current_date = datetime.now()
current_year = current_date.year

# Calculate predicted 2025 position
def calculate_position(year):
    """Calculate the predicted position for a given year based on known detections."""
    iras = POSITIONS["iras"]
    akari = POSITIONS["akari"]
    
    # Calculate rate of change
    years_diff = akari["year"] - iras["year"]
    ra_rate = (akari["ra"] - iras["ra"]) / years_diff
    dec_rate = (akari["dec"] - iras["dec"]) / years_diff
    
    # Calculate projected position
    years_since_akari = year - akari["year"]
    ra = akari["ra"] + (ra_rate * years_since_akari)
    dec = akari["dec"] + (dec_rate * years_since_akari)
    
    return {"ra": ra, "dec": dec, "year": year}

# Calculate position for search
predicted = calculate_position(current_year)
print(f"Predicted position for {current_year}: RA = {predicted['ra']:.5f}, Dec = {predicted['dec']:.5f}")

# Set up search coordinates and radius
target_coords = SkyCoord(predicted["ra"], predicted["dec"], unit="deg")
search_radius = args.radius * u.deg

print(f"Searching MAST archives with {args.radius} degree radius...")

# Search for observations in MAST that cover this region
try:
    obs_table = Observations.query_region(target_coords, radius=search_radius)
    print(f"Found {len(obs_table)} observations in MAST")
except Exception as e:
    print(f"Error querying MAST: {e}")
    obs_table = None

# Initialize results dictionary
results = {
    "search_parameters": {
        "ra": predicted["ra"],
        "dec": predicted["dec"],
        "radius_deg": args.radius,
        "search_date": current_date.strftime("%Y-%m-%d"),
    },
    "known_positions": POSITIONS,
    "predicted_position": predicted,
    "mast_results": {
        "total_observations": len(obs_table) if obs_table is not None else 0,
        "observations": []
    },
    "irsa_results": {
        "iras_sources": [],
        "akari_sources": [],
        "wise_sources": []
    }
}

# Process MAST results if available
if obs_table is not None and len(obs_table) > 0:
    # Group by observatory/instrument - handle potential missing values
    try:
        # First try standard grouping
        observatories = obs_table.group_by('obs_collection')
        groups = zip(observatories.groups.keys, observatories.groups)
        
        # Print summary of results
        print("\nMASTER Results Summary:")
        for key, group in groups:
            collection = key['obs_collection'] if isinstance(key, dict) else key
            count = len(group)
            print(f"  {collection}: {count} observations")
            
            # Add to results - safely extract observation IDs
            example_ids = []
            try:
                for row in group[:5]:  # First 5 rows
                    obs_id = safe_get(row, 'obs_id')
                    if obs_id != "Unknown":
                        example_ids.append(str(obs_id))  # Convert to string to ensure JSON serializable
            except Exception:
                # If we can't extract example IDs, continue without them
                pass
                
            results["mast_results"]["observations"].append({
                "collection": str(collection),
                "count": int(count),
                "example_ids": example_ids
            })
    except Exception as e:
        print(f"Warning: Could not group observations: {e}")
        print("Processing observations without grouping")
        
        # Just count observations by collection type
        collection_counts = {}
        for row in obs_table:
            collection = str(safe_get(row, 'obs_collection'))
            if collection not in collection_counts:
                collection_counts[collection] = 0
            collection_counts[collection] += 1
        
        for collection, count in collection_counts.items():
            print(f"  {collection}: {count} observations")
            results["mast_results"]["observations"].append({
                "collection": collection,
                "count": count,
                "example_ids": []  # No example IDs in this case
            })
    
    # Look for any infrared observations - handle potential N/A values
    # Create a safer method to search collection names
    def contains_mission(row, mission):
        """Check if a row's obs_collection contains the mission name, handling errors."""
        try:
            collection = safe_get(row, 'obs_collection')
            return mission.lower() in str(collection).lower()
        except:
            return False
    
    # Create infrared observations list manually
    ir_obs = []
    for row in obs_table:
        if (contains_mission(row, 'WISE') or 
            contains_mission(row, 'Spitzer') or 
            contains_mission(row, 'JWST') or
            contains_mission(row, 'IRAS') or
            contains_mission(row, 'AKARI')):
            ir_obs.append(row)
    
    # Convert to Table if we found any
    if ir_obs:
        from astropy.table import Table
        ir_obs = Table(rows=ir_obs, names=obs_table.colnames)
    
    if len(ir_obs) > 0:
        print(f"\nFound {len(ir_obs)} infrared observations")
        
        # Safely extract collections and IDs
        collections = set()
        example_ids = []
        
        for row in ir_obs[:5]:  # First 5 rows
            collection = safe_get(row, 'obs_collection')
            if collection != "Unknown":
                collections.add(str(collection))  # Convert to string for JSON
            
            obs_id = safe_get(row, 'obs_id')
            if obs_id != "Unknown":
                example_ids.append(str(obs_id))  # Convert to string for JSON
        
        results["mast_results"]["infrared_observations"] = {
            "count": int(len(ir_obs)),
            "collections": list(collections),  # Convert set to list for JSON
            "example_ids": example_ids
        }
else:
    print("No observations found in MAST archives for this region.")

# Function to safely convert values to JSON serializable format
def make_json_safe(value):
    """Convert values to JSON serializable types."""
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    elif isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    elif isinstance(value, (np.ndarray,)):
        return list(value)
    elif hasattr(value, 'tolist'):
        return value.tolist()
    elif hasattr(value, '__iter__') and not isinstance(value, (str, bytes, list, dict)):
        return list(value)
    elif isinstance(value, (datetime, Time)):
        return str(value)
    return value

# Function to process catalog data in a standardized way
def process_catalog_data(table, catalog_type, limit=None):
    """Process catalog data and return standardized records."""
    records = []
    
    if table is None or len(table) == 0:
        return records
    
    # Print column names to help with debugging
    print(f"{catalog_type} catalog columns: {table.colnames}")
    
    # Process only the first 'limit' rows if specified
    rows_to_process = table[:limit] if limit else table
    
    for i, src in enumerate(rows_to_process):
        source_data = {}
        
        # Add all available columns to the results
        for col in table.colnames:
            try:
                val = src[col]
                if val is not None and val != 'N/A':
                    # Convert to basic Python types for JSON serialization
                    source_data[col] = make_json_safe(val)
            except Exception as e:
                # Just skip this column if we can't process it
                pass
        
        # Ensure we have at least some basic identification based on catalog type
        if catalog_type == "IRAS":
            if 'name' not in source_data and 'ra' in source_data and 'dec' in source_data:
                source_data['name'] = f"IRAS_{i}_{source_data['ra']:.3f}_{source_data['dec']:.3f}"
            elif 'name' not in source_data:
                source_data['name'] = f"IRAS_{i}"
        
        elif catalog_type == "AKARI":
            if 'name' not in source_data and 'ra' in source_data and 'dec' in source_data:
                source_data['name'] = f"AKARI_{i}_{source_data['ra']:.3f}_{source_data['dec']:.3f}"
            elif 'name' not in source_data:
                source_data['name'] = f"AKARI_{i}"
        
        elif catalog_type == "WISE":
            if 'designation' not in source_data and 'ra' in source_data and 'dec' in source_data:
                source_data['designation'] = f"WISE_{i}_{source_data['ra']:.3f}_{source_data['dec']:.3f}"
            elif 'designation' not in source_data:
                source_data['designation'] = f"WISE_{i}"
        
        records.append(source_data)
    
    return records

# Search IRSA catalogs
print("\nSearching IRSA catalogs...")

# Search IRAS Point Source Catalog - using the correct catalog name
try:
    # From the IRSA catalogs list, we know the correct name is "iraspsc"
    print(f"Querying IRAS catalog: iraspsc")
    iras_table = Irsa.query_region(target_coords, catalog="iraspsc", radius=search_radius)
    print(f"Found {len(iras_table)} IRAS sources")
    
    # Process IRAS data if found
    results["irsa_results"]["iras_sources"] = process_catalog_data(iras_table, "IRAS")
    
except Exception as e:
    print(f"Error querying IRAS catalog: {e}")
    # Try alternative IRAS catalog
    try:
        print(f"Trying alternative IRAS catalog: irasfsc")
        iras_table = Irsa.query_region(target_coords, catalog="irasfsc", radius=search_radius)
        print(f"Found {len(iras_table)} IRAS sources")
        # Process IRAS data if found
        results["irsa_results"]["iras_sources"] = process_catalog_data(iras_table, "IRAS")
    except Exception as e2:
        print(f"Error querying alternative IRAS catalog: {e2}")

# Search AKARI - using the correct catalog name
try:
    # From the IRSA catalogs list, we know the correct name is "akari_fis"
    print(f"Querying AKARI catalog: akari_fis")
    akari_table = Irsa.query_region(target_coords, catalog="akari_fis", radius=search_radius)
    print(f"Found {len(akari_table)} AKARI sources")
    
    # Process AKARI data if found
    results["irsa_results"]["akari_sources"] = process_catalog_data(akari_table, "AKARI")
    
except Exception as e:
    print(f"Error querying AKARI catalog: {e}")
    # Try alternative AKARI catalog
    try:
        print(f"Trying alternative AKARI catalog: akari_irc")
        akari_table = Irsa.query_region(target_coords, catalog="akari_irc", radius=search_radius)
        print(f"Found {len(akari_table)} AKARI sources")
        # Process AKARI data if found
        results["irsa_results"]["akari_sources"] = process_catalog_data(akari_table, "AKARI")
    except Exception as e2:
        print(f"Error querying alternative AKARI catalog: {e2}")

# Search WISE - using the correct catalog name
try:
    # From the IRSA catalogs list, we know to use "allwise_p3as_psd"
    print(f"Querying WISE catalog: allwise_p3as_psd")
    wise_table = Irsa.query_region(target_coords, catalog="allwise_p3as_psd", radius=search_radius)
    print(f"Found {len(wise_table)} WISE sources")
    
    # Process WISE data if found - limit to 10 sources to avoid huge files
    if len(wise_table) > 0:
        results["irsa_results"]["wise_count"] = int(len(wise_table))
        results["irsa_results"]["wise_sources"] = process_catalog_data(wise_table, "WISE", limit=10)
    
except Exception as e:
    print(f"Error querying WISE catalog: {e}")
    # Try alternative WISE catalog
    try:
        print(f"Trying alternative WISE catalog: catwise_2020")
        wise_table = Irsa.query_region(target_coords, catalog="catwise_2020", radius=search_radius)
        print(f"Found {len(wise_table)} WISE sources")
        # Process WISE data if found - limit to 10 sources to avoid huge files
        if len(wise_table) > 0:
            results["irsa_results"]["wise_count"] = int(len(wise_table))
            results["irsa_results"]["wise_sources"] = process_catalog_data(wise_table, "WISE", limit=10)
    except Exception as e2:
        print(f"Error querying alternative WISE catalog: {e2}")

# Generate a list of potential data products to download
products_to_download = []

if obs_table is not None and len(obs_table) > 0 and len(ir_obs) > 0:
    # Prioritize infrared observations first
    for i, obs in enumerate(ir_obs):
        if i < 10:  # Limit to first 10 to avoid huge downloads
            obs_id = safe_get(obs, 'obs_id')
            obs_collection = safe_get(obs, 'obs_collection')
            
            # Determine priority based on collection
            priority = "medium"
            if any(mission in str(obs_collection).upper() for mission in ["WISE", "IRAS", "AKARI"]):
                priority = "high"
            elif "Spitzer" in str(obs_collection):
                priority = "high"
                
            products_to_download.append({
                "obs_id": str(obs_id),
                "obs_collection": str(obs_collection),
                "target_name": str(safe_get(obs, 'target_name')),
                "dataproduct_type": str(safe_get(obs, 'dataproduct_type')),
                "t_min": str(safe_get(obs, 't_min')),
                "t_max": str(safe_get(obs, 't_max')),
                "priority": priority
            })

# Test JSON serialization before saving to file
try:
    # Test that the results can be JSON serialized
    json_test = json.dumps(results, cls=NumpyEncoder)
    print("JSON serialization test passed")
except Exception as e:
    print(f"Error testing JSON serialization: {e}")
    
    # Alternative approach - create a clean copy with only basic types
    print("Attempting to sanitize results for JSON...")
    
    def sanitize_for_json(obj):
        if isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_for_json(item) for item in obj]
        elif isinstance(obj, (int, float, bool, str, type(None))):
            return obj
        else:
            return str(obj)  # Convert anything else to string
    
    results = sanitize_for_json(results)

# Save results to output file
try:
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
except Exception as e:
    print(f"Error saving results to {args.output}: {e}")
    
    # Try a fallback approach with minimal data
    print("Trying fallback approach to save results...")
    try:
        minimal_results = {
            "search_parameters": {
                "ra": float(predicted["ra"]),
                "dec": float(predicted["dec"]),
                "radius_deg": float(args.radius),
                "search_date": current_date.strftime("%Y-%m-%d"),
            },
            "known_positions": {
                "iras": {"ra": 35.74075, "dec": -48.5125, "year": 1983},
                "akari": {"ra": 35.18379, "dec": -49.2135, "year": 2006},
            },
            "predicted_position": {
                "ra": float(predicted["ra"]),
                "dec": float(predicted["dec"]),
                "year": int(predicted["year"]),
            },
        }
        
        with open(args.output + ".min.txt", 'w') as f:
            json.dump(minimal_results, f, indent=2)
        print(f"Saved minimal results to {args.output}.min.txt")
    except Exception as e2:
        print(f"Error saving minimal results: {e2}")

# Create a separate download manifest file
download_file = f"p9_download_manifest_{current_date.strftime('%Y%m%d')}.txt"
try:
    with open(download_file, 'w') as f:
        f.write("# Planet Nine Candidate - Download Manifest\n")
        f.write(f"# Generated: {current_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Target: RA = {predicted['ra']:.5f}, Dec = {predicted['dec']:.5f}\n\n")
        
        f.write("# MAST Observations to Download\n")
        for product in products_to_download:
            f.write(f"{product['obs_id']},{product['obs_collection']},{product['priority']}\n")
        
        # Add IRSA catalog sources if available
        f.write("\n# IRSA Catalog Sources\n")
        
        if results["irsa_results"]["iras_sources"]:
            f.write("# IRAS Sources\n")
            for src in results["irsa_results"]["iras_sources"]:
                name = src.get('name', 'Unknown')
                ra = src.get('ra', 0)
                dec = src.get('dec', 0)
                f.write(f"IRAS,{name},{ra},{dec}\n")
        
        if results["irsa_results"]["akari_sources"]:
            f.write("# AKARI Sources\n")
            for src in results["irsa_results"]["akari_sources"]:
                name = src.get('name', 'Unknown')
                ra = src.get('ra', 0)
                dec = src.get('dec', 0)
                f.write(f"AKARI,{name},{ra},{dec}\n")
        
        if results["irsa_results"]["wise_sources"]:
            f.write("# WISE Sources\n")
            for src in results["irsa_results"]["wise_sources"]:
                name = src.get('designation', 'Unknown')
                ra = src.get('ra', 0)
                dec = src.get('dec', 0)
                f.write(f"WISE,{name},{ra},{dec}\n")
except Exception as e:
    print(f"Error writing download manifest: {e}")

print(f"\nResults saved to {args.output}")
print(f"Download manifest saved to {download_file}")
print("\nNext steps:")
print("1. Review the results file to identify promising candidates")
print("2. Use the download manifest to retrieve data for analysis")
print("3. Run shift-and-stack analysis on the downloaded data")
