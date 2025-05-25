#!/usr/bin/env python
"""
SkyView-based WISE Data Downloader

This script uses SkyView to download WISE images for the Planet Nine candidate region.
It doesn't rely on SHA or IRSA direct access which was giving 404 errors.

Usage:
  python skyview_download.py [--manifest MANIFEST] [--output-dir OUTPUT_DIR]

Options:
  --manifest MANIFEST    Path to the download manifest file [default: p9_download_manifest_*.txt]
  --output-dir DIR       Directory to store downloaded files [default: p9_data]
"""

import os
import sys
import glob
import argparse
import time
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.ipac.irsa import Irsa
from astroquery.skyview import SkyView
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

# Suppress warning messages
warnings.filterwarnings("ignore")

# Set up argument parser
parser = argparse.ArgumentParser(description='Download WISE data for Planet Nine search')
parser.add_argument('--manifest', type=str, default='', help='Path to the download manifest file')
parser.add_argument('--output-dir', type=str, default='p9_data', help='Directory to store downloaded files')
args = parser.parse_args()

# Find latest manifest file if not specified
if not args.manifest:
    manifest_files = glob.glob('p9_download_manifest_*.txt')
    if not manifest_files:
        print("No manifest files found. Run the search script first.")
        sys.exit(1)
    args.manifest = sorted(manifest_files)[-1]

print(f"Using manifest file: {args.manifest}")

# Create output directories
os.makedirs(args.output_dir, exist_ok=True)
wise_dir = os.path.join(args.output_dir, 'wise')
os.makedirs(wise_dir, exist_ok=True)
metadata_dir = os.path.join(args.output_dir, 'metadata')
os.makedirs(metadata_dir, exist_ok=True)

# Parse manifest file to get WISE sources
wise_sources = []

with open(args.manifest, 'r') as f:
    section = None
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            if "WISE Sources" in line:
                section = "WISE"
            continue
        
        if section == "WISE":
            parts = line.split(',')
            if len(parts) >= 4:
                wise_sources.append({
                    'catalog': parts[0],
                    'name': parts[1],
                    'ra': float(parts[2]),
                    'dec': float(parts[3])
                })

print(f"Found {len(wise_sources)} WISE sources")

# Track download stats
download_stats = {
    'total_sources': len(wise_sources),
    'successful_downloads': 0,
    'failed_downloads': 0,
    'catalog_queries': 0,
    'total_files': 0
}

# Create source table for future reference
source_table = Table()
source_table['name'] = [src['name'] for src in wise_sources]
source_table['ra'] = [src['ra'] for src in wise_sources]
source_table['dec'] = [src['dec'] for src in wise_sources]

# Save source table
source_table_path = os.path.join(metadata_dir, 'wise_sources.csv')
source_table.write(source_table_path, format='ascii.csv', overwrite=True)
print(f"Saved source catalog to {source_table_path}")

# Function to query WISE catalog for a source
def query_wise_catalog(ra, dec, radius=10*u.arcsec):
    """Query the WISE catalog for a source."""
    coords = SkyCoord(ra, dec, unit='deg')
    
    # First try newer catalogs, then fall back to older ones
    catalogs = ['allwise_p3as_psd', 'catwise_2020', 'unwise_2019']
    
    for catalog in catalogs:
        try:
            print(f"  Querying {catalog}...")
            result = Irsa.query_region(coords, catalog=catalog, radius=radius)
            if len(result) > 0:
                print(f"  Found {len(result)} matches in {catalog}")
                return result, catalog
        except Exception as e:
            print(f"  Error querying {catalog}: {e}")
    
    # If all catalogs fail, return empty table
    return Table(), None

# Function to download WISE images using SkyView
def download_wise_images(ra, dec, name, output_dir):
    """Download WISE images using SkyView."""
    try:
        # Define SkyView parameters
        coords = SkyCoord(ra, dec, unit='deg')
        surveys = ['WISE 3.4', 'WISE 4.6', 'WISE 12', 'WISE 22']
        pixels = 300
        
        print(f"  Downloading images using SkyView...")
        images = SkyView.get_images(position=coords, survey=surveys, pixels=pixels)
        
        # Count successful downloads
        successful = 0
        
        # Save each image
        for i, (image, survey) in enumerate(zip(images, surveys)):
            # Skip empty images
            if image is None or len(image) == 0:
                print(f"  No data for {survey}")
                continue
                
            # Extract the data
            data = image[0].data
            header = image[0].header
            
            if data is None or np.all(np.isnan(data)):
                print(f"  Empty data for {survey}")
                continue
            
            # Create sanitized filename
            safe_name = name.replace(' ', '_').replace('/', '_').replace('\\', '_')
            filename = f"{safe_name}_{survey.replace(' ', '_')}.fits"
            filepath = os.path.join(output_dir, filename)
            
            # Save the FITS file
            try:
                image.writeto(filepath, overwrite=True)
                print(f"  Saved {survey} to {filename}")
                successful += 1
                download_stats['total_files'] += 1
                
                # Also create a PNG preview
                png_filename = f"{safe_name}_{survey.replace(' ', '_')}.png"
                png_filepath = os.path.join(output_dir, png_filename)
                
                # Create a simple visualization
                plt.figure(figsize=(6, 6))
                plt.imshow(data, origin='lower', cmap='viridis')
                plt.colorbar(label='Flux')
                plt.title(f"{name} - {survey}")
                plt.tight_layout()
                plt.savefig(png_filepath, dpi=100)
                plt.close()
                
                print(f"  Created preview image: {png_filename}")
                download_stats['total_files'] += 1
            except Exception as img_err:
                print(f"  Error saving image: {img_err}")
        
        return successful
    
    except Exception as e:
        print(f"  Error with SkyView: {e}")
        return 0

# Download custom DSS image for comparison
def download_dss_image(ra, dec, name, output_dir):
    """Download DSS image for comparison."""
    try:
        # Define SkyView parameters
        coords = SkyCoord(ra, dec, unit='deg')
        survey = 'DSS'
        pixels = 300
        
        print(f"  Downloading DSS image for comparison...")
        images = SkyView.get_images(position=coords, survey=survey, pixels=pixels)
        
        # Skip empty images
        if images is None or len(images) == 0 or images[0] is None:
            print(f"  No DSS data available")
            return 0
            
        # Extract the data
        data = images[0][0].data
        header = images[0][0].header
        
        if data is None or np.all(np.isnan(data)):
            print(f"  Empty DSS data")
            return 0
        
        # Create sanitized filename
        safe_name = name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        filename = f"{safe_name}_DSS.fits"
        filepath = os.path.join(output_dir, filename)
        
        # Save the FITS file
        images[0].writeto(filepath, overwrite=True)
        print(f"  Saved DSS to {filename}")
        download_stats['total_files'] += 1
        
        # Also create a PNG preview
        png_filename = f"{safe_name}_DSS.png"
        png_filepath = os.path.join(output_dir, png_filename)
        
        # Create a simple visualization
        plt.figure(figsize=(6, 6))
        plt.imshow(data, origin='lower', cmap='gray_r')
        plt.colorbar(label='Flux')
        plt.title(f"{name} - DSS")
        plt.tight_layout()
        plt.savefig(png_filepath, dpi=100)
        plt.close()
        
        print(f"  Created DSS preview image: {png_filename}")
        download_stats['total_files'] += 1
        return 1
    
    except Exception as e:
        print(f"  Error with DSS download: {e}")
        return 0

# Create a comparison image showing all bands
def create_comparison_image(ra, dec, name, output_dir):
    """Create a comparison image with all bands."""
    try:
        # Define SkyView parameters
        coords = SkyCoord(ra, dec, unit='deg')
        surveys = ['DSS', 'WISE 3.4', 'WISE 4.6', 'WISE 12', 'WISE 22']
        pixels = 200
        
        print(f"  Creating comparison image...")
        images = []
        
        # Get all images
        for survey in surveys:
            try:
                img = SkyView.get_images(position=coords, survey=survey, pixels=pixels)
                if img and len(img) > 0 and img[0] is not None:
                    images.append((img[0][0].data, survey))
                else:
                    print(f"  No data for {survey} in comparison")
            except Exception as survey_err:
                print(f"  Error getting {survey} for comparison: {survey_err}")
        
        if not images:
            print("  No images available for comparison")
            return 0
        
        # Create a multi-panel figure
        n_images = len(images)
        cols = min(3, n_images)
        rows = (n_images + cols - 1) // cols
        
        plt.figure(figsize=(4*cols, 4*rows))
        
        for i, (data, survey) in enumerate(images):
            plt.subplot(rows, cols, i+1)
            
            # Skip empty data
            if data is None or np.all(np.isnan(data)):
                plt.text(0.5, 0.5, f"No {survey} data", 
                         ha='center', va='center', transform=plt.gca().transAxes)
                continue
                
            # Choose colormap based on survey
            cmap = 'gray_r' if survey == 'DSS' else 'viridis'
            
            plt.imshow(data, origin='lower', cmap=cmap)
            plt.colorbar(label='Flux')
            plt.title(survey)
        
        plt.suptitle(f"{name} - Multi-band Comparison", fontsize=16)
        plt.tight_layout()
        
        # Save the comparison image
        safe_name = name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        comp_filename = f"{safe_name}_comparison.png"
        comp_filepath = os.path.join(output_dir, comp_filename)
        plt.savefig(comp_filepath, dpi=120)
        plt.close()
        
        print(f"  Created comparison image: {comp_filename}")
        download_stats['total_files'] += 1
        return 1
    
    except Exception as e:
        print(f"  Error creating comparison image: {e}")
        return 0

# Process each WISE source
for i, src in enumerate(wise_sources):
    try:
        print(f"Processing {i+1}/{len(wise_sources)}: {src['name']}")
        
        # Create source directory
        safe_name = src['name'].replace(' ', '_').replace('/', '_').replace('\\', '_')
        source_dir = os.path.join(wise_dir, safe_name)
        os.makedirs(source_dir, exist_ok=True)
        
        # Store results for this source
        source_results = {
            'name': src['name'],
            'ra': src['ra'],
            'dec': src['dec'],
            'catalog_matches': 0,
            'images_downloaded': 0,
            'download_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 1. Query WISE catalog data
        catalog_data, catalog_name = query_wise_catalog(src['ra'], src['dec'])
        
        if len(catalog_data) > 0:
            # Save catalog data
            catalog_file = os.path.join(source_dir, f"{safe_name}_catalog.csv")
            catalog_data.write(catalog_file, format='ascii.csv', overwrite=True)
            print(f"  Saved catalog data to {catalog_file}")
            
            source_results['catalog_matches'] = len(catalog_data)
            source_results['catalog_name'] = catalog_name
            download_stats['catalog_queries'] += 1
            download_stats['total_files'] += 1
        
        # 2. Download WISE images using SkyView
        wise_images = download_wise_images(src['ra'], src['dec'], src['name'], source_dir)
        source_results['images_downloaded'] += wise_images
        
        # 3. Download DSS image for comparison
        dss_image = download_dss_image(src['ra'], src['dec'], src['name'], source_dir)
        source_results['dss_downloaded'] = dss_image > 0
        
        # 4. Create comparison image
        comparison = create_comparison_image(src['ra'], src['dec'], src['name'], source_dir)
        source_results['comparison_created'] = comparison > 0
        
        # Save source results
        results_file = os.path.join(source_dir, f"{safe_name}_results.json")
        with open(results_file, 'w') as f:
            json.dump(source_results, f, indent=2)
        
        # Update download stats
        if source_results['images_downloaded'] > 0:
            download_stats['successful_downloads'] += 1
        else:
            download_stats['failed_downloads'] += 1
        
    except Exception as e:
        print(f"Error processing source {src['name']}: {e}")
        download_stats['failed_downloads'] += 1
    
    # Brief pause
    time.sleep(1)

# Alternative method: Try downloading the predicted 2025 location with a wider field of view
print("\nDownloading Planet Nine predicted position data...")

# Get the predicted 2025 position
p9_ra = 34.72369
p9_dec = -49.79259
p9_coords = SkyCoord(p9_ra, p9_dec, unit='deg')

# Create P9 directory
p9_dir = os.path.join(args.output_dir, 'p9_predicted')
os.makedirs(p9_dir, exist_ok=True)

# Download larger field WISE images centered on the predicted location
print(f"Downloading field around predicted position: RA={p9_ra}, Dec={p9_dec}")

# Use a larger field of view (0.5 degrees)
large_pixels = 800
surveys = ['DSS', 'WISE 3.4', 'WISE 4.6', 'WISE 12', 'WISE 22']

for survey in surveys:
    try:
        print(f"  Downloading {survey} field...")
        images = SkyView.get_images(position=p9_coords, survey=survey, pixels=large_pixels, width=0.5*u.deg)
        
        if images and len(images) > 0 and images[0] is not None:
            # Save FITS file
            fits_filename = f"P9_predicted_{survey.replace(' ', '_')}.fits"
            fits_filepath = os.path.join(p9_dir, fits_filename)
            images[0].writeto(fits_filepath, overwrite=True)
            print(f"  Saved {survey} field to {fits_filename}")
            download_stats['total_files'] += 1
            
            # Create PNG preview
            data = images[0][0].data
            if data is not None and not np.all(np.isnan(data)):
                png_filename = f"P9_predicted_{survey.replace(' ', '_')}.png"
                png_filepath = os.path.join(p9_dir, png_filename)
                
                plt.figure(figsize=(10, 10))
                plt.imshow(data, origin='lower', cmap='viridis')
                plt.colorbar(label='Flux')
                plt.title(f"P9 Predicted Position - {survey}")
                
                # Mark the exact predicted position
                # Convert to pixel coordinates - assuming center of image
                center_x = data.shape[1] // 2
                center_y = data.shape[0] // 2
                plt.plot(center_x, center_y, 'rx', markersize=10)
                plt.annotate('P9 predicted', (center_x, center_y), xytext=(10, 10), 
                             textcoords='offset points', color='red', fontsize=12)
                
                plt.tight_layout()
                plt.savefig(png_filepath, dpi=150)
                plt.close()
                
                print(f"  Created preview image: {png_filename}")
                download_stats['total_files'] += 1
    
    except Exception as e:
        print(f"  Error downloading {survey} field: {e}")

# Create final download summary
download_stats['completion_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

summary_path = os.path.join(args.output_dir, 'skyview_download_summary.json')
with open(summary_path, 'w') as f:
    json.dump(download_stats, f, indent=2)

print("\nDownload summary:")
print(f"Total sources processed: {download_stats['total_sources']}")
print(f"Successful downloads: {download_stats['successful_downloads']}")
print(f"Failed downloads: {download_stats['failed_downloads']}")
print(f"Total files created: {download_stats['total_files']}")
print(f"Data saved to {args.output_dir}")
print(f"Summary saved to {summary_path}")

print("\nNext steps:")
print("1. Review the downloaded images in the 'wise' and 'p9_predicted' directories")
print("2. Run the shift-and-stack analysis using the downloaded FITS files")