#!/usr/bin/env python
"""
Advanced Shift-and-Stack Processor for Planet Nine Detection

This script implements an advanced shift-and-stack algorithm to search for the 
Planet Nine candidate in the downloaded WISE and other infrared data.

This version includes fixes for:
1. Missing observation years in the FITS files
2. NumPy array boolean evaluation error
3. Enhanced error handling throughout

Usage:
  python shift_stack_p9.py [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR] [--rate-range RANGE]

Options:
  --data-dir DIR       Directory containing downloaded data [default: p9_data]
  --output-dir DIR     Directory to store results [default: p9_results]
  --rate-range RANGE   Range factor to search around predicted rates [default: 0.2]
  --default-year YEAR  Default observation year to use if missing [default: 2010]
"""

import os
import sys
import glob
import argparse
import time
import json
import warnings
import re
from datetime import datetime
from pathlib import Path
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
from matplotlib.patches import Circle
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time
from astropy.stats import sigma_clipped_stats, SigmaClip, gaussian_fwhm_to_sigma
from astropy.visualization import (ZScaleInterval, ImageNormalize, 
                                  MinMaxInterval, SqrtStretch, 
                                  AsinhStretch, LogStretch)
from astropy.convolution import Gaussian2DKernel, convolve
from photutils.detection import DAOStarFinder, find_peaks
from photutils.segmentation import detect_sources
from scipy.ndimage import median_filter, gaussian_filter, shift
from scipy.signal import correlate2d
from skimage import exposure, transform, filters, feature
from skimage.feature import peak_local_max
import cv2 # OpenCV for additional image processing
import scipy.ndimage  # Ensure this is imported for the map_coordinates function

# Suppress warning messages
warnings.filterwarnings("ignore")

# Set up argument parser
parser = argparse.ArgumentParser(description='Shift-and-stack analysis for Planet Nine search')
parser.add_argument('--data-dir', type=str, default='p9_data', help='Directory containing downloaded data')
parser.add_argument('--output-dir', type=str, default='p9_results', help='Directory to store results')
parser.add_argument('--rate-range', type=float, default=0.2, help='Range factor to search around predicted rates')
parser.add_argument('--default-year', type=float, default=2010.0, help='Default observation year to use if missing')
parser.add_argument('--debug', action='store_true', help='Enable debug mode with additional output')
args = parser.parse_args()

# Enable debug mode
DEBUG = True  # Force debug mode on

# Constants - the known motion of Planet Nine from IRAS to AKARI observations
# These rates are critical for the shift-and-stack method
P9_RA_RATE = -0.024216  # deg/year
P9_DEC_RATE = -0.030478  # deg/year
P9_ANGULAR_VELOCITY = 2.06  # arcmin/year

# Reference positions
P9_IRAS_POSITION = {"ra": 35.74075, "dec": -48.5125, "year": 1983}
P9_AKARI_POSITION = {"ra": 35.18379, "dec": -49.2135, "year": 2006}
P9_PREDICTED_2025 = {"ra": 34.72369, "dec": -49.79259, "year": 2025}

# Default observation year if missing from metadata
DEFAULT_OBSERVATION_YEAR = args.default_year

# Create output directories
os.makedirs(args.output_dir, exist_ok=True)
plots_dir = os.path.join(args.output_dir, 'plots')
stacks_dir = os.path.join(args.output_dir, 'stacks')
sources_dir = os.path.join(args.output_dir, 'detected_sources')
debug_dir = os.path.join(args.output_dir, 'debug')
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(stacks_dir, exist_ok=True)
os.makedirs(sources_dir, exist_ok=True)
os.makedirs(debug_dir, exist_ok=True)

# Initialize analysis log
analysis_log = {
    "start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "p9_parameters": {
        "ra_rate": P9_RA_RATE,
        "dec_rate": P9_DEC_RATE,
        "angular_velocity": P9_ANGULAR_VELOCITY,
        "iras_position": P9_IRAS_POSITION,
        "akari_position": P9_AKARI_POSITION,
        "predicted_2025": P9_PREDICTED_2025
    },
    "data_dir": args.data_dir,
    "output_dir": args.output_dir,
    "rate_range": args.rate_range,
    "default_year": DEFAULT_OBSERVATION_YEAR,
    "files_processed": 0,
    "stacks_created": 0,
    "potential_detections": [],
    "debug_info": []
}

def debug_print(*args, **kwargs):
    """Debug print function that only prints if DEBUG is True."""
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)

def debug_save_array(arr, name, metadata=None):
    """Save an array to a FITS file for debugging."""
    if DEBUG and arr is not None:
        try:
            debug_file = os.path.join(debug_dir, f"{name}.fits")
            hdu = fits.PrimaryHDU(arr)
            if metadata:
                for key, value in metadata.items():
                    hdu.header[key] = value
            hdu.writeto(debug_file, overwrite=True)
            debug_print(f"Saved debug array to {debug_file}")
        except Exception as e:
            print(f"Error saving debug array {name}: {e}")

print("==== Advanced Shift-and-Stack Planet Nine Search ====")
print(f"Data directory: {args.data_dir}")
print(f"Output directory: {args.output_dir}")
print(f"P9 motion rates: RA={P9_RA_RATE:.6f} deg/yr, Dec={P9_DEC_RATE:.6f} deg/yr")
print(f"Rate search range: ±{args.rate_range*100:.0f}%")
print(f"Default observation year if missing: {DEFAULT_OBSERVATION_YEAR}")
print(f"Debug mode: {'ON' if DEBUG else 'OFF'}")

# Function to load all FITS files from a directory recursively
def find_fits_files(data_dir):
    """Find all FITS files in the data directory recursively."""
    fits_files = []
    
    # First check if the data_dir itself is a FITS file
    if data_dir.lower().endswith(('.fits', '.fit', '.fts')) and os.path.isfile(data_dir):
        fits_files.append(data_dir)
        debug_print(f"Input is a direct FITS file: {data_dir}")
        return fits_files
    
    # First check the p9_predicted directory specifically
    p9_dir = os.path.join(data_dir, 'p9_predicted')
    if os.path.exists(p9_dir):
        for ext in ['.fits', '.fit', '.fts']:
            files = glob.glob(os.path.join(p9_dir, f'*{ext}'))
            fits_files.extend(files)
    
    # If no files found in p9_predicted, search the main data directory
    if not fits_files and os.path.exists(data_dir):
        for ext in ['.fits', '.fit', '.fts']:
            files = glob.glob(os.path.join(data_dir, f'*{ext}'))
            fits_files.extend(files)
    
    # Then search recursively through all directories if still no files found
    if not fits_files:
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(('.fits', '.fit', '.fts')):
                    fits_files.append(os.path.join(root, file))
    
    # Debug output
    if DEBUG:
        debug_print(f"Found {len(fits_files)} FITS files:")
        for i, file in enumerate(fits_files[:5]):  # Show first 5 files
            debug_print(f"  {i+1}. {file}")
        if len(fits_files) > 5:
            debug_print(f"  ... and {len(fits_files)-5} more")
    
    return fits_files

# Function to extract year from filename
def extract_year_from_filename(filename):
    """Try to extract a year from the filename."""
    # Try different regex patterns to find a year
    year_patterns = [
        r'(\d{4})[-_](\d{2})[-_](\d{2})',  # YYYY-MM-DD or YYYY_MM_DD
        r'(\d{2})[-_](\d{2})[-_](\d{4})',  # DD-MM-YYYY or DD_MM_YYYY
        r'(\d{4})(\d{2})(\d{2})',          # YYYYMMDD
        r'y(\d{4})',                       # yYYYY
        r'(\d{4})',                        # Just find any 4-digit number
    ]
    
    for pattern in year_patterns:
        matches = re.findall(pattern, filename)
        if matches:
            if isinstance(matches[0], tuple):
                # If it's a tuple, extract the year part based on the pattern
                if pattern == r'(\d{2})[-_](\d{2})[-_](\d{4})':
                    year = int(matches[0][2])  # YYYY is the 3rd group
                else:
                    year = int(matches[0][0])  # YYYY is the 1st group
            else:
                # If it's not a tuple, it's the year itself
                year = int(matches[0])
            
            # Basic validation: only accept years between 1980 and current year
            current_year = datetime.now().year
            if 1980 <= year <= current_year:
                # Add a fraction to represent middle of the year
                return float(year) + 0.5
    
    return None

# Function to extract metadata from FITS file
def extract_fits_metadata(fits_file):
    """Extract metadata from a FITS file."""
    try:
        with fits.open(fits_file) as hdul:
            # Start with primary HDU
            header = hdul[0].header
            
            # Look for extension with image data if primary HDU has no data
            data_hdu = 0
            for i, hdu in enumerate(hdul):
                if hasattr(hdu, 'data') and hdu.data is not None and hdu.data.ndim >= 2:
                    data_hdu = i
                    header = hdu.header
                    break
            
            # Extract basic metadata
            metadata = {
                'file': fits_file,
                'filename': os.path.basename(fits_file),
                'shape': hdul[data_hdu].data.shape if hdul[data_hdu].data is not None else None,
                'data_hdu': data_hdu
            }
            
            # Try to get WCS
            try:
                wcs = WCS(header)
                metadata['has_wcs'] = True
            except Exception as e:
                metadata['has_wcs'] = False
                if DEBUG:
                    debug_print(f"WCS error in {fits_file}: {e}")
            
            # Extract common header keywords
            for key in ['TELESCOP', 'INSTRUME', 'DATE-OBS', 'DATE_OBS', 'FILTER', 
                        'EXPTIME', 'OBJECT', 'CRVAL1', 'CRVAL2', 'SURVEY']:
                if key in header:
                    metadata[key.lower()] = header[key]
            
            # Determine observation date
            if 'date-obs' in metadata:
                metadata['obs_date'] = metadata['date-obs']
            elif 'date_obs' in metadata:
                metadata['obs_date'] = metadata['date_obs']
            
            # Try to extract observation year
            metadata['obs_year'] = None
            
            # Method 1: Try to extract from date-obs
            if 'obs_date' in metadata:
                try:
                    date_obj = Time(metadata['obs_date'])
                    metadata['obs_year'] = date_obj.ymdhms.year + (date_obj.ymdhms.month-1)/12 + date_obj.ymdhms.day/365.25
                    debug_print(f"Extracted year from DATE-OBS: {metadata['obs_year']:.2f}")
                except:
                    # Fallback: try to extract year from string
                    try:
                        year_str = metadata['obs_date'].split('-')[0]
                        metadata['obs_year'] = float(year_str)
                        debug_print(f"Extracted year from DATE-OBS string: {metadata['obs_year']:.2f}")
                    except:
                        metadata['obs_year'] = None
            
            # Method 2: Check other potential date fields in header
            if metadata['obs_year'] is None:
                for key in ['DATE', 'OBS-DATE', 'OBSDATE', 'DATE_OBS', 'DATE-OBS', 'MJD-OBS']:
                    if key in header:
                        try:
                            if 'MJD' in key:
                                # Convert Modified Julian Date to year
                                mjd = float(header[key])
                                date_obj = Time(mjd, format='mjd')
                                metadata['obs_year'] = date_obj.ymdhms.year + (date_obj.ymdhms.month-1)/12 + date_obj.ymdhms.day/365.25
                                debug_print(f"Extracted year from {key}: {metadata['obs_year']:.2f}")
                                break
                            else:
                                date_str = header[key]
                                # Try to extract year from date string
                                if '-' in date_str:
                                    year_str = date_str.split('-')[0]
                                    if len(year_str) == 4 and year_str.isdigit():
                                        metadata['obs_year'] = float(year_str)
                                        debug_print(f"Extracted year from {key}: {metadata['obs_year']:.2f}")
                                        break
                        except:
                            continue
            
            # Method 3: Try to extract from filename
            if metadata['obs_year'] is None:
                year_from_filename = extract_year_from_filename(metadata['filename'])
                if year_from_filename is not None:
                    metadata['obs_year'] = year_from_filename
                    debug_print(f"Extracted year from filename: {metadata['obs_year']:.2f}")
            
            # Method 4: If all else fails, assign a default year
            if metadata['obs_year'] is None:
                metadata['obs_year'] = DEFAULT_OBSERVATION_YEAR
                debug_print(f"Using default year: {metadata['obs_year']:.2f}")
            
            # Determine wavelength/band from filename if not in header
            filename = metadata['filename'].lower()
            if 'wise' in filename:
                if '3.4' in filename or 'w1' in filename:
                    metadata['band'] = 'WISE_3.4'
                    metadata['wavelength'] = 3.4
                elif '4.6' in filename or 'w2' in filename:
                    metadata['band'] = 'WISE_4.6'
                    metadata['wavelength'] = 4.6
                elif '12' in filename or 'w3' in filename:
                    metadata['band'] = 'WISE_12'
                    metadata['wavelength'] = 12.0
                elif '22' in filename or 'w4' in filename:
                    metadata['band'] = 'WISE_22'
                    metadata['wavelength'] = 22.0
            elif 'dss' in filename:
                metadata['band'] = 'DSS'
                metadata['wavelength'] = 0.6  # Approximate visible wavelength
            
            # Debug: print key metadata
            if DEBUG:
                debug_info = {
                    'file': metadata['filename'],
                    'shape': metadata['shape'],
                    'has_wcs': metadata.get('has_wcs', False),
                    'obs_date': metadata.get('obs_date', 'Unknown'),
                    'obs_year': metadata.get('obs_year', 'Unknown'),
                    'band': metadata.get('band', 'Unknown')
                }
                debug_print(f"Metadata for {metadata['filename']}: {debug_info}")
            
            return metadata
            
    except Exception as e:
        print(f"Error extracting metadata from {fits_file}: {e}")
        if DEBUG:
            import traceback
            debug_print(traceback.format_exc())
        return None

# Function to load and preprocess FITS file
def load_fits_data(fits_file, metadata=None):
    """Load and preprocess a FITS file."""
    try:
        if metadata is None:
            metadata = extract_fits_metadata(fits_file)
            if metadata is None:
                return None, None, None
        
        with fits.open(fits_file) as hdul:
            data_hdu = metadata.get('data_hdu', 0)
            data = hdul[data_hdu].data
            header = hdul[data_hdu].header
            
            # Handle 3D data (take first slice)
            if data is not None and data.ndim > 2:
                debug_print(f"Handling 3D data in {fits_file}, shape: {data.shape}")
                data = data[0]
            
            try:
                wcs = WCS(header)
                
                # Basic WCS validation
                if not wcs.has_celestial:
                    debug_print(f"Warning: WCS in {fits_file} lacks celestial information")
                    
                # Check if WCS is valid
                test_pix = (0, 0)
                test_world = wcs.wcs_pix2world(test_pix[0], test_pix[1], 0)
                if not (np.isfinite(test_world[0]) and np.isfinite(test_world[1])):
                    debug_print(f"Warning: WCS in {fits_file} gives non-finite coordinates")
                    wcs = None
                    
            except Exception as e:
                debug_print(f"WCS error in {fits_file}: {e}")
                wcs = None
            
            # Debug: save original image
            if DEBUG:
                debug_save_array(data, f"original_{os.path.basename(fits_file).replace('.', '_')}")
            
            return data, header, wcs
    
    except Exception as e:
        print(f"Error loading {fits_file}: {e}")
        if DEBUG:
            import traceback
            debug_print(traceback.format_exc())
        return None, None, None

# Function to preprocess an image for shift-and-stack
def preprocess_image(data, header=None, apply_filters=True):
    """Preprocess an image for shift-and-stack."""
    if data is None:
        return None
    
    # Make a copy to avoid modifying the original
    img = data.copy()
    
    # Replace NaNs with zeros
    img = np.nan_to_num(img, nan=0.0)
    
    # Make sure the image is float type
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32)
    
    # Apply sigma clipping to remove outliers - FIXED 
    mean, median, std = sigma_clipped_stats(img, sigma=3, maxiters=5)
    
    debug_print(f"Image stats: mean={mean:.2f}, median={median:.2f}, std={std:.2f}")
    
    # Check if the standard deviation is reasonable
    if std <= 0 or not np.isfinite(std):
        debug_print("Warning: Invalid standard deviation, using robust alternative")
        # Use a more robust method to estimate background deviation
        sorted_img = np.sort(img.flatten())
        n = len(sorted_img)
        
        # Use interquartile range for robustness
        q1_idx = int(n * 0.25)
        q3_idx = int(n * 0.75)
        q1 = sorted_img[q1_idx]
        q3 = sorted_img[q3_idx]
        iqr = q3 - q1
        
        # Estimate standard deviation from IQR
        std = iqr / 1.349  # Standard conversion factor
        
        # If still invalid, use a simple percentile spread
        if std <= 0 or not np.isfinite(std):
            p10 = sorted_img[int(n * 0.1)]
            p90 = sorted_img[int(n * 0.9)]
            std = (p90 - p10) / (2 * 1.28)  # Rough approximation
        
        median = sorted_img[int(n * 0.5)]
        debug_print(f"Revised stats: median={median:.2f}, std={std:.2f}")
    
    # Normalize the image
    if std > 0:
        img_norm = (img - median) / std
    else:
        img_norm = img - median
    
    # Debug: save normalized image
    if DEBUG:
        debug_save_array(img_norm, f"normalized_{np.random.randint(10000)}")
    
    # Apply filtering if requested
    if apply_filters:
        # Try different filters to enhance faint signals
        img_filtered = median_filter(img_norm, size=3)
        img_filtered = gaussian_filter(img_filtered, sigma=1.0)
    else:
        img_filtered = img_norm
    
    # Debug: save filtered image
    if DEBUG:
        debug_save_array(img_filtered, f"filtered_{np.random.randint(10000)}")
    
    # Scale to [0, 1] range for easier visualization
    if np.max(img_filtered) > np.min(img_filtered):
        img_scaled = (img_filtered - np.min(img_filtered)) / (np.max(img_filtered) - np.min(img_filtered))
    else:
        img_scaled = img_filtered
    
    # Apply additional contrast enhancement to bring out faint features
    img_enhanced = exposure.equalize_hist(img_scaled)
    
    # Create result with different preprocessing levels
    result = {
        'original': img,
        'normalized': img_norm,
        'filtered': img_filtered,
        'scaled': img_scaled,
        'enhanced': img_enhanced,
        'stats': {
            'mean': mean,
            'median': median,
            'std': std
        }
    }
    
    return result

# Function to calculate expected shift for a target
def calculate_shift(wcs, ra_rate, dec_rate, base_year, obs_year):
    """Calculate the expected shift in pixels for a target."""
    if wcs is None or base_year is None or obs_year is None:
        debug_print("calculate_shift: Missing required inputs")
        return None, None
    
    try:
        # Time difference in years
        dt = obs_year - base_year
        debug_print(f"Time difference: {dt:.2f} years")
        
        # Calculate position shift in degrees
        dra = ra_rate * dt
        ddec = dec_rate * dt
        debug_print(f"Position shift: dRA={dra:.6f}°, dDec={ddec:.6f}°")
        
        # Reference position (center of field)
        ref_ra = P9_PREDICTED_2025['ra']
        ref_dec = P9_PREDICTED_2025['dec']
        
        # Calculate target positions
        ra1 = ref_ra
        dec1 = ref_dec
        ra2 = ref_ra + dra
        dec2 = ref_dec + ddec
        
        # Convert to pixel coordinates
        x1, y1 = wcs.wcs_world2pix(ra1, dec1, 0)
        x2, y2 = wcs.wcs_world2pix(ra2, dec2, 0)
        
        # Calculate shift in pixels
        x_shift = x1 - x2
        y_shift = y1 - y2
        
        debug_print(f"Pixel shift: dx={x_shift:.2f}, dy={y_shift:.2f}")
        
        # Check if shift is reasonable
        if not np.isfinite(x_shift) or not np.isfinite(y_shift):
            debug_print("Warning: Non-finite shift values")
            return None, None
        
        # If shift is too large, it might be a WCS issue
        if abs(x_shift) > 100 or abs(y_shift) > 100:
            debug_print("Warning: Unusually large shift values, might be a WCS issue")
        
        return x_shift, y_shift
    
    except Exception as e:
        print(f"Error calculating shift: {e}")
        if DEBUG:
            import traceback
            debug_print(traceback.format_exc())
        return None, None

# Function to apply shift to an image
def apply_shift(image, x_shift, y_shift, mode='constant'):
    """Apply a shift to an image."""
    if image is None:
        return None
    
    try:
        # Check if shifts are valid
        if x_shift is None or y_shift is None or not np.isfinite(x_shift) or not np.isfinite(y_shift):
            debug_print("apply_shift: Invalid shift values")
            return None
        
        # Apply shift using scipy.ndimage.shift
        shifted = shift(image, (y_shift, x_shift), mode=mode, cval=0.0)
        
        # Debug: save shifted image
        if DEBUG:
            debug_save_array(shifted, f"shifted_x{x_shift:.1f}_y{y_shift:.1f}_{np.random.randint(10000)}")
        
        return shifted
    
    except Exception as e:
        print(f"Error applying shift: {e}")
        if DEBUG:
            import traceback
            debug_print(traceback.format_exc())
        return None

# Function to perform shift-and-stack - FIXED NUMPY ARRAY BOOLEAN EVALUATION
def shift_and_stack(images, wcs_list, metadata_list, base_year, ra_rates, dec_rates, stack_method='mean'):
    """Perform shift-and-stack with different motion rates."""
    debug_print(f"Starting shift-and-stack with {len(images)} images")
    debug_print(f"Testing {len(ra_rates)}x{len(dec_rates)} rate combinations")
    
    if not images or len(images) < 2:
        debug_print("Not enough images for stacking")
        return None, None
    
    # Prepare result containers
    stacks = {}
    best_snr = 0
    best_stack = None
    best_params = None
    
    # Try different stacking methods
    if stack_method == 'all':
        methods = ['mean', 'median', 'weighted']
    else:
        methods = [stack_method]
    
    debug_print(f"Using stacking methods: {methods}")
    
    # Loop over all combinations of rates and methods
    for method in methods:
        for ra_rate, dec_rate in zip(ra_rates, dec_rates):
            key = f"{method}_ra{ra_rate:.6f}_dec{dec_rate:.6f}"
            debug_print(f"\nProcessing stack: {key}")
            
            # Create a clean slate for each stack attempt
            if method == 'median':
                shifted_images = []
            else:  # mean or weighted
                stack_sum = np.zeros_like(images[0])
                weight_sum = np.zeros_like(images[0])
            
            # Track how many images were successfully shifted and added
            successful_shifts = 0
            
            # Apply shifts and stack images
            for i, (img, wcs, meta) in enumerate(zip(images, wcs_list, metadata_list)):
                if img is None or wcs is None:
                    debug_print(f"  Image {i}: Missing image data or WCS")
                    continue
                
                # Get observation year - should always have one now with our improved logic
                obs_year = meta.get('obs_year', DEFAULT_OBSERVATION_YEAR)
                debug_print(f"  Image {i}: Using observation year {obs_year:.2f}")
                
                # Calculate shift
                x_shift, y_shift = calculate_shift(wcs, ra_rate, dec_rate, base_year, obs_year)
                if x_shift is None or y_shift is None:
                    debug_print(f"  Image {i}: Failed to calculate shift")
                    continue
                
                # Apply shift
                shifted = apply_shift(img, x_shift, y_shift)
                if shifted is None:
                    debug_print(f"  Image {i}: Failed to apply shift")
                    continue
                
                # Determine weight
                if method == 'weighted':
                    # Weight by inverse of background noise
                    bg_pixels = shifted[shifted < np.median(shifted)]
                    if len(bg_pixels) > 0:
                        bg_std = np.std(bg_pixels)
                        if bg_std > 0 and np.isfinite(bg_std):
                            weight = 1.0 / bg_std
                        else:
                            weight = 1.0
                    else:
                        weight = 1.0
                else:
                    weight = 1.0
                
                # Add to stack
                if method == 'median':
                    shifted_images.append(shifted)
                else:  # mean or weighted
                    stack_sum += shifted * weight
                    weight_sum += weight * np.ones_like(shifted)
                
                successful_shifts += 1
                debug_print(f"  Image {i}: Successfully shifted and added to stack")
            
            debug_print(f"Successfully processed {successful_shifts} out of {len(images)} images")
            
            # Skip if no images were successfully shifted
            if successful_shifts == 0:
                debug_print("  No images were successfully shifted. Skipping this combination.")
                continue
            
            # Finalize the stack
            if method == 'median':
                if len(shifted_images) > 0:
                    # Convert list to array and compute median
                    shifted_array = np.array(shifted_images)
                    stacked = np.median(shifted_array, axis=0)
                    debug_print(f"  Created median stack from {len(shifted_images)} images")
                else:
                    debug_print("  No images available for median stack")
                    continue
            else:  # mean or weighted
                # Normalize by weights
                mask = weight_sum > 0
                stacked = np.zeros_like(stack_sum)
                stacked[mask] = stack_sum[mask] / weight_sum[mask]
                debug_print(f"  Created {method} stack, {np.sum(mask)} valid pixels")
            
            # Basic validation of the stacked image
            if not np.any(stacked) or np.all(np.isnan(stacked)):
                debug_print("  Stack contains all zeros or NaNs. Skipping.")
                continue
            
            # Save debug image of the stack
            debug_save_array(stacked, f"stack_{key}", 
                            {'METHOD': method, 'RARATE': ra_rate, 'DECRATE': dec_rate})
            
            # Measure SNR in stacked image
            background = np.median(stacked)
            if np.isfinite(background):
                background_pixels = stacked[stacked < background * 1.5]
                if len(background_pixels) > 0:
                    bg_std = np.std(background_pixels)
                    if bg_std > 0 and np.isfinite(bg_std):
                        # Find peaks in the stacked image
                        threshold = background + 5 * bg_std
                        peaks = feature.peak_local_max(stacked, min_distance=5, threshold_abs=threshold)
                        
                        # FIXED: Properly check if array has elements
                        if len(peaks) > 0:
                            # Calculate SNR for each peak
                            peak_snrs = [(stacked[y, x] - background) / bg_std for y, x in peaks]
                            max_snr = max(peak_snrs) if peak_snrs else 0
                            max_idx = np.argmax(peak_snrs) if peak_snrs else 0
                            
                            # FIXED: Properly handle indexing into peaks only if array has elements
                            max_peak = peaks[max_idx] if len(peaks) > 0 else None
                            
                            debug_print(f"  Found {len(peaks)} peaks, max SNR: {max_snr:.1f}")
                        else:
                            max_snr = 0
                            max_peak = None
                            debug_print("  No peaks found in stack")
                        
                        # Update best result if this is better
                        if max_snr > best_snr:
                            best_snr = max_snr
                            best_stack = stacked
                            best_params = {
                                'method': method,
                                'ra_rate': ra_rate,
                                'dec_rate': dec_rate,
                                'snr': max_snr,
                                'peak': max_peak
                            }
                            debug_print(f"  New best stack: SNR={max_snr:.1f}")
            
            # Store the result
            stacks[key] = {
                'stack': stacked,
                'method': method,
                'ra_rate': ra_rate,
                'dec_rate': dec_rate
            }
            debug_print(f"  Stack {key} created successfully")
    
    # Final result
    if stacks:
        debug_print(f"Created {len(stacks)} valid stacks")
        if best_params:
            debug_print(f"Best stack: {best_params['method']} method, "
                       f"RA rate={best_params['ra_rate']:.6f}, "
                       f"Dec rate={best_params['dec_rate']:.6f}, "
                       f"SNR={best_params['snr']:.1f}")
        else:
            debug_print("No best stack found (no significant peaks)")
    else:
        debug_print("No valid stacks created")
    
    return stacks, best_params

# Function to detect sources in a stacked image
def detect_sources_in_stack(stack, snr_threshold=5.0):
    """Detect sources in a stacked image."""
    if stack is None:
        return []
    
    try:
        # Calculate background statistics
        mean, median, std = sigma_clipped_stats(stack, sigma=3.0)
        
        debug_print(f"Source detection - background stats: mean={mean:.2f}, median={median:.2f}, std={std:.2f}")
        
        # Make sure std is valid
        if std <= 0 or not np.isfinite(std):
            debug_print("Warning: Invalid background std for source detection")
            return []
        
        # Set detection threshold
        threshold = median + (snr_threshold * std)
        
        # Use DAOStarFinder to detect sources
        daofind = DAOStarFinder(fwhm=3.0, threshold=threshold)
        sources = daofind(stack - median)
        
        if sources is None or len(sources) == 0:
            debug_print("No sources detected")
            return []
        
        # Calculate SNR for each source
        sources['snr'] = (sources['peak'] - median) / std
        
        # Sort by SNR
        sources.sort('snr', reverse=True)
        
        debug_print(f"Detected {len(sources)} sources, SNR range: {sources['snr'][0]:.1f} to {sources['snr'][-1]:.1f}")
        
        return sources
    
    except Exception as e:
        print(f"Error detecting sources: {e}")
        if DEBUG:
            import traceback
            debug_print(traceback.format_exc())
        return []

# Function to visualize a stack result
def visualize_stack(stack, params, output_file, sources=None, show_sources=True):
    """Visualize a stack result."""
    if stack is None:
        debug_print("Cannot visualize: stack is None")
        return
    
    try:
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Display the stacked image
        plt.subplot(2, 2, 1)
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(stack)
        plt.imshow(stack, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(label='Flux')
        plt.title(f"Stack (Linear Scale)\n"
                f"RA rate={params['ra_rate']:.6f}, Dec rate={params['dec_rate']:.6f}")
        
        # Plot with log normalization
        plt.subplot(2, 2, 2)
        plt.imshow(stack, origin='lower', cmap='viridis',
                norm=LogNorm(vmin=max(1e-3, vmin), vmax=max(1e-2, vmax)))
        plt.colorbar(label='Flux (log scale)')
        plt.title(f"Stack (Log Scale)\nMethod: {params['method']}")
        
        # Plot with histogram equalization
        plt.subplot(2, 2, 3)
        img_eq = exposure.equalize_hist(stack)
        plt.imshow(img_eq, origin='lower', cmap='viridis')
        plt.colorbar(label='Equalized Flux')
        plt.title("Stack (Histogram Equalization)")
        
        # Plot with source detection
        plt.subplot(2, 2, 4)
        plt.imshow(stack, origin='lower', cmap='gray_r',
                norm=PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax))
        
        if show_sources and sources is not None and len(sources) > 0:
            # Mark detected sources
            for i, src in enumerate(sources[:10]):  # Limit to top 10
                x, y = src['xcentroid'], src['ycentroid']
                snr = src['snr']
                plt.plot(x, y, 'ro', markersize=10, alpha=0.7)
                plt.text(x+5, y+5, f"{i+1}: SNR={snr:.1f}", color='red', fontsize=8)
        
        plt.colorbar(label='Flux (power scale)')
        plt.title("Source Detection")
        
        # Add overall title
        plt.suptitle(f"Shift-and-Stack Results\nBest SNR: {params.get('snr', 0):.1f}", fontsize=16)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(output_file, dpi=150)
        plt.close()
        
        debug_print(f"Visualization saved to {output_file}")
    
    except Exception as e:
        print(f"Error visualizing stack: {e}")
        if DEBUG:
            import traceback
            debug_print(traceback.format_exc())

# Main function for shift-and-stack processing
def process_shift_and_stack():
    """Main function to process shift-and-stack."""
    # Find all FITS files
    fits_files = find_fits_files(args.data_dir)
    print(f"Found {len(fits_files)} FITS files")
    
    if not fits_files:
        print("No FITS files found. Exiting.")
        return
    
    # Extract metadata from all files
    metadata_list = []
    for fits_file in fits_files:
        metadata = extract_fits_metadata(fits_file)
        if metadata is not None:
            metadata_list.append(metadata)
    
    print(f"Extracted metadata from {len(metadata_list)} files")
    analysis_log['files_processed'] = len(metadata_list)
    
    # Group files by band/wavelength
    bands = {}
    for meta in metadata_list:
        band = meta.get('band', 'Unknown')
        if band not in bands:
            bands[band] = []
        bands[band].append(meta)
    
    print("Files grouped by band:")
    for band, files in bands.items():
        print(f"  {band}: {len(files)} files")
    
    # If no bands were identified, try to process all files together
    if len(bands) == 0 or (len(bands) == 1 and 'Unknown' in bands and len(bands['Unknown']) < 2):
        print("No bands identified or not enough files per band. Trying to process all files together.")
        if len(metadata_list) >= 2:
            bands['All'] = metadata_list
    
    # Process each band separately
    for band, files in bands.items():
        print(f"\nProcessing band: {band}")
        
        # Skip bands with too few files
        if len(files) < 2:
            print(f"  Skipping {band} - need at least 2 files for shift-and-stack")
            continue
        
        # Sort files by observation date if available
        files_with_dates = [f for f in files if 'obs_year' in f]
        if files_with_dates:
            files_with_dates.sort(key=lambda x: x['obs_year'])
            print(f"  Date range: {files_with_dates[0]['obs_year']:.1f} to {files_with_dates[-1]['obs_year']:.1f}")
        
        # Load all images
        images = []
        wcs_list = []
        loaded_metadata = []
        
        for meta in files:
            data, header, wcs = load_fits_data(meta['file'], meta)
            if data is not None:
                # Basic validation of the image data
                if not np.any(data) or np.all(np.isnan(data)):
                    debug_print(f"  Skipping {meta['filename']} - all zeros or NaNs")
                    continue
                
                # Preprocess image
                processed = preprocess_image(data)
                if processed is not None:
                    images.append(processed['filtered'])  # Use filtered version for stacking
                    wcs_list.append(wcs)
                    loaded_metadata.append(meta)
                    debug_print(f"  Successfully loaded and preprocessed {meta['filename']}")
                else:
                    debug_print(f"  Failed to preprocess {meta['filename']}")
            else:
                debug_print(f"  Failed to load data from {meta['filename']}")
        
        print(f"  Loaded {len(images)} images")
        
        if len(images) < 2:
            print("  Skipping - need at least 2 images for shift-and-stack")
            continue
        
        # Define range of rates to try - USE SMALLER STEPS
        ra_range = args.rate_range
        dec_range = args.rate_range
        
        # Use more steps for better coverage
        ra_rates = np.linspace(P9_RA_RATE * (1 - ra_range), P9_RA_RATE * (1 + ra_range), 7)
        dec_rates = np.linspace(P9_DEC_RATE * (1 - dec_range), P9_DEC_RATE * (1 + dec_range), 7)
        
        # Also try zero motion as a control test
        ra_rates = np.append(ra_rates, 0.0)
        dec_rates = np.append(dec_rates, 0.0)
        
        print(f"  Trying {len(ra_rates)} RA rates and {len(dec_rates)} Dec rates")
        print(f"  RA rates: {ra_rates}")
        print(f"  Dec rates: {dec_rates}")
        
        # Set base year for shift calculations (use AKARI detection)
        base_year = P9_AKARI_POSITION['year']
        
        # Perform shift-and-stack with different methods
        stacks, best_params = shift_and_stack(images, wcs_list, loaded_metadata, 
                                             base_year, ra_rates, dec_rates, 
                                             stack_method='all')
        
        if stacks and best_params:
            print(f"  Created {len(stacks)} stacks")
            print(f"  Best result: {best_params['method']} method, "
                 f"RA rate={best_params['ra_rate']:.6f}, "
                 f"Dec rate={best_params['dec_rate']:.6f}, "
                 f"SNR={best_params['snr']:.1f}")
            
            analysis_log['stacks_created'] += len(stacks)
            
            # Get the best stack
            best_key = f"{best_params['method']}_ra{best_params['ra_rate']:.6f}_dec{best_params['dec_rate']:.6f}"
            best_stack = stacks[best_key]['stack']
            
            # Detect sources in the best stack
            sources = detect_sources_in_stack(best_stack)
            print(f"  Detected {len(sources)} sources in the best stack")
            
            # Save the stack as FITS
            stack_file = os.path.join(stacks_dir, f"{band}_best_stack.fits")
            hdu = fits.PrimaryHDU(best_stack)
            hdu.header['BAND'] = band
            hdu.header['METHOD'] = best_params['method']
            hdu.header['RARATE'] = best_params['ra_rate']
            hdu.header['DECRATE'] = best_params['dec_rate']
            hdu.header['SNR'] = best_params['snr']
            hdu.header['NIMAGES'] = len(images)
            hdu.header['BASEYR'] = base_year
            hdu.writeto(stack_file, overwrite=True)
            
            # Save source catalog
            if len(sources) > 0:
                sources_file = os.path.join(sources_dir, f"{band}_sources.csv")
                sources.write(sources_file, format='ascii.csv', overwrite=True)
                
                # Record potential detections
                for src in sources[:5]:  # Top 5 sources
                    if src['snr'] > 5.0:  # Only record significant detections
                        detection = {
                            'band': band,
                            'x': float(src['xcentroid']),
                            'y': float(src['ycentroid']),
                            'snr': float(src['snr']),
                            'ra_rate': best_params['ra_rate'],
                            'dec_rate': best_params['dec_rate']
                        }
                        analysis_log['potential_detections'].append(detection)
            
            # Visualize the best stack
            vis_file = os.path.join(plots_dir, f"{band}_best_stack.png")
            visualize_stack(best_stack, best_params, vis_file, sources)
            
            # Save the top 3 stacks for comparison
            if len(stacks) > 1:
                # Sort stacks by SNR
                stack_snrs = []
                for key, stack_data in stacks.items():
                    # Calculate stack SNR
                    stack_img = stack_data['stack']
                    mean, median, std = sigma_clipped_stats(stack_img, sigma=3.0)
                    peaks = feature.peak_local_max(stack_img, min_distance=5, 
                                                  threshold_abs=median + 5*std)
                    if len(peaks) > 0:
                        peak_values = [stack_img[y, x] for y, x in peaks]
                        max_peak = max(peak_values)
                        snr = (max_peak - median) / std
                    else:
                        snr = 0
                    
                    stack_snrs.append((key, snr))
                
                # Sort by SNR
                stack_snrs.sort(key=lambda x: x[1], reverse=True)
                
                # Create comparison figure of top stacks
                plt.figure(figsize=(15, 5 * min(3, len(stack_snrs))))
                
                for i, (key, snr) in enumerate(stack_snrs[:3]):
                    if i >= 3:
                        break
                    
                    stack_data = stacks[key]
                    stack_img = stack_data['stack']
                    
                    # Regular scale
                    plt.subplot(3, 3, i*3 + 1)
                    interval = ZScaleInterval()
                    vmin, vmax = interval.get_limits(stack_img)
                    plt.imshow(stack_img, origin='lower', cmap='viridis', 
                              vmin=vmin, vmax=vmax)
                    plt.colorbar(label='Flux')
                    plt.title(f"Stack {i+1} (Linear) - SNR: {snr:.1f}\n"
                             f"Method: {stack_data['method']}\n"
                             f"RA={stack_data['ra_rate']:.6f}, Dec={stack_data['dec_rate']:.6f}")
                    
                    # Log scale
                    plt.subplot(3, 3, i*3 + 2)
                    plt.imshow(stack_img, origin='lower', cmap='viridis',
                              norm=LogNorm(vmin=max(1e-3, vmin), vmax=max(1e-2, vmax)))
                    plt.colorbar(label='Log Flux')
                    plt.title(f"Stack {i+1} (Log scale)")
                    
                    # Histogram equalization
                    plt.subplot(3, 3, i*3 + 3)
                    img_eq = exposure.equalize_hist(stack_img)
                    plt.imshow(img_eq, origin='lower', cmap='viridis')
                    plt.colorbar(label='Equalized Flux')
                    plt.title(f"Stack {i+1} (Equalized)")
                
                plt.suptitle(f"{band} - Top 3 Stacks Comparison", fontsize=16)
                plt.tight_layout()
                
                # Save comparison figure
                comp_file = os.path.join(plots_dir, f"{band}_stack_comparison.png")
                plt.savefig(comp_file, dpi=150)
                plt.close()
        else:
            print("  No valid stacks created")
    
    # If there are potential detections, create a summary
    if analysis_log['potential_detections']:
        print("\nPotential detections found:")
        for i, det in enumerate(analysis_log['potential_detections']):
            print(f"  {i+1}. {det['band']} - SNR: {det['snr']:.1f}")
        
        # Create a summary plot of all detections
        plt.figure(figsize=(10, 8))
        
        # Plot detections by SNR and band
        bands = list(set(det['band'] for det in analysis_log['potential_detections']))
        band_colors = plt.cm.tab10(np.linspace(0, 1, len(bands)))
        band_to_color = dict(zip(bands, band_colors))
        
        for i, det in enumerate(analysis_log['potential_detections']):
            plt.scatter(i+1, det['snr'], s=100, c=[band_to_color[det['band']]], 
                       label=det['band'] if i == 0 or det['band'] != analysis_log['potential_detections'][i-1]['band'] else "")
            plt.text(i+1, det['snr'] + 0.5, f"{det['band']}", ha='center')
        
        plt.axhline(y=5.0, linestyle='--', color='r', label='5σ threshold')
        plt.xlabel('Detection ID')
        plt.ylabel('SNR')
        plt.title('Potential Planet Nine Detections by SNR')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        det_file = os.path.join(plots_dir, 'potential_detections.png')
        plt.savefig(det_file, dpi=150)
        plt.close()
    
    # Complete analysis log
    analysis_log['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    analysis_log_file = os.path.join(args.output_dir, 'analysis_log.json')
    
    with open(analysis_log_file, 'w') as f:
        json.dump(analysis_log, f, indent=2)
    
    print(f"\nAnalysis complete. Results saved to {args.output_dir}")
    print(f"Log file: {analysis_log_file}")

# Run the main function
if __name__ == "__main__":
    process_shift_and_stack()
