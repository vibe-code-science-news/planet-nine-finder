#!/usr/bin/env python
"""
Planet Nine Trajectory Tracker

This script calculates the trajectory of Planet Nine over time based on two known
positions. It determines the appropriate time granularity based on the transit time
across a field of view, visualizes the trajectory, and outputs the positions to a file.

Usage:
  python planet_nine_tracker.py [--radius RADIUS] [--output OUTPUT] [--start YEAR] [--end YEAR]

Options:
  --radius RADIUS    Search radius in degrees [default: 0.2]
  --output OUTPUT    Output base name [default: p9_trajectory]
  --start YEAR       Start year [default: 1983]
  --end YEAR         End year [default: 2025]
  --step STEP        Time step in years (0 for auto) [default: 0]
"""

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from astropy.time import Time
import os
import os

def calculate_angular_separation(ra1, dec1, ra2, dec2):
    """Calculate angular separation between two celestial coordinates (in arcminutes)"""
    # Convert to radians
    ra1_rad = math.radians(ra1)
    dec1_rad = math.radians(dec1)
    ra2_rad = math.radians(ra2)
    dec2_rad = math.radians(dec2)
    
    # Haversine formula
    dra = ra2_rad - ra1_rad
    ddec = dec2_rad - dec1_rad
    
    a = math.sin(ddec/2)**2 + math.cos(dec1_rad) * math.cos(dec2_rad) * math.sin(dra/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    dist_rad = c
    
    # Convert to arcminutes
    dist_arcmin = math.degrees(dist_rad) * 60
    
    return dist_arcmin

def calculate_position(ra1, dec1, ra2, dec2, years1to2, year):
    """Calculate the position at a given year based on two known positions"""
    # Calculate the rate of change per year
    ra_rate = (ra2 - ra1) / years1to2
    dec_rate = (dec2 - dec1) / years1to2
    
    # Calculate years since first observation
    years_since_first = year - ra1_year
    
    # Calculate projected position
    ra = ra1 + (ra_rate * years_since_first)
    dec = dec1 + (dec_rate * years_since_first)
    
    return ra, dec, ra_rate, dec_rate

def calculate_transit_time(ra_rate, dec_rate, radius_deg):
    """Calculate time to transit a field of view with given radius"""
    # Calculate total angular motion rate
    total_rate_deg_per_year = math.sqrt(ra_rate**2 + dec_rate**2)
    
    # Calculate diameter of field of view
    diameter = 2 * radius_deg
    
    # Calculate transit time
    transit_time = diameter / total_rate_deg_per_year
    
    return transit_time

def determine_time_step(transit_time, min_steps=10):
    """Determine appropriate time step based on transit time"""
    # We want at least min_steps positions while crossing the field
    time_step = transit_time / min_steps
    
    # Round to nearest 0.1 year, but no less than 0.1
    time_step = max(0.1, round(time_step * 10) / 10)
    
    return time_step

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Calculate and visualize Planet Nine's trajectory")
    parser.add_argument('--radius', type=float, default=0.2, help='Search radius in degrees')
    parser.add_argument('--output', type=str, default='config/p9_trajectory', help='Output base name')
    parser.add_argument('--start', type=float, default=1983, help='Start year for calculations')
    parser.add_argument('--end', type=float, default=2025, help='End year for calculations')
    parser.add_argument('--output-start', type=float, default=2015, help='Start year for output file')
    parser.add_argument('--step', type=float, default=0, help='Time step in years (0 for auto)')
    args = parser.parse_args()

    # Known positions
    ra1_year = 1983  # IRAS observation year
    ra1 = 35.74075   # RA for IRAS detection in 1983
    dec1 = -48.5125  # Dec for IRAS detection in 1983
    
    ra2_year = 2006  # AKARI observation year
    ra2 = 35.18379   # RA for AKARI detection in 2006
    dec2 = -49.2135  # Dec for AKARI detection in 2006
    
    # Calculate years between observations
    years_between = ra2_year - ra1_year
    
    # Calculate initial position and rates
    _, _, ra_rate, dec_rate = calculate_position(ra1, dec1, ra2, dec2, years_between, ra1_year)
    
    # Calculate transit time across the field of view
    transit_time = calculate_transit_time(ra_rate, dec_rate, args.radius)
    
    # Determine appropriate time step
    if args.step <= 0:
        time_step = determine_time_step(transit_time)
    else:
        time_step = args.step
    
    print(f"Planet Nine Motion Analysis")
    print(f"===========================")
    print(f"Known positions:")
    print(f"  1983: RA = {ra1:.5f}, Dec = {dec1:.5f} (IRAS)")
    print(f"  2006: RA = {ra2:.5f}, Dec = {dec2:.5f} (AKARI)")
    print(f"")
    print(f"Motion rates:")
    print(f"  RA rate:  {ra_rate:.6f} degrees/year")
    print(f"  Dec rate: {dec_rate:.6f} degrees/year")
    print(f"  Angular velocity: {abs(calculate_angular_separation(ra1, dec1, ra2, dec2) / years_between):.4f} arcmin/year")
    print(f"")
    print(f"Field of view transit analysis:")
    print(f"  Search radius: {args.radius:.2f} degrees")
    print(f"  Time to cross field: {transit_time:.2f} years")
    print(f"  Recommended time step: {time_step:.1f} years")
    print(f"")
    print(f"Trajectory output:")
    print(f"  Full calculation period: {args.start} to {args.end}")
    print(f"  Output period: {args.output_start} to {args.end}")
    print(f"")
    
    # Generate positions table
    start_year = args.start
    end_year = args.end
    
    # Create time points with appropriate step size
    time_points = np.arange(start_year, end_year + time_step, time_step)
    
    # Calculate positions at each time point
    positions = []
    for year in time_points:
        ra, dec, _, _ = calculate_position(ra1, dec1, ra2, dec2, years_between, year)
        positions.append((year, ra, dec))
    
    # Output to text file
    txt_filename = f"{args.output}.txt"
    with open(txt_filename, 'w') as f:
        f.write(f"# Planet Nine Predicted Positions\n")
        f.write(f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Based on:\n")
        f.write(f"#   1983: RA = {ra1:.5f}, Dec = {dec1:.5f} (IRAS)\n")
        f.write(f"#   2006: RA = {ra2:.5f}, Dec = {dec2:.5f} (AKARI)\n")
        f.write(f"# Motion rates: RA = {ra_rate:.6f} deg/yr, Dec = {dec_rate:.6f} deg/yr\n")
        f.write(f"# Time to cross {args.radius*2:.2f}° field: {transit_time:.2f} years\n")
        f.write(f"#\n")
        f.write(f"# Year       RA (deg)      Dec (deg)\n")
        f.write(f"# -----      --------      ---------\n")
        
        for year, ra, dec in positions:
            f.write(f"{year:<10.1f} {ra:<15.5f} {dec:<15.5f}\n")
    
    print(f"Wrote {len(positions)} positions to {txt_filename}")

    # Create visualization of the trajectory
    plt.figure(figsize=(12, 8))
    
    # Plot the path
    ra_values = [p[1] for p in positions]
    dec_values = [p[2] for p in positions]
    plt.plot(ra_values, dec_values, 'b-', linewidth=1.5)
    
    # Plot the known positions
    plt.plot(ra1, dec1, 'ro', markersize=8, label=f'IRAS (1983)')
    plt.plot(ra2, dec2, 'go', markersize=8, label=f'AKARI (2006)')
    
    # Plot the predicted position for 2025
    ra2025, dec2025, _, _ = calculate_position(ra1, dec1, ra2, dec2, years_between, 2025)
    plt.plot(ra2025, dec2025, 'yo', markersize=10, label=f'Predicted (2025)')
    
    # Plot markers at regular intervals (e.g., every 5 years)
    marker_years = np.arange(5 * math.ceil(start_year/5), end_year, 5)
    for year in marker_years:
        ra, dec, _, _ = calculate_position(ra1, dec1, ra2, dec2, years_between, year)
        plt.plot(ra, dec, 'kx', markersize=6)
        # Only add year labels for more recent years to avoid clutter
        if year >= 2000:
            plt.text(ra, dec+0.05, f"{int(year)}", fontsize=8, ha='center')
    
    # Draw search radius for 2025 position
    circle = plt.Circle((ra2025, dec2025), args.radius, color='y', fill=False, linestyle='--', alpha=0.7)
    plt.gca().add_patch(circle)
    
    # Configure plot
    plt.xlabel('Right Ascension (degrees)')
    plt.ylabel('Declination (degrees)')
    plt.title(f'Planet Nine Trajectory ({int(start_year)}-{int(end_year)})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Invert RA axis (astronomical convention)
    plt.gca().invert_xaxis()
    
    # Equal aspect ratio ensures circles look circular
    plt.axis('equal')
    plt.tight_layout()
    
    # Save visualization
    # Make sure the directory exists for the image as well
    img_path = f"{args.output}.png"
    img_dir = os.path.dirname(img_path)
    if img_dir and not os.path.exists(img_dir):
        os.makedirs(img_dir, exist_ok=True)
    
    plt.savefig(img_path, dpi=150)
    print(f"Saved trajectory visualization to {img_path}")