#!/usr/bin/env python
"""
Planet Nine Data Acquisition Core Module

This module provides core functionality for searching astronomical repositories
for data that might contain Planet Nine. It defines base classes and utilities
used by specific repository handlers.
"""

import os
import yaml
import json
import logging
import numpy as np
from datetime import datetime
from astropy.coordinates import SkyCoord
import astropy.units as u

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RepositoryHandler:
    """Base class for all astronomical repository handlers"""
    
    def __init__(self, config, trajectory_data, tight_search=True):
        """
        Initialize the repository handler.
        
        Args:
            config (dict): Configuration dictionary
            trajectory_data (dict): Planet Nine trajectory data
            tight_search (bool): Whether to use tight search parameters
        """
        self.config = config
        self.trajectory_data = trajectory_data
        self.tight_search = tight_search
        self.results = []
        self.name = "Base"  # Override in subclasses
        
        # Set search radius based on tight/loose setting
        if tight_search:
            self.search_radius = config.get('search_radii', {}).get('tight', 0.2)
        else:
            self.search_radius = config.get('search_radii', {}).get('loose', 0.5)
            
        logger.info(f"Initialized {self.name} handler with search radius {self.search_radius} degrees")
        
    def search(self):
        """
        Search the repository for data.
        Must be implemented by subclasses.
        """
        raise NotImplementedError
        
    def download(self, download_dir):
        """
        Download data files from the repository.
        Must be implemented by subclasses.
        
        Args:
            download_dir (str): Directory to download files to
        """
        raise NotImplementedError
    
    def classify_results(self):
        """
        Classify results into perfect, good, or expanded matches.
        
        This applies the classification criteria from the config to each result.
        """
        # Get classification criteria from config
        perfect_criteria = self.config.get('quality_criteria', {}).get('perfect_match', {})
        good_criteria = self.config.get('quality_criteria', {}).get('good_match', {})
        
        # Get wavelength ranges
        wavelength_ranges = self.config.get('wavelength_ranges', {})
        
        # Process each result
        for result in self.results:
            # Initialize classification as 'expanded'
            result['classification'] = 'expanded'
            
            # Check if it meets 'good' criteria
            if self._meets_criteria(result, good_criteria, wavelength_ranges):
                result['classification'] = 'good'
                
                # Check if it also meets 'perfect' criteria
                if self._meets_criteria(result, perfect_criteria, wavelength_ranges):
                    result['classification'] = 'perfect'
        
        # Count results by classification
        classifications = {
            'perfect': sum(1 for r in self.results if r.get('classification') == 'perfect'),
            'good': sum(1 for r in self.results if r.get('classification') == 'good'),
            'expanded': sum(1 for r in self.results if r.get('classification') == 'expanded')
        }
        
        logger.info(f"Classification results for {self.name}: {classifications}")
        return classifications
    
    def _meets_criteria(self, result, criteria, wavelength_ranges):
        """
        Check if a result meets the specified criteria.
        
        Args:
            result (dict): The result to check
            criteria (dict): The criteria to check against
            wavelength_ranges (dict): Wavelength range definitions
            
        Returns:
            bool: True if the result meets all criteria
        """
        # Check wavelength criteria
        if 'wavelength' in criteria:
            wavelength_range = wavelength_ranges.get(criteria['wavelength'])
            if wavelength_range and 'wavelength' in result:
                # Handle None values in wavelength
                if result['wavelength'] is None:
                    return False
                
                if not (wavelength_range[0] <= result['wavelength'] <= wavelength_range[1]):
                    return False
        
        # Check year criteria
        if 'year_min' in criteria and 'obs_year' in result:
            # Handle None values in observation year
            if result['obs_year'] is None:
                return False
                
            if result['obs_year'] < criteria['year_min']:
                return False
                
        # Check position match criteria
        if 'position_match' in criteria and 'separation_deg' in result:
            # Handle None values in separation
            if result['separation_deg'] is None:
                return False
                
            if criteria['position_match'] == 'exact' and result['separation_deg'] > 0.05:
                return False
            elif criteria['position_match'] == 'within_tight' and result['separation_deg'] > self.config['search_radii']['tight']:
                return False
        
        # If we get here, all criteria are met
        return True
    
    def generate_report(self, output_dir):
        """
        Generate a standardized report for this repository.
        
        Args:
            output_dir (str): Directory to write the report to
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Classify results if not already done
        if not any('classification' in r for r in self.results):
            self.classify_results()
        
        # Count results by classification
        classifications = {
            'perfect': sum(1 for r in self.results if r.get('classification') == 'perfect'),
            'good': sum(1 for r in self.results if r.get('classification') == 'good'),
            'expanded': sum(1 for r in self.results if r.get('classification') == 'expanded'),
            'total': len(self.results)
        }
        
        # Generate report file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(output_dir, f"{self.name.lower()}_report_{timestamp}.md")
        
        with open(report_file, 'w') as f:
            f.write(f"# {self.name} Repository Search Results\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Search Parameters\n\n")
            f.write(f"- Search mode: {'Tight' if self.tight_search else 'Loose'}\n")
            f.write(f"- Search radius: {self.search_radius} degrees\n\n")
            
            f.write("## Results Summary\n\n")
            f.write(f"- Total results: {classifications['total']}\n")
            f.write(f"- Perfect matches: {classifications['perfect']}\n")
            f.write(f"- Good matches: {classifications['good']}\n")
            f.write(f"- Expanded matches: {classifications['expanded']}\n\n")
            
            # Write details about perfect matches
            if classifications['perfect'] > 0:
                f.write("### Perfect Matches\n\n")
                f.write("| ID | Instrument | Wavelength | Year | RA | Dec | Separation |\n")
                f.write("|---|------------|------------|------|----|----|------------|\n")
                for result in self.results:
                    if result.get('classification') == 'perfect':
                        f.write(f"| {result.get('id', 'N/A')} | " +
                                f"{result.get('instrument', 'N/A')} | " +
                                f"{result.get('wavelength', 'N/A')} | " +
                                f"{result.get('obs_year', 'N/A')} | " +
                                f"{result.get('ra', 'N/A'):.5f} | " +
                                f"{result.get('dec', 'N/A'):.5f} | " +
                                f"{result.get('separation_deg', 'N/A'):.5f} |\n")
                f.write("\n")
            
            # Write details about good matches
            if classifications['good'] > 0:
                f.write("### Good Matches\n\n")
                f.write("| ID | Instrument | Wavelength | Year | RA | Dec | Separation |\n")
                f.write("|---|------------|------------|------|----|----|------------|\n")
                for result in self.results:
                    if result.get('classification') == 'good':
                        f.write(f"| {result.get('id', 'N/A')} | " +
                                f"{result.get('instrument', 'N/A')} | " +
                                f"{result.get('wavelength', 'N/A')} | " +
                                f"{result.get('obs_year', 'N/A')} | " +
                                f"{result.get('ra', 'N/A'):.5f} | " +
                                f"{result.get('dec', 'N/A'):.5f} | " +
                                f"{result.get('separation_deg', 'N/A'):.5f} |\n")
                f.write("\n")
            
            # Save complete results
            f.write("## Full Results\n\n")
            f.write("The complete results are available in the accompanying JSON file.\n")
        
        # Save JSON results file
        json_file = os.path.join(output_dir, f"{self.name.lower()}_results_{timestamp}.json")
        with open(json_file, 'w') as f:
            # Convert any non-serializable objects to strings
            serializable_results = []
            for result in self.results:
                serializable = {}
                for key, value in result.items():
                    if isinstance(value, (int, float, str, bool, list, dict, type(None))):
                        serializable[key] = value
                    else:
                        serializable[key] = str(value)
                serializable_results.append(serializable)
            
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Report generated at {report_file}")
        logger.info(f"Full results saved to {json_file}")
        
        return report_file

def load_config(config_file):
    """
    Load configuration from a YAML file.
    
    Args:
        config_file (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_file}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

def load_trajectory(trajectory_file):
    """
    Load Planet Nine trajectory data from a file.
    
    Args:
        trajectory_file (str): Path to the trajectory file
        
    Returns:
        dict: Trajectory data indexed by year
    """
    trajectory_data = {}
    
    try:
        with open(trajectory_file, 'r') as f:
            # Skip header lines (starting with #)
            for line in f:
                if not line.startswith('#'):
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        year = float(parts[0])
                        ra = float(parts[1])
                        dec = float(parts[2])
                        trajectory_data[year] = {'ra': ra, 'dec': dec}
        
        logger.info(f"Loaded trajectory data from {trajectory_file}: {len(trajectory_data)} positions")
        return trajectory_data
    except Exception as e:
        logger.error(f"Error loading trajectory data: {e}")
        raise

def get_position_for_date(trajectory_data, date):
    """
    Get the expected position of Planet Nine for a specific date.
    
    Args:
        trajectory_data (dict): Trajectory data dictionary
        date (str or float): Date as a string (YYYY-MM-DD) or as a year float
        
    Returns:
        tuple: (ra, dec) in degrees
    """
    # Convert date string to year if needed
    if isinstance(date, str):
        try:
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            year = date_obj.year + (date_obj.month - 1) / 12 + date_obj.day / 365.25
        except ValueError:
            logger.warning(f"Invalid date format: {date}. Using as a year value.")
            year = float(date)
    else:
        year = float(date)
    
    # Find the closest years in the trajectory data
    years = sorted(trajectory_data.keys())
    
    if year <= years[0]:
        # Before the first year, use the first position
        return trajectory_data[years[0]]['ra'], trajectory_data[years[0]]['dec']
    
    if year >= years[-1]:
        # After the last year, use the last position
        return trajectory_data[years[-1]]['ra'], trajectory_data[years[-1]]['dec']
    
    # Find the years that bracket the requested date
    for i in range(len(years) - 1):
        if years[i] <= year <= years[i + 1]:
            # Interpolate between the two positions
            y1, y2 = years[i], years[i + 1]
            pos1, pos2 = trajectory_data[y1], trajectory_data[y2]
            
            # Linear interpolation
            frac = (year - y1) / (y2 - y1)
            ra = pos1['ra'] + frac * (pos2['ra'] - pos1['ra'])
            dec = pos1['dec'] + frac * (pos2['dec'] - pos1['dec'])
            
            return ra, dec
    
    # Should not reach here
    logger.error(f"Error finding position for year {year}")
    return None, None

def calculate_separation(ra1, dec1, ra2, dec2):
    """
    Calculate angular separation between two sky positions.
    
    Args:
        ra1, dec1: Position 1 in degrees
        ra2, dec2: Position 2 in degrees
        
    Returns:
        float: Separation in degrees
    """
    coord1 = SkyCoord(ra1, dec1, unit=u.deg)
    coord2 = SkyCoord(ra2, dec2, unit=u.deg)
    
    return coord1.separation(coord2).deg