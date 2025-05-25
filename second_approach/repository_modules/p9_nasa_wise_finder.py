#!/usr/bin/env python
"""
Planet Nine NASA WISE Repository Handler

This module provides functionality for searching NASA's WISE/NEOWISE archive
for data that might contain Planet Nine, focusing on the longer wavelength bands
(W3: 12μm and W4: 22μm) which are more suitable for detecting cold distant objects.

Uses NASA's public APIs for direct access to WISE data products.
"""

import os
import logging
import requests
import json
from datetime import datetime
import time

from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u

from p9_data_core import RepositoryHandler, get_position_for_date, calculate_separation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NASAWISEHandler(RepositoryHandler):
    """Handler for NASA WISE/NEOWISE repository searches"""
    
    def __init__(self, config, trajectory_data, tight_search=True):
        """
        Initialize the NASA WISE repository handler.
        
        Args:
            config (dict): Configuration dictionary
            trajectory_data (dict): Planet Nine trajectory data
            tight_search (bool): Whether to use tight search parameters
        """
        super().__init__(config, trajectory_data, tight_search)
        self.name = "NASA_WISE"
        
        # Define WISE-specific properties
        self.bands = {
            "W1": {
                "wavelength": 3.4,  # Microns
                "years": [2010, 2023],  # WISE + NEOWISE years
                "relevant": False,  # Too short wavelength for Planet Nine
                "description": "3.4 micron band"
            },
            "W2": {
                "wavelength": 4.6,  # Microns
                "years": [2010, 2023],
                "relevant": False,  # Too short wavelength for Planet Nine
                "description": "4.6 micron band"
            },
            "W3": {
                "wavelength": 12.0,  # Microns
                "years": [2010, 2011],  # Only available in primary mission
                "relevant": True,  # Potentially useful for Planet Nine
                "description": "12 micron band"
            },
            "W4": {
                "wavelength": 22.0,  # Microns
                "years": [2010, 2011],  # Only available in primary mission
                "relevant": True,  # Best WISE band for Planet Nine
                "description": "22 micron band"
            }
        }
        
        # Set minimum relevant wavelength from config
        self.min_relevant_wavelength = config.get('wavelength_ranges', {}).get('acceptable', [20, 500])[0]
        
        # WISE mission phases
        self.mission_phases = {
            "wise_full": {
                "start": 2010.0,  # January 2010
                "end": 2010.67,   # August 2010
                "bands": ["W1", "W2", "W3", "W4"],
                "description": "Primary WISE mission with all four bands"
            },
            "wise_3band": {
                "start": 2010.67,  # August 2010
                "end": 2011.17,   # February 2011
                "bands": ["W1", "W2", "W3"],
                "description": "WISE mission with W1, W2, W3 bands (cryogen depletion)"
            },
            "neowise_post": {
                "start": 2011.17,  # February 2011
                "end": 2011.33,   # April 2011
                "bands": ["W1", "W2"],
                "description": "Post-cryogenic NEOWISE with W1, W2 bands"
            },
            "neowise_reactivation": {
                "start": 2013.88,  # December 2013
                "end": 2023.0,    # Still ongoing
                "bands": ["W1", "W2"],
                "description": "NEOWISE Reactivation with W1, W2 bands"
            }
        }
        
        # UPDATED NASA WISE API URLs - Direct Access to IRSA Archive
        # WISE Atlas Images URLs
        self.base_url = "https://irsa.ipac.caltech.edu/cgi-bin/Atlas/nph-atlas"
        
        # WISE Single Exposure URLs
        self.single_exp_url = "https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query"
        
        # WISE Multiframe URLs 
        self.multiframe_url = "https://irsa.ipac.caltech.edu/cgi-bin/ICORE/nph-icore"
        
        # Direct file download URL
        self.download_url = "https://irsa.ipac.caltech.edu/data/download/wise-allsky"
        
        # Request headers
        self.headers = {
            "Accept": "application/json,text/plain",
            "User-Agent": "PlanetNineFinder/1.0",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        # Cache for API results to reduce duplicate calls
        self.api_cache = {}
        
    def search(self, specific_years=None, specific_bands=None):
        """
        Search NASA WISE archive for data.
        
        Args:
            specific_years (list): Optional list of years to search
            specific_bands (list): Optional list of bands to search
            
        Returns:
            list: Search results
        """
        logger.info(f"Searching NASA WISE archive with radius {self.search_radius} degrees")
        
        # Clear previous results
        self.results = []
        
        # Filter bands if not specified
        if not specific_bands:
            # Default to W3 and W4 bands which are most relevant for Planet Nine
            specific_bands = ["W3", "W4"]
            logger.info(f"Using default WISE bands: {specific_bands}")
        
        # Determine years to search
        if specific_years:
            years_to_search = specific_years
        else:
            # Use all years in trajectory data
            years_to_search = sorted(self.trajectory_data.keys())
        
        logger.info(f"Searching for years: {years_to_search}")
        
        # Filter years based on band availability
        valid_years = []
        for year in years_to_search:
            # For each band, check if this year is valid
            for band in specific_bands:
                band_info = self.bands[band]
                if band_info["years"][0] <= year <= band_info["years"][1]:
                    valid_years.append(year)
                    break
        
        if not valid_years:
            logger.warning(f"No valid years found for the specified bands {specific_bands}")
            return self.results
        
        # Process each year
        for year in valid_years:
            # Get expected position for this year
            ra, dec = self.trajectory_data[year]['ra'], self.trajectory_data[year]['dec']
            
            # Log search details
            logger.info(f"Searching year {year} at position RA={ra:.5f}, Dec={dec:.5f}")
            
            # Determine which mission phase this year falls into
            mission_phase = None
            for phase_name, phase_info in self.mission_phases.items():
                if phase_info["start"] <= year <= phase_info["end"]:
                    mission_phase = phase_name
                    break
            
            if not mission_phase:
                logger.warning(f"No WISE mission phase found for year {year}")
                continue
            
            # Get bands available in this mission phase
            available_bands = self.mission_phases[mission_phase]["bands"]
            
            # Filter for requested bands
            bands_to_search = [b for b in specific_bands if b in available_bands]
            
            if not bands_to_search:
                logger.warning(f"None of the requested bands {specific_bands} were active in {mission_phase} during {year}")
                continue
            
            # Search for each band
            for band in bands_to_search:
                self._search_wise_atlas(year, ra, dec, band, mission_phase)
        
        # Sort results by relevance
        if self.results:
            self.results.sort(key=lambda x: 
                             (x.get('classification', 'expanded') != 'perfect',
                              x.get('classification', 'expanded') != 'good',
                              x.get('separation_deg', 999)))
        
        logger.info(f"NASA WISE search complete. Found {len(self.results)} relevant results.")
        return self.results
    
    def _search_wise_atlas(self, year, ra, dec, band, mission_phase):
        """
        Search for WISE data in the AllWISE Atlas.
        
        Args:
            year (float): Year to search
            ra (float): Right Ascension in degrees
            dec (float): Declination in degrees
            band (str): WISE band (W1, W2, W3, W4)
            mission_phase (str): WISE mission phase
        """
        logger.info(f"Searching for {band} data at RA={ra:.5f}, Dec={dec:.5f} from {mission_phase}")
        
        # Check if we're looking for W3/W4 bands in the correct phase
        if band in ["W3", "W4"] and mission_phase not in ["wise_full", "wise_3band"] and band == "W4" and mission_phase != "wise_full":
            logger.warning(f"{band} band not available in {mission_phase}")
            return
        
        try:
            # Use Atlas Search Query
            params = {
                "mode": "getdetails",
                "location": f"{ra:.5f} {dec:.5f}",
                "survey": "wise",
                "band": int(band[-1]),  # Extract band number (1, 2, 3, 4)
                "radius": f"{self.search_radius}", 
                "sizeX": "0.5",
                "sizeY": "0.5",
                "type": "image",
                "subsetsize": "500"
            }
            
            # Convert params to URL-encoded string for POST request
            payload = "&".join(f"{k}={v}" for k, v in params.items())
            
            # Construct API URL
            api_url = self.base_url
            
            # Cache key to avoid duplicate requests
            cache_key = f"{ra}_{dec}_{band}_{mission_phase}_atlas"
            
            # Check cache first
            if cache_key in self.api_cache:
                logger.info(f"Using cached results for {cache_key}")
                atlas_results = self.api_cache[cache_key]
            else:
                # Execute query
                logger.info(f"Querying WISE Atlas search")
                response = requests.post(api_url, data=payload, headers=self.headers)
                
                if response.status_code != 200:
                    logger.error(f"Atlas search failed with status {response.status_code}: {response.text}")
                    return
                
                # Parse results
                atlas_results = response.text
                
                # Cache results
                self.api_cache[cache_key] = atlas_results
            
            # Look for image IDs in the response
            if "No images were found" in atlas_results:
                logger.info(f"No WISE Atlas images found for {band} at this position")
                return
            
            # Extract image IDs
            import re
            image_ids = re.findall(r'([0-9]{4}[wm][0-9]{3}_[ab][bc]_coadd-atlas-irsa-[0-9]+)', atlas_results)
            
            if not image_ids:
                logger.info(f"Could not extract image IDs from Atlas search results")
                return
            
            # Process each image
            for image_id in image_ids:
                # Create a result for this image
                result = self._process_atlas_image(image_id, band, year, ra, dec)
                if result:
                    self.results.append(result)
            
            logger.info(f"Found {len(image_ids)} WISE Atlas images for {band}")
                
        except Exception as e:
            logger.error(f"Error searching for {band} data: {e}")
    
    def _process_atlas_image(self, image_id, band, search_year, search_ra, search_dec):
        """
        Process a WISE Atlas image.
        
        Args:
            image_id: Atlas image ID
            band: WISE band (W1, W2, W3, W4)
            search_year: Year being searched
            search_ra, search_dec: Search position
            
        Returns:
            dict: Processed image data or None if invalid
        """
        try:
            # Extract basic metadata from image ID
            # Format: [tile]w[band]_[region]_coadd-atlas-irsa-[id]
            parts = image_id.split('_')
            tile = parts[0]
            region = parts[1]
            
            # Get observation date
            # Note: Atlas images are coadds of multiple observations
            # For simplicity, we'll use the search year
            obs_year = search_year
            
            # Get image center coordinates
            # For Atlas images, we'd need to query additional metadata
            # For now, we'll use the search coordinates
            img_ra = search_ra
            img_dec = search_dec
            
            # Calculate separation from expected Planet Nine position
            p9_ra, p9_dec = search_ra, search_dec
            if obs_year != search_year:
                # Recalculate expected position for the observation year
                p9_ra, p9_dec = get_position_for_date(self.trajectory_data, obs_year)
            
            separation = calculate_separation(p9_ra, p9_dec, img_ra, img_dec)
            
            # Get wavelength from band
            wavelength = self.bands[band]["wavelength"]
            
            # Construct file URL
            file_url = f"https://irsa.ipac.caltech.edu/data/WISE/wise-allsky-4band/Tile/{tile[0:2]}/{tile}/coadd/{image_id}.fits"
            
            # Create result dictionary
            result = {
                'id': image_id,
                'repository': 'NASA_WISE',
                'instrument': 'WISE',
                'band': band,
                'wavelength': wavelength,
                'obs_year': obs_year,
                'ra': img_ra,
                'dec': img_dec,
                'search_ra': p9_ra,
                'search_dec': p9_dec,
                'separation_deg': separation,
                'file_url': file_url,
                'target_name': f"WISE_Tile_{tile}",
                'est_file_size': 50 * 1024 * 1024,  # Approx. 50MB for Atlas images
                'relevance_score': self._calculate_relevance_score(wavelength, separation, band),
                'tile': tile,
                'region': region,
                'is_atlas': True
            }
            
            return result
            
        except Exception as e:
            logger.warning(f"Error processing WISE Atlas image: {e}")
            return None
    
    def _calculate_relevance_score(self, wavelength, separation, band):
        """
        Calculate a relevance score for a detection.
        
        Args:
            wavelength: Wavelength in microns
            separation: Separation in degrees
            band: WISE band
            
        Returns:
            float: Relevance score (higher is better)
        """
        # Base score - wavelength component
        if wavelength >= 22.0:  # W4 band - best for Planet Nine
            wavelength_score = 100
        elif wavelength >= 12.0:  # W3 band - good for Planet Nine
            wavelength_score = 80
        elif wavelength >= 4.6:  # W2 band - not great but potentially useful
            wavelength_score = 40
        else:  # W1 band - least useful
            wavelength_score = 20
        
        # Position component - closer is better
        position_score = 100 * (1 - separation / self.search_radius)
        
        # Band bonus
        band_bonus = 0
        if band == 'W4':
            band_bonus = 20  # W4 is ideal for Planet Nine
        elif band == 'W3':
            band_bonus = 10  # W3 is good for Planet Nine
        
        # Final score - weighted average
        score = (wavelength_score * 0.6) + (position_score * 0.3) + band_bonus
        
        return score
    
    def download(self, download_dir, filter_quality=None):
        """
        Download WISE FITS files.
        
        Args:
            download_dir (str): Directory to download files to
            filter_quality (str): Optional filter to only download certain quality matches
                                  ('perfect', 'good', or None for all)
        
        Returns:
            list: Paths to downloaded files
        """
        # Create download directory if it doesn't exist
        os.makedirs(download_dir, exist_ok=True)
        
        # Ensure results have been classified
        if not any('classification' in r for r in self.results):
            self.classify_results()
        
        # Filter results if requested
        if filter_quality:
            results_to_download = [r for r in self.results if r.get('classification') == filter_quality]
            logger.info(f"Filtered to {len(results_to_download)} '{filter_quality}' results for download")
        else:
            results_to_download = self.results
        
        if not results_to_download:
            logger.warning("No results to download")
            return []
        
        # Create a directory for organization
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        wise_dir = os.path.join(download_dir, f"wise_{timestamp}")
        os.makedirs(wise_dir, exist_ok=True)
        
        # Download each result
        downloaded_files = []
        for i, result in enumerate(results_to_download):
            try:
                logger.info(f"Downloading {i+1}/{len(results_to_download)}: {result['id']}")
                
                # Get file URL
                file_url = result.get('file_url')
                
                if not file_url:
                    logger.warning(f"No file URL found for {result['id']}")
                    continue
                
                # Determine local file path
                filename = file_url.split('/')[-1]
                local_file = os.path.join(wise_dir, filename)
                
                # Download the file
                logger.info(f"Downloading {filename} from {file_url}")
                
                try:
                    # Stream download to handle large files
                    with requests.get(file_url, stream=True, timeout=180) as response:
                        response.raise_for_status()
                        with open(local_file, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                    
                    logger.info(f"Downloaded {filename} to {local_file}")
                    downloaded_files.append(local_file)
                    
                    # Update result with local file path
                    result['local_file'] = local_file
                    
                except Exception as e:
                    logger.error(f"Error downloading {filename}: {e}")
                
            except Exception as e:
                logger.error(f"Error processing download for {result['id']}: {e}")
        
        logger.info(f"Downloaded {len(downloaded_files)} files to {wise_dir}")
        return downloaded_files

def main():
    """
    Run a standalone NASA WISE repository search.
    """
    import argparse
    from p9_data_core import load_config, load_trajectory
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Search NASA WISE for Planet Nine data")
    parser.add_argument("--config", default="config/search_params.yaml", help="Configuration file")
    parser.add_argument("--trajectory", default="config/p9_trajectory.txt", help="Trajectory file")
    parser.add_argument("--tight", action="store_true", help="Use tight search parameters")
    parser.add_argument("--loose", action="store_true", help="Use expanded search parameters")
    parser.add_argument("--output", default="data/reports", help="Output directory for reports")
    parser.add_argument("--download", action="store_true", help="Download matching files")
    parser.add_argument("--download-dir", default="data/fits", help="Directory to download files to")
    parser.add_argument("--quality", choices=["perfect", "good", "all"], default="all", 
                        help="Quality filter for downloads")
    parser.add_argument("--bands", type=str, default="W3,W4", 
                       help="Comma-separated list of WISE bands to search")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load trajectory
    trajectory_data = load_trajectory(args.trajectory)
    
    # Set search mode (tight is default)
    tight_search = not args.loose
    
    # Parse bands to search
    if args.bands:
        bands = args.bands.split(",")
    else:
        bands = ["W3", "W4"]
    
    # Initialize and run WISE handler
    wise_handler = NASAWISEHandler(config, trajectory_data, tight_search)
    wise_handler.search(specific_bands=bands)
    
    # Generate report
    report_file = wise_handler.generate_report(args.output)
    print(f"Report generated: {report_file}")
    
    # Download files if requested
    if args.download:
        filter_quality = None if args.quality == "all" else args.quality
        downloaded_files = wise_handler.download(args.download_dir, filter_quality)
        print(f"Downloaded {len(downloaded_files)} files")

if __name__ == "__main__":
    main()