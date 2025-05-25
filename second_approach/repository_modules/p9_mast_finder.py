#!/usr/bin/env python
"""
Planet Nine MAST Repository Handler

This module provides functionality for searching the MAST (Mikulski Archive for Space
Telescopes) repository for data that might contain Planet Nine.
"""

import os
import logging
import numpy as np
from datetime import datetime
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.mast import Observations, ObservationsClass, Catalogs

from p9_data_core import RepositoryHandler, get_position_for_date, calculate_separation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MASTHandler(RepositoryHandler):
    """Handler for MAST repository searches"""
    
    def __init__(self, config, trajectory_data, tight_search=True):
        """
        Initialize the MAST repository handler.
        
        Args:
            config (dict): Configuration dictionary
            trajectory_data (dict): Planet Nine trajectory data
            tight_search (bool): Whether to use tight search parameters
        """
        super().__init__(config, trajectory_data, tight_search)
        self.name = "MAST"
        
        # Define MAST-specific properties
        self.instruments = {
            "SPITZER_SHA": {
                "wavelengths": {
                    "IRAC": [3.6, 4.5, 5.8, 8.0],
                    "MIPS": [24.0, 70.0, 160.0]
                },
                "years": [2003, 2020],
                "relevant": True,  # Relevant for Planet Nine
                "ir_capable": True  # Has far-IR capabilities
            },
            "HST": {
                "wavelengths": {
                    "WFC3": [0.9, 1.7],  # Near-IR only, approximated
                    "NICMOS": [0.8, 2.5]
                },
                "years": [1990, 2023],
                "relevant": False,  # Not ideal for Planet Nine
                "ir_capable": False  # Only near-IR, not far-IR
            },
            "JWST": {
                "wavelengths": {
                    "MIRI": [5.0, 28.0],
                    "NIRCam": [0.6, 5.0]
                },
                "years": [2022, 2025],
                "relevant": True,  # Mid-IR is somewhat relevant
                "ir_capable": True  # Has mid-IR capabilities
            },
            "TESS": {
                "wavelengths": {
                    "Photometer": [0.6, 1.0]
                },
                "years": [2018, 2025],
                "relevant": False,  # Optical only, not relevant
                "ir_capable": False  # No IR capabilities
            },
            "GALEX": {
                "wavelengths": {
                    "FUV": [0.15, 0.28],
                    "NUV": [0.175, 0.28]
                },
                "years": [2003, 2013],
                "relevant": False,  # UV only, not relevant
                "ir_capable": False  # No IR capabilities
            }
            # Other MAST instruments can be added as needed
        }
        
        # Define relevant wavelength range
        self.min_relevant_wavelength = config.get('wavelength_ranges', {}).get('acceptable', [20, 500])[0]
        self.max_separation = min(0.5, self.search_radius * 1.5)  # Strict separation limit
        
        # Set up astroquery MAST interface
        self.mast = Observations
        
    def search(self, specific_years=None, specific_instruments=None):
        """
        Search MAST repository for data.
        
        Args:
            specific_years (list): Optional list of years to search
            specific_instruments (list): Optional list of instruments to search
            
        Returns:
            list: Search results
        """
        logger.info(f"Searching MAST with radius {self.search_radius} degrees")
        logger.info(f"Using strict filters: max separation {self.max_separation}°, min wavelength {self.min_relevant_wavelength}μm")
        
        # Clear previous results
        self.results = []
        
        # Filter instruments for IR-capable ones if not specified
        if not specific_instruments:
            ir_instruments = [inst for inst, data in self.instruments.items() 
                            if data.get('ir_capable', False) or data.get('relevant', False)]
            if ir_instruments:
                specific_instruments = ir_instruments
                logger.info(f"Filtering for IR-capable instruments: {specific_instruments}")
        
        # Determine years to search
        if specific_years:
            years_to_search = specific_years
        else:
            # Use all years in trajectory data
            years_to_search = sorted(self.trajectory_data.keys())
        
        logger.info(f"Searching for years: {years_to_search}")
        
        # Iterate through years
        for year in years_to_search:
            # Get expected position for this year
            ra, dec = self.trajectory_data[year]['ra'], self.trajectory_data[year]['dec']
            
            # Create coordinate object
            coords = SkyCoord(ra, dec, unit="deg")
            
            # Log search details
            logger.info(f"Searching year {year} at position RA={ra:.5f}, Dec={dec:.5f}")
            
            try:
                # Perform search
                obs_table = self.mast.query_region(
                    coordinates=coords,
                    radius=self.search_radius * u.deg
                )
                
                # Process results
                if obs_table and len(obs_table) > 0:
                    logger.info(f"Found {len(obs_table)} observations for year {year}")
                    
                    # Filter by instruments if specified
                    if specific_instruments:
                        obs_table = obs_table[
                            np.isin(obs_table['obs_collection'], specific_instruments)
                        ]
                        logger.info(f"Filtered to {len(obs_table)} observations by instrument")
                    
                    # Process each observation
                    for obs in obs_table:
                        result = self._process_observation(obs, year, ra, dec)
                        if result:
                            self.results.append(result)
                else:
                    logger.info(f"No observations found for year {year}")
            
            except Exception as e:
                logger.error(f"Error searching MAST for year {year}: {e}")
        
        logger.info(f"MAST search complete. Found {len(self.results)} relevant results.")
        return self.results
    
    def _process_observation(self, obs, search_year, search_ra, search_dec):
        """
        Process a single MAST observation result.
        
        Args:
            obs: Observation table row
            search_year: Year being searched
            search_ra, search_dec: Search position
            
        Returns:
            dict: Processed observation data or None if invalid
        """
        try:
            # Extract basic metadata
            obs_id = obs['obs_id']
            collection = obs['obs_collection']
            
            # Get observation date
            if 't_min' in obs.colnames and obs['t_min'] is not None:
                try:
                    t = Time(obs['t_min'], format='mjd')
                    obs_date = t.datetime
                    obs_year = obs_date.year + (obs_date.month - 1) / 12 + obs_date.day / 365.25
                except:
                    # If date conversion fails, use the search year
                    obs_date = None
                    obs_year = search_year
            else:
                obs_date = None
                obs_year = search_year
            
            # Get observation coordinates if available
            if 's_ra' in obs.colnames and 's_dec' in obs.colnames:
                obs_ra = obs['s_ra']
                obs_dec = obs['s_dec']
                
                # Calculate separation from expected Planet Nine position
                if obs_year != search_year:
                    # Recalculate expected position for the observation year
                    p9_ra, p9_dec = get_position_for_date(self.trajectory_data, obs_year)
                else:
                    p9_ra, p9_dec = search_ra, search_dec
                
                separation = calculate_separation(p9_ra, p9_dec, obs_ra, obs_dec)
            else:
                obs_ra = None
                obs_dec = None
                separation = self.search_radius  # Conservative estimate
            
            # Determine wavelength based on instrument
            wavelength = self._get_wavelength(collection, obs)
            
            # Create result dictionary
            result = {
                'id': str(obs_id),
                'repository': 'MAST',
                'collection': collection,
                'instrument': obs.get('instrument_name', collection),
                'wavelength': wavelength,
                'obs_date': obs_date.isoformat() if obs_date else None,
                'obs_year': obs_year,
                'ra': obs_ra,
                'dec': obs_dec,
                'search_ra': p9_ra,
                'search_dec': p9_dec,
                'separation_deg': separation,
                'dataproduct_type': obs.get('dataproduct_type', None),
                'target_name': obs.get('target_name', None),
                'raw_metadata': {k: str(obs[k]) for k in obs.colnames}
            }
            
            return result
            
        except Exception as e:
            logger.warning(f"Error processing observation {obs.get('obs_id', 'unknown')}: {e}")
            return None
    
    def _get_wavelength(self, collection, obs):
        """
        Determine observation wavelength based on instrument and filters.
        
        Args:
            collection: Observation collection
            obs: Observation table row
            
        Returns:
            float: Estimated wavelength in microns or None if unknown
        """
        # First check if collection is known
        if collection in self.instruments:
            instrument_info = self.instruments[collection]
            
            # Try to determine specific instrument
            instrument = obs.get('instrument_name', '')
            
            # Check if we know the wavelengths for this instrument
            if instrument in instrument_info['wavelengths']:
                wavelengths = instrument_info['wavelengths'][instrument]
                
                # If there are multiple wavelengths, try to determine which one
                # based on filters or other metadata
                if len(wavelengths) > 1:
                    # Try to determine from filters
                    filters = obs.get('filters', '')
                    
                    # Spitzer-specific logic
                    if collection == 'SPITZER_SHA':
                        if instrument == 'IRAC':
                            if 'CH1' in filters:
                                return wavelengths[0]  # 3.6 microns
                            elif 'CH2' in filters:
                                return wavelengths[1]  # 4.5 microns
                            elif 'CH3' in filters:
                                return wavelengths[2]  # 5.8 microns
                            elif 'CH4' in filters:
                                return wavelengths[3]  # 8.0 microns
                        elif instrument == 'MIPS':
                            if '24' in filters:
                                return wavelengths[0]  # 24 microns
                            elif '70' in filters:
                                return wavelengths[1]  # 70 microns
                            elif '160' in filters:
                                return wavelengths[2]  # 160 microns
                    
                    # If we couldn't determine the specific wavelength,
                    # return the middle of the range
                    return np.mean(wavelengths)
                else:
                    # Only one wavelength for this instrument
                    return wavelengths[0]
        
        # If we don't know the wavelength, make a best guess based on collection
        if collection == 'SPITZER_SHA':
            # If we reach here, we don't know the specific instrument
            # Let's make a reasonable guess based on relevance to Planet Nine
            logger.debug(f"Unknown Spitzer instrument, guessing MIPS (70µm)")
            return 70.0  # Guess MIPS as it's more relevant for Planet Nine
        elif collection == 'JWST':
            logger.debug(f"Unknown JWST instrument, guessing MIRI (20µm)")
            return 20.0  # Guess MIRI
        elif collection == 'HST':
            logger.debug(f"Unknown HST instrument, guessing near-IR (1.6µm)")
            return 1.6   # Guess near-IR
        elif collection == 'GALEX':
            logger.debug(f"GALEX observation, using UV (0.2µm)")
            return 0.2   # Ultraviolet
        elif collection == 'TESS':
            logger.debug(f"TESS observation, using optical (0.7µm)")
            return 0.7   # Optical
        
        # Default unknown - explicitly return None
        logger.debug(f"Unknown wavelength for collection {collection}")
        return None
    
    def download(self, download_dir, filter_quality=None):
        """
        Download data files from MAST.
        
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
        mast_dir = os.path.join(download_dir, f"mast_{timestamp}")
        os.makedirs(mast_dir, exist_ok=True)
        
        # Download each result
        downloaded_files = []
        for i, result in enumerate(results_to_download):
            try:
                logger.info(f"Downloading {i+1}/{len(results_to_download)}: {result['id']}")
                
                # Get observation data products
                data_products = self.mast.get_product_list(result['id'])
                
                # Filter for relevant data products (e.g., fits files)
                fits_products = data_products[data_products['productType'] == 'SCIENCE']
                
                if len(fits_products) == 0:
                    logger.warning(f"No science products found for {result['id']}")
                    continue
                
                # Download the products
                download_path = self.mast.download_products(
                    fits_products,
                    download_dir=mast_dir,
                    cache=False
                )
                
                # Record downloaded files
                if download_path and len(download_path) > 0:
                    downloaded_files.extend(download_path['Local Path'])
                    
                    # Update result with local file paths
                    result['local_files'] = download_path['Local Path'].tolist()
                
            except Exception as e:
                logger.error(f"Error downloading {result['id']}: {e}")
        
        logger.info(f"Downloaded {len(downloaded_files)} files to {mast_dir}")
        return downloaded_files

def main():
    """
    Run a standalone MAST repository search.
    """
    import argparse
    from p9_data_core import load_config, load_trajectory
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Search MAST for Planet Nine data")
    parser.add_argument("--config", default="config/search_params.yaml", help="Configuration file")
    parser.add_argument("--trajectory", default="config/p9_trajectory.txt", help="Trajectory file")
    parser.add_argument("--tight", action="store_true", help="Use tight search parameters")
    parser.add_argument("--loose", action="store_true", help="Use expanded search parameters")
    parser.add_argument("--output", default="data/reports", help="Output directory for reports")
    parser.add_argument("--download", action="store_true", help="Download matching files")
    parser.add_argument("--download-dir", default="data/fits", help="Directory to download files to")
    parser.add_argument("--quality", choices=["perfect", "good", "all"], default="all", 
                        help="Quality filter for downloads")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load trajectory
    trajectory_data = load_trajectory(args.trajectory)
    
    # Set search mode (tight is default)
    tight_search = not args.loose
    
    # Initialize and run MAST handler
    mast_handler = MASTHandler(config, trajectory_data, tight_search)
    mast_handler.search()
    
    # Generate report
    report_file = mast_handler.generate_report(args.output)
    print(f"Report generated: {report_file}")
    
    # Download files if requested
    if args.download:
        filter_quality = None if args.quality == "all" else args.quality
        downloaded_files = mast_handler.download(args.download_dir, filter_quality)
        print(f"Downloaded {len(downloaded_files)} files")

if __name__ == "__main__":
    main()