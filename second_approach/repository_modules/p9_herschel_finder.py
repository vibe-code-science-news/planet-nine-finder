#!/usr/bin/env python
"""
Planet Nine Herschel Repository Handler

This module provides functionality for searching the Herschel Science Archive (HSA)
for data that might contain Planet Nine, focusing on PACS and SPIRE instruments
which operate in the far-infrared range ideal for cold object detection.
"""

import os
import logging
import numpy as np
from datetime import datetime
import requests
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.io.fits as fits
from astropy.table import Table

from p9_data_core import RepositoryHandler, get_position_for_date, calculate_separation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HerschelHandler(RepositoryHandler):
    """Handler for Herschel Space Observatory data searches"""
    
    def __init__(self, config, trajectory_data, tight_search=True):
        """
        Initialize the Herschel repository handler.
        
        Args:
            config (dict): Configuration dictionary
            trajectory_data (dict): Planet Nine trajectory data
            tight_search (bool): Whether to use tight search parameters
        """
        super().__init__(config, trajectory_data, tight_search)
        self.name = "HERSCHEL"
        
        # Define Herschel-specific properties
        self.instruments = {
            "PACS": {
                "wavelengths": [70.0, 100.0, 160.0],
                "years": [2009, 2013],
                "relevant": True,
                "priority": "high",
                "description": "Photodetector Array Camera and Spectrometer"
            },
            "SPIRE": {
                "wavelengths": [250.0, 350.0, 500.0],
                "years": [2009, 2013],
                "relevant": True,
                "priority": "high",
                "description": "Spectral and Photometric Imaging Receiver"
            },
            "HIFI": {
                "wavelengths": [157.0, 212.0, 240.0],  # Approximate range
                "years": [2009, 2013],
                "relevant": False,
                "priority": "low",
                "description": "Heterodyne Instrument for the Far Infrared (spectroscopy)"
            }
        }
        
        # Set minimum relevant wavelength from config
        self.min_relevant_wavelength = config.get('wavelength_ranges', {}).get('acceptable', [20, 500])[0]
        
        # Herschel mission dates
        self.mission_start = 2009.4  # May 2009
        self.mission_end = 2013.3    # April 2013
        
        # Base URL for HSA API (Herschel Science Archive)
        self.hsa_base_url = "http://archives.esac.esa.int/hsa/aio/jsp/product-action.jsp"
        self.tap_base_url = "https://hsa.esac.esa.int/taphs/tap"
        
        # Define URL for metadata queries
        self.metadata_url = "https://archives.esac.esa.int/hsa/whsa-tap-server/tap/sync"
        
        # User agent for API requests
        self.user_agent = "Planet9Finder/1.0"
        
        # Headers for API requests
        self.headers = {
            "User-Agent": self.user_agent,
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        
    def search(self, specific_years=None, specific_instruments=None):
        """
        Search Herschel Science Archive for data.
        
        Args:
            specific_years (list): Optional list of years to search
            specific_instruments (list): Optional list of instruments to search
            
        Returns:
            list: Search results
        """
        logger.info(f"Searching Herschel archive with radius {self.search_radius} degrees")
        
        # Clear previous results
        self.results = []
        
        # Filter instruments if not specified
        if not specific_instruments:
            relevant_instruments = [inst for inst, data in self.instruments.items() 
                                   if data.get('relevant', False)]
            if relevant_instruments:
                specific_instruments = relevant_instruments
                logger.info(f"Filtering for relevant instruments: {specific_instruments}")
        
        # Determine years to search
        # Herschel only operated from 2009-2013
        if specific_years:
            # Filter for years in Herschel mission lifetime
            years_to_search = [year for year in specific_years
                             if year >= self.mission_start and year <= self.mission_end]
        else:
            # Use all years in trajectory data that overlap with Herschel mission
            years_to_search = [year for year in sorted(self.trajectory_data.keys())
                             if year >= self.mission_start and year <= self.mission_end]
        
        if not years_to_search:
            logger.warning("No valid years found for Herschel search (mission: 2009-2013)")
            return self.results
        
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
                # Search for observations
                for instrument in specific_instruments:
                    logger.info(f"Querying {instrument} observations")
                    
                    # Construct ADQL query for Herschel TAP service
                    query = self._construct_adql_query(ra, dec, instrument)
                    
                    # Execute query
                    observations = self._execute_tap_query(query)
                    
                    if observations and len(observations) > 0:
                        logger.info(f"Found {len(observations)} {instrument} observations for year {year}")
                        
                        # Process observations
                        for obs in observations:
                            result = self._process_observation(obs, instrument, year, ra, dec)
                            if result:
                                self.results.append(result)
                    else:
                        logger.info(f"No {instrument} observations found for year {year}")
            
            except Exception as e:
                logger.error(f"Error searching Herschel for year {year}: {e}")
        
        # Sort results by relevance (wavelength and separation)
        if self.results:
            self.results.sort(key=lambda x: 
                             (x.get('classification', 'expanded') != 'perfect',
                              x.get('classification', 'expanded') != 'good',
                              x.get('separation_deg', 999)))
        
        logger.info(f"Herschel search complete. Found {len(self.results)} relevant results.")
        return self.results
    
    def _construct_adql_query(self, ra, dec, instrument):
        """
        Construct an ADQL query for the Herschel TAP service.
        
        Args:
            ra (float): Right Ascension in degrees
            dec (float): Declination in degrees
            instrument (str): Herschel instrument
            
        Returns:
            str: ADQL query
        """
        # Convert search radius to degrees for query
        radius_deg = self.search_radius
        
        # Basic query template
        query = f"""
        SELECT 
            o.observation_id, o.proposal_id, o.instrument, o.target_name,
            o.ra, o.dec, o.observation_start, o.observation_end,
            i.filename, i.file_type, i.wavelength, i.obsid, i.object, 
            i.instrument AS image_instrument, i.filter, i.processing_level
        FROM 
            hsa.observation AS o
        JOIN 
            hsa.image AS i ON o.observation_id = i.obsid
        WHERE 
            o.instrument = '{instrument}'
            AND 1=CONTAINS(
                POINT('ICRS', o.ra, o.dec),
                CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
            )
        """
        
        # Add wavelength filter for relevant observations
        if instrument == "PACS":
            # PACS bands: 70, 100, 160 microns
            query += " AND (i.wavelength >= 60.0)"
        elif instrument == "SPIRE":
            # SPIRE bands: 250, 350, 500 microns
            query += " AND (i.wavelength >= 200.0)"
        
        return query
    
    def _execute_tap_query(self, query):
        """
        Execute a TAP query against the Herschel Science Archive.
        
        Args:
            query (str): ADQL query
            
        Returns:
            astropy.table.Table: Query results
        """
        try:
            # Parameters for TAP request
            params = {
                "REQUEST": "doQuery",
                "LANG": "ADQL",
                "FORMAT": "JSON",
                "QUERY": query
            }
            
            # Execute request
            response = requests.post(
                self.tap_base_url,
                data=params,
                headers=self.headers,
                timeout=120
            )
            
            if response.status_code == 200:
                # Parse JSON response
                data = response.json()
                
                # Convert to Astropy table
                if 'data' in data:
                    table = Table(rows=data['data'], names=data['columns'])
                    return table
                else:
                    logger.warning(f"TAP query returned no data: {data}")
                    return None
            else:
                logger.error(f"TAP query failed with status {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error executing TAP query: {e}")
            return None
    
    def _process_observation(self, obs, instrument, search_year, search_ra, search_dec):
        """
        Process a single Herschel observation result.
        
        Args:
            obs: Observation table row
            instrument: Instrument name
            search_year: Year being searched
            search_ra, search_dec: Search position
            
        Returns:
            dict: Processed observation data or None if invalid
        """
        try:
            # Extract basic metadata
            obs_id = obs['observation_id']
            
            # Get observation date
            if 'observation_start' in obs.colnames and obs['observation_start'] is not None:
                try:
                    obs_date = Time(obs['observation_start'], format='isot').datetime
                    obs_year = obs_date.year + (obs_date.month - 1) / 12 + obs_date.day / 365.25
                except:
                    # If date conversion fails, use the search year
                    obs_date = None
                    obs_year = search_year
            else:
                obs_date = None
                obs_year = search_year
            
            # Get observation coordinates
            if 'ra' in obs.colnames and 'dec' in obs.colnames:
                obs_ra = float(obs['ra'])
                obs_dec = float(obs['dec'])
                
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
                p9_ra, p9_dec = search_ra, search_dec
            
            # Determine wavelength
            wavelength = self._get_wavelength(obs, instrument)
            
            # Check if the wavelength is in the relevant range
            if wavelength is None or wavelength < self.min_relevant_wavelength:
                logger.debug(f"Skipping observation {obs_id}: wavelength {wavelength} below minimum {self.min_relevant_wavelength}")
                return None
            
            # Create result dictionary
            result = {
                'id': str(obs_id),
                'repository': 'HERSCHEL',
                'instrument': instrument,
                'sub_instrument': obs.get('image_instrument', instrument),
                'filter': obs.get('filter', ''),
                'wavelength': wavelength,
                'obs_date': obs_date.isoformat() if obs_date else None,
                'obs_year': obs_year,
                'ra': obs_ra,
                'dec': obs_dec,
                'search_ra': p9_ra,
                'search_dec': p9_dec,
                'separation_deg': separation,
                'target_name': obs.get('target_name', ''),
                'filename': obs.get('filename', ''),
                'file_type': obs.get('file_type', ''),
                'processing_level': obs.get('processing_level', ''),
                'proposal_id': obs.get('proposal_id', ''),
                'relevance_score': self._calculate_relevance_score(wavelength, separation, instrument),
                'raw_metadata': {k: str(obs[k]) for k in obs.colnames}
            }
            
            return result
            
        except Exception as e:
            logger.warning(f"Error processing observation {obs.get('observation_id', 'unknown')}: {e}")
            return None
    
    def _get_wavelength(self, obs, instrument):
        """
        Determine observation wavelength based on instrument and filters.
        
        Args:
            obs: Observation table row
            instrument: Instrument name
            
        Returns:
            float: Estimated wavelength in microns or None if unknown
        """
        # First check if wavelength is directly provided
        if 'wavelength' in obs.colnames and obs['wavelength'] is not None:
            try:
                return float(obs['wavelength'])
            except (ValueError, TypeError):
                pass
        
        # Check filter information
        filter_val = obs.get('filter', '')
        
        # PACS specific filter logic
        if instrument == 'PACS':
            if 'blue' in str(filter_val).lower() or '70' in str(filter_val):
                return 70.0
            elif 'green' in str(filter_val).lower() or '100' in str(filter_val):
                return 100.0
            elif 'red' in str(filter_val).lower() or '160' in str(filter_val):
                return 160.0
            else:
                # Default to middle wavelength for PACS
                return 100.0
        
        # SPIRE specific filter logic
        elif instrument == 'SPIRE':
            if 'PSW' in str(filter_val) or '250' in str(filter_val):
                return 250.0
            elif 'PMW' in str(filter_val) or '350' in str(filter_val):
                return 350.0
            elif 'PLW' in str(filter_val) or '500' in str(filter_val):
                return 500.0
            else:
                # Default to middle wavelength for SPIRE
                return 350.0
        
        # HIFI (less relevant for Planet Nine, but included for completeness)
        elif instrument == 'HIFI':
            # HIFI is a spectrometer covering 157-212 μm and 240-625 μm
            # Default to middle range
            return 240.0
        
        # Default case - use instrument's default wavelength
        if instrument in self.instruments:
            wavelengths = self.instruments[instrument]['wavelengths']
            if wavelengths:
                # Use the middle wavelength value
                return wavelengths[len(wavelengths) // 2]
        
        # No wavelength could be determined
        return None
    
    def _calculate_relevance_score(self, wavelength, separation, instrument):
        """
        Calculate a relevance score for a detection.
        
        Args:
            wavelength: Wavelength in microns
            separation: Separation in degrees
            instrument: Instrument name
            
        Returns:
            float: Relevance score (higher is better)
        """
        # Base score - wavelength component
        if 70 <= wavelength <= 100:  # Ideal PACS range for Planet Nine
            wavelength_score = 100
        elif 50 <= wavelength <= 200:  # Good range (includes PACS 160)
            wavelength_score = 80
        elif 200 <= wavelength <= 350:  # SPIRE shorter wavelengths
            wavelength_score = 70
        elif 350 <= wavelength <= 500:  # SPIRE longer wavelengths
            wavelength_score = 60
        elif wavelength > 500:
            wavelength_score = 40
        else:  # wavelength < 50
            wavelength_score = 30
        
        # Position component - closer is better
        position_score = 100 * (1 - separation / self.search_radius)
        
        # Instrument bonus
        instrument_bonus = 0
        if instrument == 'PACS':
            instrument_bonus = 20  # PACS is ideal for Planet Nine
        elif instrument == 'SPIRE':
            instrument_bonus = 10  # SPIRE is good but longer wavelength
        
        # Final score - weighted average
        score = (wavelength_score * 0.6) + (position_score * 0.3) + instrument_bonus
        
        return score
    
    def download(self, download_dir, filter_quality=None):
        """
        Download data files from Herschel Science Archive.
        
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
        herschel_dir = os.path.join(download_dir, f"herschel_{timestamp}")
        os.makedirs(herschel_dir, exist_ok=True)
        
        # Download each result
        downloaded_files = []
        for i, result in enumerate(results_to_download):
            try:
                logger.info(f"Downloading {i+1}/{len(results_to_download)}: {result['id']}")
                
                # Get file information
                observation_id = result['id']
                filename = result.get('filename')
                
                if not filename:
                    logger.warning(f"No filename found for {observation_id}, skipping")
                    continue
                
                # Determine local file path
                local_file = os.path.join(herschel_dir, filename)
                
                # Download the file
                success = self._download_file(observation_id, filename, local_file)
                
                if success:
                    downloaded_files.append(local_file)
                    # Update result with local file path
                    result['local_files'] = [local_file]
                
            except Exception as e:
                logger.error(f"Error downloading {result['id']}: {e}")
        
        logger.info(f"Downloaded {len(downloaded_files)} files to {herschel_dir}")
        return downloaded_files
    
    def _download_file(self, observation_id, filename, local_file):
        """
        Download a specific file from Herschel Science Archive.
        
        Args:
            observation_id (str): Observation ID
            filename (str): File name
            local_file (str): Local file path to save to
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Construct download URL
            download_url = f"{self.hsa_base_url}?PROTOCOL=HTTP&OBSERVATION_ID={observation_id}&PRODUCT_ID={filename}"
            
            # Download the file
            response = requests.get(
                download_url,
                headers={"User-Agent": self.user_agent},
                stream=True,
                timeout=300
            )
            
            if response.status_code == 200:
                with open(local_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"Successfully downloaded {filename} to {local_file}")
                return True
            else:
                logger.error(f"Failed to download {filename}: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            return False

def main():
    """
    Run a standalone Herschel repository search.
    """
    import argparse
    from p9_data_core import load_config, load_trajectory
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Search Herschel for Planet Nine data")
    parser.add_argument("--config", default="config/search_params.yaml", help="Configuration file")
    parser.add_argument("--trajectory", default="config/p9_trajectory.txt", help="Trajectory file")
    parser.add_argument("--tight", action="store_true", help="Use tight search parameters")
    parser.add_argument("--loose", action="store_true", help="Use expanded search parameters")
    parser.add_argument("--output", default="data/reports", help="Output directory for reports")
    parser.add_argument("--download", action="store_true", help="Download matching files")
    parser.add_argument("--download-dir", default="data/fits", help="Directory to download files to")
    parser.add_argument("--quality", choices=["perfect", "good", "all"], default="all", 
                        help="Quality filter for downloads")
    parser.add_argument("--instruments", type=str, default="PACS,SPIRE", 
                       help="Comma-separated list of instruments to search")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load trajectory
    trajectory_data = load_trajectory(args.trajectory)
    
    # Set search mode (tight is default)
    tight_search = not args.loose
    
    # Parse instruments to search
    if args.instruments:
        instruments = args.instruments.split(",")
    else:
        instruments = None
    
    # Initialize and run Herschel handler
    herschel_handler = HerschelHandler(config, trajectory_data, tight_search)
    herschel_handler.search(specific_instruments=instruments)
    
    # Generate report
    report_file = herschel_handler.generate_report(args.output)
    print(f"Report generated: {report_file}")
    
    # Download files if requested
    if args.download:
        filter_quality = None if args.quality == "all" else args.quality
        downloaded_files = herschel_handler.download(args.download_dir, filter_quality)
        print(f"Downloaded {len(downloaded_files)} files")

if __name__ == "__main__":
    main()