#!/usr/bin/env python
"""
Planet Nine IRSA Repository Handler

This module provides functionality for searching the IRSA (Infrared Science Archive)
repository for data that might contain Planet Nine, with emphasis on WISE, AKARI,
and other infrared surveys.
"""

import os
import logging
import numpy as np
from datetime import datetime
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.irsa import Irsa

from p9_data_core import RepositoryHandler, get_position_for_date, calculate_separation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IRSAHandler(RepositoryHandler):
    """Handler for IRSA repository searches (WISE, AKARI, etc.)"""
    
    def __init__(self, config, trajectory_data, tight_search=True):
        """
        Initialize the IRSA repository handler.
        
        Args:
            config (dict): Configuration dictionary
            trajectory_data (dict): Planet Nine trajectory data
            tight_search (bool): Whether to use tight search parameters
        """
        super().__init__(config, trajectory_data, tight_search)
        self.name = "IRSA"
        
        # Define IRSA catalog-specific properties
        self.catalogs = {
            "wise": {
                "primary": "allwise_p3as_psd",
                "alternates": ["allsky_4band_p3as_psd", "catwise_2020"],
                "wavelengths": [3.4, 4.6, 12, 22],
                "years": [2010, 2023],  # WISE + NEOWISE years
                "relevant": True,
                "priority": "high"
            },
            "akari": {
                "primary": "akari_fis",
                "alternates": ["akari_irc"],
                "wavelengths": [65, 90, 140, 160],  # FIS bands
                "years": [2006, 2007],
                "relevant": True,
                "priority": "high"
            },
            "iras": {
                "primary": "iraspsc",
                "alternates": ["irasfsc"],
                "wavelengths": [12, 25, 60, 100],
                "years": [1983, 1983],
                "relevant": True,
                "priority": "medium"
            },
            "herschel": {
                "primary": "herschel.hpdp_images",
                "alternates": ["herschel.phpdp_images", "herschel.hermes_images"],
                "wavelengths": [70, 160, 250, 350, 500],
                "years": [2009, 2013],
                "relevant": True,
                "priority": "high"
            }
        }
        
        # Get preferred catalogs from config
        repo_config = config.get('repositories', {}).get('irsa', {})
        self.priority_catalogs = repo_config.get('priority_catalogs', ['wise', 'akari', 'herschel'])
        
        # Set up astroquery IRSA interface
        self.irsa = Irsa
        
        # Record what catalogs are available
        self.available_catalogs = self._get_available_catalogs()
    
    def _get_available_catalogs(self):
        """Get available IRSA catalogs."""
        try:
            all_catalogs = self.irsa.list_catalogs()
            logger.info(f"Found {len(all_catalogs)} IRSA catalogs")
            return all_catalogs
        except Exception as e:
            logger.error(f"Error getting IRSA catalogs: {e}")
            return []
    
    def search(self, specific_years=None, specific_catalogs=None):
        """
        Search IRSA catalogs for Planet Nine candidates.
        
        Args:
            specific_years (list): Optional list of years to search
            specific_catalogs (list): Optional list of catalogs to search
            
        Returns:
            list: Search results
        """
        logger.info(f"Searching IRSA with radius {self.search_radius} degrees")
        
        # Clear previous results
        self.results = []
        
        # Determine catalogs to search
        if specific_catalogs:
            catalogs_to_search = specific_catalogs
        else:
            # Use priority catalogs from config
            catalogs_to_search = []
            for cat_type in self.priority_catalogs:
                if cat_type in self.catalogs:
                    cat_info = self.catalogs[cat_type]
                    primary_cat = cat_info['primary']
                    
                    # Check if primary catalog is available
                    if primary_cat in self.available_catalogs:
                        catalogs_to_search.append(primary_cat)
                    else:
                        # Try alternate catalogs
                        for alt_cat in cat_info.get('alternates', []):
                            if alt_cat in self.available_catalogs:
                                catalogs_to_search.append(alt_cat)
                                break
        
        logger.info(f"Searching catalogs: {catalogs_to_search}")
        
        # Determine years to search
        if specific_years:
            years_to_search = specific_years
        else:
            # Use all years in trajectory data
            years_to_search = sorted(self.trajectory_data.keys())
        
        logger.info(f"Searching for years: {years_to_search}")
        
        # Process each catalog
        for catalog in catalogs_to_search:
            # Determine catalog type (wise, akari, etc.)
            catalog_type = self._get_catalog_type(catalog)
            logger.info(f"Searching {catalog} (type: {catalog_type})")
            
            # Get relevant years for this catalog
            if catalog_type and catalog_type in self.catalogs:
                cat_years = self.catalogs[catalog_type]['years']
                relevant_years = [y for y in years_to_search 
                                if y >= cat_years[0] and y <= cat_years[1]]
            else:
                relevant_years = years_to_search
            
            if not relevant_years:
                logger.info(f"No relevant years for {catalog}, skipping")
                continue
            
            # For each year, query the catalog
            for year in relevant_years:
                # Get expected position for this year
                ra, dec = self.trajectory_data[year]['ra'], self.trajectory_data[year]['dec']
                
                # Create coordinate object
                coords = SkyCoord(ra, dec, unit="deg")
                
                # Log search details
                logger.info(f"Searching {catalog} for year {year} at position RA={ra:.5f}, Dec={dec:.5f}")
                
                try:
                    # Query the catalog with appropriate radius
                    result_table = self.irsa.query_region(
                        coords, 
                        catalog=catalog,
                        radius=self.search_radius * u.deg
                    )
                    
                    # Process results
                    if result_table and len(result_table) > 0:
                        logger.info(f"Found {len(result_table)} sources in {catalog} near position for year {year}")
                        
                        # Process each source
                        for i, source in enumerate(result_table):
                            # Process the source and add to results if valid
                            result = self._process_source(source, catalog, catalog_type, year, ra, dec)
                            if result:
                                self.results.append(result)
                    else:
                        logger.info(f"No sources found in {catalog} for year {year}")
                
                except Exception as e:
                    logger.error(f"Error querying {catalog} for year {year}: {e}")
        
        # Sort results by relevance (using wavelength as proxy)
        if self.results:
            self.results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        logger.info(f"IRSA search complete. Found {len(self.results)} results.")
        return self.results
    
    def _get_catalog_type(self, catalog):
        """
        Determine the catalog type (wise, akari, iras, etc.).
        
        Args:
            catalog (str): Catalog name
            
        Returns:
            str: Catalog type or None if unknown
        """
        catalog_lower = catalog.lower()
        
        if 'wise' in catalog_lower:
            return 'wise'
        elif 'akari' in catalog_lower:
            return 'akari'
        elif 'iras' in catalog_lower:
            return 'iras'
        elif 'herschel' in catalog_lower:
            return 'herschel'
            
        # Check if it's in the catalog types
        for cat_type, cat_info in self.catalogs.items():
            if catalog == cat_info['primary'] or catalog in cat_info.get('alternates', []):
                return cat_type
        
        return None
    
    def _process_source(self, source, catalog, catalog_type, search_year, search_ra, search_dec):
        """
        Process a source from IRSA catalog results.
        
        Args:
            source: Source table row
            catalog: Catalog name
            catalog_type: Catalog type (wise, akari, etc.)
            search_year: Year being searched
            search_ra, search_dec: Search position
            
        Returns:
            dict: Processed source data or None if invalid
        """
        try:
            # Extract position data - column names depend on catalog
            source_ra, source_dec = self._get_source_position(source, catalog_type)
            
            if source_ra is None or source_dec is None:
                logger.warning(f"Could not determine position for source in {catalog}")
                return None
            
            # Calculate separation from expected Planet Nine position
            p9_ra, p9_dec = search_ra, search_dec
            separation = calculate_separation(p9_ra, p9_dec, source_ra, source_dec)
            
            # Skip if separation is too large
            if separation > self.search_radius:
                return None
            
            # Get observations year - depends on catalog
            obs_year = self._get_observation_year(source, catalog_type) or search_year
            
            # Calculate wavelength and flux information
            wavelength_info = self._get_wavelength_info(source, catalog_type)
            
            # Skip if no wavelength info available
            if not wavelength_info:
                return None
            
            # Calculate a relevance score based on wavelength
            relevance_score = self._calculate_relevance_score(
                wavelength_info['wavelength'],
                separation, 
                catalog_type
            )
            
            # Create result dictionary
            result = {
                'id': self._get_source_id(source, catalog_type),
                'repository': 'IRSA',
                'catalog': catalog,
                'catalog_type': catalog_type,
                'wavelength': wavelength_info['wavelength'],
                'flux': wavelength_info.get('flux'),
                'flux_units': wavelength_info.get('flux_units'),
                'snr': wavelength_info.get('snr'),
                'obs_year': obs_year,
                'ra': source_ra,
                'dec': source_dec,
                'search_ra': p9_ra,
                'search_dec': p9_dec,
                'separation_deg': separation,
                'relevance_score': relevance_score,
                'raw_metadata': {k: str(source[k]) for k in source.colnames}
            }
            
            return result
            
        except Exception as e:
            logger.warning(f"Error processing source: {e}")
            if hasattr(source, 'colnames'):
                logger.debug(f"Source columns: {source.colnames}")
            return None
    
    def _get_source_position(self, source, catalog_type):
        """
        Get RA and Dec from a source, handling different column names per catalog.
        
        Args:
            source: Source table row
            catalog_type: Catalog type
            
        Returns:
            tuple: (ra, dec) or (None, None) if not found
        """
        # Try standard column names first
        ra_cols = ['ra', 'RA', 'RAJ2000']
        dec_cols = ['dec', 'Dec', 'DEJ2000', 'DEC']
        
        # For each column name, check if it exists
        for ra_col in ra_cols:
            if ra_col in source.colnames:
                ra = source[ra_col]
                break
        else:
            # Catalog-specific handling
            if catalog_type == 'wise':
                ra = source.get('ra', None)
            elif catalog_type == 'akari':
                ra = source.get('ra', None)
            elif catalog_type == 'iras':
                ra = source.get('ra', None)
            else:
                ra = None
        
        for dec_col in dec_cols:
            if dec_col in source.colnames:
                dec = source[dec_col]
                break
        else:
            # Catalog-specific handling
            if catalog_type == 'wise':
                dec = source.get('dec', None)
            elif catalog_type == 'akari':
                dec = source.get('dec', None)
            elif catalog_type == 'iras':
                dec = source.get('dec', None)
            else:
                dec = None
        
        # Convert to float if possible
        try:
            ra = float(ra)
            dec = float(dec)
            return ra, dec
        except (TypeError, ValueError):
            return None, None
    
    def _get_observation_year(self, source, catalog_type):
        """
        Get observation year from a source.
        
        Args:
            source: Source table row
            catalog_type: Catalog type
            
        Returns:
            float: Observation year or None if not found
        """
        # For WISE, try to get actual observation date
        if catalog_type == 'wise':
            # Different WISE catalogs have different date columns
            date_cols = ['w1mjdmean', 'w1_mjd', 'mjd']
            
            for col in date_cols:
                if col in source.colnames:
                    try:
                        # Convert MJD to year
                        mjd = float(source[col])
                        # MJD 51544.0 = 2000-01-01
                        year = 2000.0 + (mjd - 51544.0) / 365.25
                        return year
                    except (ValueError, TypeError):
                        pass
            
            # If no date found, use the catalog's typical observation period
            return self.catalogs[catalog_type]['years'][0] + 0.5  # Middle of observation period
        
        # For AKARI, IRAS, use the fixed observation year
        elif catalog_type in ['akari', 'iras', 'herschel']:
            return self.catalogs[catalog_type]['years'][0]
        
        return None
    
    def _get_source_id(self, source, catalog_type):
        """
        Get a unique ID for the source.
        
        Args:
            source: Source table row
            catalog_type: Catalog type
            
        Returns:
            str: Source ID
        """
        # Try common ID columns
        id_cols = ['id', 'ID', 'source_id', 'designation', 'name']
        
        for col in id_cols:
            if col in source.colnames and source[col] is not None:
                return str(source[col])
        
        # Catalog-specific handling
        if catalog_type == 'wise':
            if 'designation' in source.colnames:
                return str(source['designation'])
            elif 'source_id' in source.colnames:
                return str(source['source_id'])
        elif catalog_type == 'akari':
            if 'name' in source.colnames:
                return str(source['name'])
        elif catalog_type == 'iras':
            if 'name' in source.colnames:
                return str(source['name'])
        
        # If no ID found, create one from position
        if hasattr(source, 'colnames'):
            ra, dec = self._get_source_position(source, catalog_type)
            if ra is not None and dec is not None:
                return f"{catalog_type.upper()}_{ra:.4f}_{dec:.4f}"
        
        # Last resort
        return f"{catalog_type.upper()}_{id(source)}"
    
    def _get_wavelength_info(self, source, catalog_type):
        """
        Get wavelength and flux information for a source.
        
        Args:
            source: Source table row
            catalog_type: Catalog type
            
        Returns:
            dict: Wavelength information or None if not found
        """
        # Different handling based on catalog type
        if catalog_type == 'wise':
            return self._get_wise_wavelength_info(source)
        elif catalog_type == 'akari':
            return self._get_akari_wavelength_info(source)
        elif catalog_type == 'iras':
            return self._get_iras_wavelength_info(source)
        elif catalog_type == 'herschel':
            return self._get_herschel_wavelength_info(source)
        
        # Default handling - try to find any wavelength info
        for col in source.colnames:
            if 'wave' in col.lower() or 'lambda' in col.lower():
                try:
                    return {'wavelength': float(source[col])}
                except (ValueError, TypeError):
                    pass
        
        # If no wavelength found, use catalog default
        if catalog_type in self.catalogs:
            # Use the longest wavelength as default
            wavelengths = self.catalogs[catalog_type]['wavelengths']
            if wavelengths:
                return {'wavelength': max(wavelengths)}
        
        return None
    
    def _get_wise_wavelength_info(self, source):
        """
        Get wavelength and flux information for a WISE source.
        
        Args:
            source: WISE source row
            
        Returns:
            dict: Wavelength information
        """
        # Check for W4 (22 μm) detection first, as it's most relevant for cold objects
        if 'w4mpro' in source.colnames and source['w4mpro'] is not None:
            w4mag = source['w4mpro']
            w4snr = source.get('w4snr', None)
            
            # Check if this is a good detection
            if w4snr is not None and float(w4snr) > 3:
                return {
                    'wavelength': 22.0,
                    'flux': w4mag,
                    'flux_units': 'mag',
                    'snr': float(w4snr)
                }
        
        # Next try W3 (12 μm)
        if 'w3mpro' in source.colnames and source['w3mpro'] is not None:
            w3mag = source['w3mpro']
            w3snr = source.get('w3snr', None)
            
            # Check if this is a good detection
            if w3snr is not None and float(w3snr) > 3:
                return {
                    'wavelength': 12.0,
                    'flux': w3mag,
                    'flux_units': 'mag',
                    'snr': float(w3snr)
                }
        
        # Fallback to W2 (4.6 μm)
        if 'w2mpro' in source.colnames and source['w2mpro'] is not None:
            return {
                'wavelength': 4.6,
                'flux': source['w2mpro'],
                'flux_units': 'mag',
                'snr': source.get('w2snr', None)
            }
        
        # Last resort: W1 (3.4 μm)
        if 'w1mpro' in source.colnames and source['w1mpro'] is not None:
            return {
                'wavelength': 3.4,
                'flux': source['w1mpro'],
                'flux_units': 'mag',
                'snr': source.get('w1snr', None)
            }
        
        # Default wavelength if no specific band info found
        return {'wavelength': 12.0}  # Default to W3
    
    def _get_akari_wavelength_info(self, source):
        """
        Get wavelength and flux information for an AKARI source.
        
        Args:
            source: AKARI source row
            
        Returns:
            dict: Wavelength information
        """
        # AKARI FIS bands:
        # N60 (65 μm), WIDE-S (90 μm), WIDE-L (140 μm), N160 (160 μm)
        
        # Look for longest wavelength first (N160)
        if 'flux160' in source.colnames:
            return {
                'wavelength': 160.0,
                'flux': source['flux160'],
                'flux_units': 'Jy',
                'snr': source.get('sn160', None)
            }
        
        # Next try WIDE-L (140 μm)
        if 'fluxwl' in source.colnames:
            return {
                'wavelength': 140.0,
                'flux': source['fluxwl'],
                'flux_units': 'Jy',
                'snr': source.get('snwl', None)
            }
        
        # Next WIDE-S (90 μm)
        if 'fluxws' in source.colnames:
            return {
                'wavelength': 90.0,
                'flux': source['fluxws'],
                'flux_units': 'Jy',
                'snr': source.get('snws', None)
            }
        
        # Finally N60 (65 μm)
        if 'flux65' in source.colnames:
            return {
                'wavelength': 65.0,
                'flux': source['flux65'],
                'flux_units': 'Jy',
                'snr': source.get('sn65', None)
            }
        
        # Check for IRC (mid-infrared) bands as fallback
        if 'flux18' in source.colnames:
            return {
                'wavelength': 18.0,
                'flux': source['flux18'],
                'flux_units': 'Jy'
            }
        
        # Default wavelength if no specific band info found
        return {'wavelength': 90.0}  # Default to WIDE-S
    
    def _get_iras_wavelength_info(self, source):
        """
        Get wavelength and flux information for an IRAS source.
        
        Args:
            source: IRAS source row
            
        Returns:
            dict: Wavelength information
        """
        # IRAS bands: 12, 25, 60, 100 μm
        # Try 100 μm first
        if 'fnu_100' in source.colnames or 'flux100' in source.colnames:
            flux_col = 'fnu_100' if 'fnu_100' in source.colnames else 'flux100'
            return {
                'wavelength': 100.0,
                'flux': source[flux_col],
                'flux_units': 'Jy'
            }
        
        # Next 60 μm
        if 'fnu_60' in source.colnames or 'flux60' in source.colnames:
            flux_col = 'fnu_60' if 'fnu_60' in source.colnames else 'flux60'
            return {
                'wavelength': 60.0,
                'flux': source[flux_col],
                'flux_units': 'Jy'
            }
        
        # Next 25 μm
        if 'fnu_25' in source.colnames or 'flux25' in source.colnames:
            flux_col = 'fnu_25' if 'fnu_25' in source.colnames else 'flux25'
            return {
                'wavelength': 25.0,
                'flux': source[flux_col],
                'flux_units': 'Jy'
            }
        
        # Finally 12 μm
        if 'fnu_12' in source.colnames or 'flux12' in source.colnames:
            flux_col = 'fnu_12' if 'fnu_12' in source.colnames else 'flux12'
            return {
                'wavelength': 12.0,
                'flux': source[flux_col],
                'flux_units': 'Jy'
            }
        
        # Default wavelength if no specific band info found
        return {'wavelength': 60.0}  # Default to 60 μm
    
    def _get_herschel_wavelength_info(self, source):
        """
        Get wavelength and flux information for a Herschel source.
        
        Args:
            source: Herschel source row
            
        Returns:
            dict: Wavelength information
        """
        # Herschel bands: PACS (70, 100, 160 μm), SPIRE (250, 350, 500 μm)
        # Try to infer wavelength from available columns
        wavelength = None
        
        # Check columns for wavelength indicators
        for col in source.colnames:
            if '500' in col:
                wavelength = 500.0
                break
            elif '350' in col:
                wavelength = 350.0
                break
            elif '250' in col:
                wavelength = 250.0
                break
            elif '160' in col:
                wavelength = 160.0
                break
            elif '100' in col:
                wavelength = 100.0
                break
            elif '70' in col:
                wavelength = 70.0
                break
        
        # If no wavelength found from columns, use a default
        if wavelength is None:
            wavelength = 250.0  # Middle SPIRE band
        
        return {'wavelength': wavelength}
    
    def _calculate_relevance_score(self, wavelength, separation, catalog_type):
        """
        Calculate a relevance score for a detection.
        
        Args:
            wavelength: Wavelength in microns
            separation: Separation in degrees
            catalog_type: Catalog type
            
        Returns:
            float: Relevance score (higher is better)
        """
        # Base score - wavelength component
        # Highest scores for wavelengths in the ideal range (70-100μm)
        if 70 <= wavelength <= 100:
            wavelength_score = 100
        elif 50 <= wavelength <= 200:
            wavelength_score = 80
        elif 20 <= wavelength <= 500:
            wavelength_score = 60
        elif wavelength > 500:
            wavelength_score = 40
        else:  # wavelength < 20
            wavelength_score = 20 * (wavelength / 20)
        
        # Position component - closer is better
        position_score = 100 * (1 - separation / self.search_radius)
        
        # Catalog bonus - prioritize certain catalogs
        catalog_bonus = 0
        if catalog_type in self.catalogs:
            priority = self.catalogs[catalog_type].get('priority', 'medium')
            if priority == 'high':
                catalog_bonus = 20
            elif priority == 'medium':
                catalog_bonus = 10
        
        # Final score - weighted average
        score = (wavelength_score * 0.6) + (position_score * 0.3) + catalog_bonus
        
        return score
    
    def download(self, download_dir, filter_quality=None):
        """
        Download data files from IRSA.
        
        Note: Most IRSA catalogs don't provide direct FITS file access.
        This function primarily downloads catalog data.
        
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
        irsa_dir = os.path.join(download_dir, f"irsa_{timestamp}")
        os.makedirs(irsa_dir, exist_ok=True)
        
        # For IRSA, we primarily save catalog data
        # Most actual FITS files require custom downloading per catalog
        catalog_file = os.path.join(irsa_dir, f"irsa_catalog_data.csv")
        
        # Save classification groups to separate files
        classifications = {'perfect': [], 'good': [], 'expanded': []}
        for result in results_to_download:
            cls = result.get('classification', 'expanded')
            classifications[cls].append(result)
        
        downloaded_files = []
        for cls, results in classifications.items():
            if results:
                cls_file = os.path.join(irsa_dir, f"irsa_{cls}_results.csv")
                
                # Prepare header and data
                if results:
                    # Get all column names from results
                    columns = set()
                    for r in results:
                        columns.update(k for k in r.keys() if k != 'raw_metadata')
                    
                    # Convert to sorted list for consistent order
                    columns = sorted(columns)
                    
                    # Write CSV file
                    with open(cls_file, 'w') as f:
                        # Write header
                        f.write(','.join(columns) + '\n')
                        
                        # Write data rows
                        for r in results:
                            row = [str(r.get(col, '')) for col in columns]
                            f.write(','.join(row) + '\n')
                    
                    downloaded_files.append(cls_file)
                    logger.info(f"Saved {len(results)} {cls} results to {cls_file}")
        
        # Note: For actual FITS files, would need to implement custom downloaders per catalog
        # e.g., WISE, AKARI, etc. have different data access methods
        
        logger.info(f"Downloaded {len(downloaded_files)} files to {irsa_dir}")
        return downloaded_files

def main():
    """
    Run a standalone IRSA repository search.
    """
    import argparse
    from p9_data_core import load_config, load_trajectory
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Search IRSA for Planet Nine data")
    parser.add_argument("--config", default="config/search_params.yaml", help="Configuration file")
    parser.add_argument("--trajectory", default="config/p9_trajectory.txt", help="Trajectory file")
    parser.add_argument("--tight", action="store_true", help="Use tight search parameters")
    parser.add_argument("--loose", action="store_true", help="Use expanded search parameters")
    parser.add_argument("--output", default="data/reports", help="Output directory for reports")
    parser.add_argument("--download", action="store_true", help="Download matching files")
    parser.add_argument("--download-dir", default="data/fits", help="Directory to download files to")
    parser.add_argument("--quality", choices=["perfect", "good", "all"], default="all", 
                        help="Quality filter for downloads")
    parser.add_argument("--catalogs", type=str, help="Comma-separated list of catalogs to search")
    parser.add_argument("--years", type=str, default="recent", 
                        help="Years to search: 'recent' (2015+), 'all', or comma-separated list")
    parser.add_argument("--min-year", type=float, default=2015.0, 
                        help="Minimum year to search when using 'recent'")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load trajectory
    trajectory_data = load_trajectory(args.trajectory)
    
    # Set search mode (tight is default)
    tight_search = not args.loose
    
    # Parse specific catalogs if provided
    specific_catalogs = None
    if args.catalogs:
        specific_catalogs = args.catalogs.split(",")
    
    # Parse specific years
    if args.years == "recent":
        specific_years = [year for year in trajectory_data.keys() if year >= args.min_year]
    elif args.years == "all":
        specific_years = None
    else:
        try:
            specific_years = [float(y) for y in args.years.split(",")]
        except:
            logger.error("Invalid years format. Use 'recent', 'all', or comma-separated years (e.g., 2015,2020,2025)")
            import sys
            sys.exit(1)
    
    # Initialize and run IRSA handler
    irsa_handler = IRSAHandler(config, trajectory_data, tight_search)
    irsa_handler.search(specific_years=specific_years, specific_catalogs=specific_catalogs)
    
    # Generate report
    report_file = irsa_handler.generate_report(args.output)
    print(f"Report generated: {report_file}")
    
    # Download files if requested
    if args.download:
        filter_quality = None if args.quality == "all" else args.quality
        downloaded_files = irsa_handler.download(args.download_dir, filter_quality)
        print(f"Downloaded {len(downloaded_files)} files")

if __name__ == "__main__":
    main()