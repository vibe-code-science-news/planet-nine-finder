#!/usr/bin/env python
"""
Direct WISE Data Downloader for Planet Nine

This script takes a more direct approach to getting WISE data by using the IRSA 
Finder Chart service and direct cutout requests, bypassing the need to know the 
exact archive structure.
"""

import os
import sys
import requests
import json
import logging
import time
from urllib.parse import quote
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WISEDownloader:
    """Class for directly downloading WISE data from IRSA using multiple methods"""
    
    def __init__(self, output_dir="data/fits/wise_direct"):
        """
        Initialize the WISE downloader.
        
        Args:
            output_dir (str): Directory to download files to
        """
        self.output_dir = output_dir
        
        # IRSA service URLs
        self.finder_chart_url = "https://irsa.ipac.caltech.edu/irsaviewer/IrsaViewer"
        self.cutout_service_url = "https://irsa.ipac.caltech.edu/applications/wise/IM/DiskSearch.html"
        self.atlas_search_url = "https://irsa.ipac.caltech.edu/cgi-bin/Atlas/nph-atlas"
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Track successful and failed downloads
        self.successful_downloads = []
        self.failed_downloads = []
    
    def download_region(self, ra, dec, radius=0.2, bands=None):
        """
        Download WISE data for a region using multiple methods.
        
        Args:
            ra (float): RA in degrees
            dec (float): Dec in degrees
            radius (float): Search radius in degrees (max 0.5)
            bands (list): WISE bands to try ('1', '2', '3', '4') or None for all
            
        Returns:
            list: Paths to downloaded files
        """
        if bands is None:
            bands = ['1', '2', '3', '4']
        
        logger.info(f"Attempting to download WISE data for position RA={ra}, Dec={dec}, radius={radius}°")
        
        # List of methods to try (in order of preference)
        methods = [
            self._download_via_atlas_search,
            self._download_via_cutout_service,
            self._download_via_finder_chart
        ]
        
        # Try each method until one succeeds
        for method in methods:
            logger.info(f"Trying method: {method.__name__}")
            try:
                results = method(ra, dec, radius, bands)
                if results:
                    logger.info(f"Successfully downloaded {len(results)} files using {method.__name__}")
                    self.successful_downloads.extend(results)
                    return results
            except Exception as e:
                logger.error(f"Error with {method.__name__}: {e}")
        
        # If we reach here, all methods failed
        logger.warning(f"All download methods failed for RA={ra}, Dec={dec}")
        return []
    
    def _download_via_atlas_search(self, ra, dec, radius, bands):
        """
        Download WISE data using the IRSA Atlas search service.
        
        This is often the most direct and reliable method.
        
        Args:
            ra (float): RA in degrees
            dec (float): Dec in degrees
            radius (float): Search radius in degrees
            bands (list): WISE bands to try
            
        Returns:
            list: Paths to downloaded files
        """
        downloaded_files = []
        
        # Convert radius to arcminutes for the Atlas service
        radius_arcmin = radius * 60.0
        
        # Map band numbers to WISE dataset names
        band_datasets = {
            '1': 'wise.allwise_p3am_cdd',  # W1
            '2': 'wise.allwise_p3am_cdd',  # W2
            '3': 'wise.allwise_p3am_cdd',  # W3
            '4': 'wise.allwise_p3am_cdd'   # W4
        }
        
        # Try each band
        for band in bands:
            if band not in band_datasets:
                logger.warning(f"Invalid band: {band}, skipping")
                continue
            
            dataset = band_datasets[band]
            logger.info(f"Searching Atlas for band W{band} using dataset {dataset}")
            
            # Construct Atlas search parameters
            params = {
                'mode': 'getImage',
                'dataset': dataset,
                'band': int(band),
                'locstr': f"{ra},{dec}",
                'objstr': f"P9_search_b{band}",
                'size': f"{radius_arcmin}arcmin"
            }
            
            try:
                # Send Atlas search request
                response = requests.get(self.atlas_search_url, params=params)
                
                if response.status_code == 200 and response.content:
                    # Check if the response is a FITS file
                    if response.headers.get('content-type') == 'application/fits' or \
                       response.content[:8] == b'SIMPLE  ':
                        
                        # Save the FITS file
                        outfile = os.path.join(self.output_dir, f"wise_w{band}_ra{ra:.3f}_dec{dec:.3f}.fits")
                        with open(outfile, 'wb') as f:
                            f.write(response.content)
                        
                        logger.info(f"Downloaded W{band} image to {outfile}")
                        downloaded_files.append(outfile)
                    else:
                        # Check if we got an error message or redirect
                        if 'No data found' in response.text:
                            logger.warning(f"No data found for W{band} at RA={ra}, Dec={dec}")
                        else:
                            logger.warning(f"Unexpected response for W{band}: not a FITS file")
                else:
                    logger.warning(f"Failed to download W{band} data: HTTP {response.status_code}")
            
            except Exception as e:
                logger.error(f"Error downloading W{band} via Atlas: {e}")
        
        return downloaded_files
    
    def _download_via_cutout_service(self, ra, dec, radius, bands):
        """
        Download WISE data using the IRSA cutout service.
        
        Args:
            ra (float): RA in degrees
            dec (float): Dec in degrees
            radius (float): Search radius in degrees
            bands (list): WISE bands to try
            
        Returns:
            list: Paths to downloaded files
        """
        downloaded_files = []
        
        # Convert radius to arcseconds for the cutout service (max 300)
        radius_arcsec = min(radius * 3600.0, 300)
        
        # Different survey options to try
        surveys = ['allwise', 'allsky']
        
        # Try each band
        for band in bands:
            # Try each survey
            for survey in surveys:
                logger.info(f"Trying cutout for W{band} using {survey} survey")
                
                # Construct cutout URL (using direct format known to work)
                cutout_url = (f"https://irsa.ipac.caltech.edu/cgi-bin/ICORE/nph-icore?"
                             f"locstr={ra}+{dec}&survey={survey}&band={band}"
                             f"&sizeX={radius_arcsec}&sizeY={radius_arcsec}"
                             f"&subsetsize=&mode=PI")
                
                try:
                    # Send cutout request
                    response = requests.get(cutout_url)
                    
                    if response.status_code == 200 and response.content:
                        # Check if the response is a FITS file
                        if response.headers.get('content-type') == 'application/fits' or \
                           response.content[:8] == b'SIMPLE  ':
                            
                            # Save the FITS file
                            outfile = os.path.join(self.output_dir, f"wise_{survey}_w{band}_ra{ra:.3f}_dec{dec:.3f}.fits")
                            with open(outfile, 'wb') as f:
                                f.write(response.content)
                            
                            logger.info(f"Downloaded W{band} cutout from {survey} to {outfile}")
                            downloaded_files.append(outfile)
                            
                            # Success with this band, move to next
                            break
                    else:
                        logger.warning(f"Failed to download W{band} cutout from {survey}: HTTP {response.status_code}")
                
                except Exception as e:
                    logger.error(f"Error downloading W{band} cutout from {survey}: {e}")
        
        return downloaded_files
    
    def _download_via_finder_chart(self, ra, dec, radius, bands):
        """
        Download WISE data using the IRSA Finder Chart service.
        
        Args:
            ra (float): RA in degrees
            dec (float): Dec in degrees
            radius (float): Search radius in degrees
            bands (list): WISE bands to try
            
        Returns:
            list: Paths to downloaded files
        """
        downloaded_files = []
        
        # Convert radius to arcminutes for the Finder Chart service
        radius_arcmin = radius * 60.0
        
        # Construct Finder Chart request
        params = {
            'subsetsize': radius_arcmin,
            'imagetype': 'atlas',
            'refcat': 'none',
            'objstr': f"{ra} {dec}",
            'survey': 'allwise',
            'band': ','.join([f'w{b}' for b in bands]),
            'filetypes': 'fits',
            'additionalinfo': 'true',
            'download': 'true'
        }
        
        try:
            # Send Finder Chart request
            response = requests.post(self.finder_chart_url, data=params)
            
            if response.status_code == 200:
                # Parse response to find download URLs
                if 'Content-Disposition' in response.headers:
                    # Direct download response
                    filename = response.headers.get('Content-Disposition', '').split('filename=')[1].strip('"')
                    outfile = os.path.join(self.output_dir, filename)
                    with open(outfile, 'wb') as f:
                        f.write(response.content)
                    
                    logger.info(f"Downloaded finder chart result to {outfile}")
                    downloaded_files.append(outfile)
                else:
                    # Response with download links
                    for line in response.text.split('\n'):
                        if 'fits' in line.lower() and 'href' in line.lower():
                            # Extract download URL
                            url_start = line.find('href="') + 6
                            url_end = line.find('"', url_start)
                            if url_start > 6 and url_end > url_start:
                                download_url = line[url_start:url_end]
                                
                                # Download the FITS file
                                try:
                                    if not download_url.startswith('http'):
                                        if download_url.startswith('/'):
                                            download_url = f"https://irsa.ipac.caltech.edu{download_url}"
                                        else:
                                            download_url = f"https://irsa.ipac.caltech.edu/{download_url}"
                                    
                                    file_response = requests.get(download_url)
                                    
                                    if file_response.status_code == 200:
                                        # Extract filename from URL or Content-Disposition
                                        if 'Content-Disposition' in file_response.headers:
                                            filename = file_response.headers.get('Content-Disposition').split('filename=')[1].strip('"')
                                        else:
                                            filename = download_url.split('/')[-1]
                                        
                                        outfile = os.path.join(self.output_dir, filename)
                                        with open(outfile, 'wb') as f:
                                            f.write(file_response.content)
                                        
                                        logger.info(f"Downloaded FITS file to {outfile}")
                                        downloaded_files.append(outfile)
                                except Exception as e:
                                    logger.error(f"Error downloading from {download_url}: {e}")
            else:
                logger.warning(f"Failed to get finder chart: HTTP {response.status_code}")
        
        except Exception as e:
            logger.error(f"Error with finder chart download: {e}")
        
        return downloaded_files
    
    def download_from_results_file(self, results_file, max_attempts=10, band_filter=None):
        """
        Download WISE data for candidates in a results JSON file.
        
        Args:
            results_file (str): Path to results JSON file
            max_attempts (int): Maximum number of candidates to try
            band_filter (list): Optional list of bands to filter for (e.g., ['W3', 'W4'])
            
        Returns:
            tuple: (successful_downloads, failed_attempts)
        """
        # Load results from file
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            logger.info(f"Loaded {len(results)} results from {results_file}")
        except Exception as e:
            logger.error(f"Error loading results from {results_file}: {e}")
            return [], []
        
        # Filter results by band if specified
        if band_filter and isinstance(results, list):
            filtered_results = []
            for result in results:
                wavelength = result.get('wavelength')
                catalog_type = result.get('catalog_type')
                
                # Check if it's a WISE result with the right band
                if catalog_type == 'wise' and wavelength is not None:
                    if wavelength == 22.0 and 'W4' in band_filter:
                        filtered_results.append(result)
                    elif wavelength == 12.0 and 'W3' in band_filter:
                        filtered_results.append(result)
                    elif wavelength == 4.6 and 'W2' in band_filter:
                        filtered_results.append(result)
                    elif wavelength == 3.4 and 'W1' in band_filter:
                        filtered_results.append(result)
                
                # If not filtering by band, include all
                elif not band_filter:
                    filtered_results.append(result)
            
            results = filtered_results
            logger.info(f"Filtered to {len(results)} results matching bands: {band_filter}")
        
        # Sort results by relevance score if available
        if isinstance(results, list) and len(results) > 0 and 'relevance_score' in results[0]:
            results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            logger.info(f"Sorted results by relevance score")
        
        # Limit to max_attempts
        if len(results) > max_attempts:
            results = results[:max_attempts]
            logger.info(f"Limited to {max_attempts} most relevant results")
        
        # Process each result
        successful_attempts = 0
        failed_attempts = 0
        
        for i, result in enumerate(results):
            logger.info(f"Processing result {i+1}/{len(results)}")
            
            try:
                # Extract coordinates
                ra = result.get('ra')
                dec = result.get('dec')
                
                if ra is None or dec is None:
                    logger.warning("Missing RA/Dec information, skipping")
                    failed_attempts += 1
                    continue
                
                # Determine which band to download
                wavelength = result.get('wavelength')
                
                if wavelength == 22.0:
                    bands = ['4']  # W4
                elif wavelength == 12.0:
                    bands = ['3']  # W3
                elif wavelength == 4.6:
                    bands = ['2']  # W2
                elif wavelength == 3.4:
                    bands = ['1']  # W1
                else:
                    # Default to W3 and W4 as they're most relevant for Planet Nine
                    bands = ['3', '4']
                
                # Create a subdirectory for this result
                result_dir = os.path.join(self.output_dir, f"result_{i+1}")
                os.makedirs(result_dir, exist_ok=True)
                
                # Save metadata
                metadata_file = os.path.join(result_dir, "metadata.json")
                with open(metadata_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                # Attempt to download
                downloaded = self.download_region(ra, dec, radius=0.1, bands=bands)
                
                if downloaded:
                    logger.info(f"Downloaded {len(downloaded)} files for result {i+1}")
                    successful_attempts += 1
                    
                    # Move downloaded files to the result directory
                    for src_file in downloaded:
                        filename = os.path.basename(src_file)
                        dst_file = os.path.join(result_dir, filename)
                        os.rename(src_file, dst_file)
                else:
                    logger.warning(f"No files downloaded for result {i+1}")
                    failed_attempts += 1
            
            except Exception as e:
                logger.error(f"Error processing result {i+1}: {e}")
                failed_attempts += 1
        
        logger.info(f"Download summary: {successful_attempts} successful, {failed_attempts} failed")
        return successful_attempts, failed_attempts

def main():
    """
    Command line interface for the WISE Downloader.
    """
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Download WISE data directly from IRSA")
    parser.add_argument("--results", type=str, help="Path to results JSON file")
    parser.add_argument("--output-dir", type=str, default="data/fits/wise_direct", help="Output directory")
    parser.add_argument("--max-attempts", type=int, default=10, help="Maximum number of download attempts")
    parser.add_argument("--band-filter", type=str, help="Comma-separated list of bands to download (e.g., W3,W4)")
    parser.add_argument("--region", action="store_true", help="Download a specific region")
    parser.add_argument("--ra", type=float, help="RA for region download")
    parser.add_argument("--dec", type=float, help="Dec for region download")
    parser.add_argument("--radius", type=float, default=0.1, help="Radius for region download (degrees)")
    args = parser.parse_args()
    
    # Create downloader
    downloader = WISEDownloader(output_dir=args.output_dir)
    
    # Parse band filter
    band_filter = None
    if args.band_filter:
        band_filter = args.band_filter.split(',')
    
    # Download based on results file
    if args.results:
        successful, failed = downloader.download_from_results_file(
            args.results, 
            max_attempts=args.max_attempts,
            band_filter=band_filter
        )
        
        print(f"Download summary: {successful} successful, {failed} failed")
    
    # Download specific region
    elif args.region and args.ra is not None and args.dec is not None:
        # Convert band filter to WISE bands
        wise_bands = []
        if band_filter:
            for band in band_filter:
                if band.upper() == 'W1':
                    wise_bands.append('1')
                elif band.upper() == 'W2':
                    wise_bands.append('2')
                elif band.upper() == 'W3':
                    wise_bands.append('3')
                elif band.upper() == 'W4':
                    wise_bands.append('4')
        
        if not wise_bands:
            wise_bands = ['3', '4']  # Default to W3 and W4
        
        downloaded = downloader.download_region(
            args.ra, 
            args.dec, 
            radius=args.radius,
            bands=wise_bands
        )
        
        print(f"Downloaded {len(downloaded)} files")
    
    else:
        print("Error: Must specify either --results or --region with --ra and --dec")
        print("Example: python p9_irsa_direct_download.py --region --ra 34.7237 --dec -49.7926 --band-filter W3,W4")
        print("Example: python p9_irsa_direct_download.py --results data/reports/irsa_results.json --band-filter W3,W4")

if __name__ == "__main__":
    main()