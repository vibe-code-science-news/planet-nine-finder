#!/usr/bin/env python
"""
Planet Nine Multi-Repository FITS Finder

This script orchestrates searches across multiple astronomical data repositories
to find FITS files that might contain Planet Nine.

Usage:
  python p9_master_finder.py [--tight/--loose] [--repos REPOS] [--years YEARS]

Options:
  --tight             Use tight search parameters [default]
  --loose             Use expanded search parameters
  --repos REPOS       Comma-separated list of repositories to search [default: nasa_wise]
  --years YEARS       Comma-separated list of years to search [default: all]
  --download          Download matching FITS files
  --quality QUALITY   Quality filter for downloads: perfect, good, all [default: all]
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Import core functionality
from p9_data_core import load_config, load_trajectory

# Import repository handlers
# Commented out handlers that had API issues
# from repository_modules.p9_mast_finder import MASTHandler
# from repository_modules.p9_irsa_finder import IRSAHandler
# from repository_modules.p9_herschel_finder import HerschelHandler
from repository_modules.p9_nasa_wise_finder import NASAWISEHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to orchestrate search across repositories.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Search for Planet Nine FITS files")
    parser.add_argument("--config", default="config/search_params.yaml", help="Configuration file")
    parser.add_argument("--trajectory", default="config/p9_trajectory.txt", help="Trajectory file")
    parser.add_argument("--tight", action="store_true", help="Use tight search parameters")
    parser.add_argument("--loose", action="store_true", help="Use expanded search parameters")
    parser.add_argument("--repos", type=str, default="nasa_wise", help="Repositories to search (comma-separated, default: nasa_wise)")
    parser.add_argument("--years", type=str, default="recent", help="Years to search: 'recent' (2010+), 'all', or comma-separated list")
    parser.add_argument("--min-year", type=float, default=2010.0, help="Minimum year to search when using 'recent' (default: 2010.0 for WISE)")
    parser.add_argument("--output", default="data/reports", help="Output directory for reports")
    parser.add_argument("--download", action="store_true", help="Download matching files")
    parser.add_argument("--download-dir", default="data/fits", help="Directory to download files to")
    parser.add_argument("--quality", choices=["perfect", "good", "all"], default="all", 
                        help="Quality filter for downloads")
    parser.add_argument("--bands", type=str, default="W3,W4", 
                        help="Comma-separated list of WISE bands to search (for NASA WISE)")
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/reports", exist_ok=True)
    os.makedirs("data/fits", exist_ok=True)
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Load trajectory
        trajectory_data = load_trajectory(args.trajectory)
        
        # Set search mode (tight is default)
        tight_search = not args.loose
        
        # Determine repositories to search
        if args.repos == "all":
            # Only include NASA WISE by default since others had API issues
            repositories = ["nasa_wise"]
        else:
            repositories = args.repos.split(",")
        
        # Parse specific years if provided
        if args.years.lower() == "all":
            years = None
            logger.info("Searching all years in trajectory data")
        elif args.years.lower() == "recent":
            # Filter for recent years only
            min_year = args.min_year
            years = [year for year in trajectory_data.keys() if year >= min_year]
            if years:
                years.sort()
                logger.info(f"Searching recent years ({min_year}+): {years}")
            else:
                logger.warning(f"No years found >= {min_year} in trajectory data")
                years = None
        else:
            try:
                years = [float(y) for y in args.years.split(",")]
                logger.info(f"Searching specific years: {years}")
            except:
                logger.error("Invalid years format. Use 'recent', 'all', or comma-separated years (e.g., 2010,2011,2013)")
                sys.exit(1)
        
        # Parse WISE bands if provided
        if args.bands:
            bands = args.bands.split(",")
        else:
            bands = ["W3", "W4"]  # Default bands for WISE
        
        # Overall results mapping
        all_results = {}
        all_classifications = {}
        
        # Process each repository
        for repo in repositories:
            if repo.lower() == "nasa_wise":
                handler = NASAWISEHandler(config, trajectory_data, tight_search)
                logger.info(f"Initialized {handler.name} handler")
            # Commented out handlers that had API issues
            # elif repo.lower() == "mast":
            #     handler = MASTHandler(config, trajectory_data, tight_search)
            #     logger.info(f"Initialized {handler.name} handler")
            # elif repo.lower() == "irsa":
            #     handler = IRSAHandler(config, trajectory_data, tight_search)
            #     logger.info(f"Initialized {handler.name} handler")
            # elif repo.lower() == "herschel":
            #     handler = HerschelHandler(config, trajectory_data, tight_search)
            #     logger.info(f"Initialized {handler.name} handler")
            else:
                logger.warning(f"Repository {repo} not supported yet")
                continue
            
            # Perform search
            logger.info(f"Searching {handler.name}...")
            if repo.lower() == "nasa_wise":
                # Pass bands specifically for NASA WISE
                handler.search(specific_years=years, specific_bands=bands)
            else:
                handler.search(specific_years=years)
            
            # Classify results
            classifications = handler.classify_results()
            
            # Generate report
            report_file = handler.generate_report(args.output)
            logger.info(f"Generated {handler.name} report: {report_file}")
            
            # Store results
            all_results[repo] = handler.results
            all_classifications[repo] = classifications
            
            # Download files if requested
            if args.download:
                filter_quality = None if args.quality == "all" else args.quality
                logger.info(f"Downloading {handler.name} files (quality={args.quality})...")
                downloaded_files = handler.download(args.download_dir, filter_quality)
                logger.info(f"Downloaded {len(downloaded_files)} files from {handler.name}")
        
        # Generate master summary report
        generate_master_report(all_results, all_classifications, args.output)
    
    except Exception as e:
        logger.error(f"Error in main process: {e}", exc_info=True)
        sys.exit(1)

def generate_master_report(all_results, all_classifications, output_dir):
    """
    Generate a master summary report of all repository results.
    
    Args:
        all_results (dict): Dictionary of results by repository
        all_classifications (dict): Dictionary of classification counts by repository
        output_dir (str): Directory to write the report to
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate report file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = os.path.join(output_dir, f"master_summary_{timestamp}.md")
    
    total_perfect = sum(c.get('perfect', 0) for c in all_classifications.values())
    total_good = sum(c.get('good', 0) for c in all_classifications.values())
    total_expanded = sum(c.get('expanded', 0) for c in all_classifications.values())
    total_results = sum(len(results) for results in all_results.values())
    
    # Open file with explicit UTF-8 encoding to handle special characters
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Planet Nine Search Master Summary\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overall Results\n\n")
        f.write(f"- Total results across all repositories: {total_results}\n")
        f.write(f"- Perfect matches: {total_perfect}\n")
        f.write(f"- Good matches: {total_good}\n")
        f.write(f"- Expanded matches: {total_expanded}\n\n")
        
        f.write("## Results by Repository\n\n")
        f.write("| Repository | Total | Perfect | Good | Expanded |\n")
        f.write("|------------|-------|---------|------|----------|\n")
        
        for repo, classifications in all_classifications.items():
            total = len(all_results[repo])
            perfect = classifications.get('perfect', 0)
            good = classifications.get('good', 0)
            expanded = classifications.get('expanded', 0)
            
            f.write(f"| {repo.upper()} | {total} | {perfect} | {good} | {expanded} |\n")
        
        f.write("\n## Perfect Matches\n\n")
        
        if total_perfect > 0:
            f.write("| Repository | ID | Instrument | Wavelength | Year | RA | Dec | Separation |\n")
            f.write("|------------|---|------------|------------|------|----|----|------------|\n")
            
            for repo, results in all_results.items():
                for result in results:
                    if result.get('classification') == 'perfect':
                        f.write(f"| {repo.upper()} | " +
                                f"{result.get('id', 'N/A')} | " +
                                f"{result.get('instrument', 'N/A')} | " +
                                f"{result.get('wavelength', 'N/A')} | " +
                                f"{result.get('obs_year', 'N/A')} | " +
                                f"{result.get('ra', 'N/A'):.5f} | " +
                                f"{result.get('dec', 'N/A'):.5f} | " +
                                f"{result.get('separation_deg', 'N/A'):.5f} |\n")
        else:
            f.write("No perfect matches found.\n")
        
        f.write("\n## Conclusions and Recommendations\n\n")
        
        if total_perfect > 0:
            f.write("**Perfect matches found!** The search has identified ideal candidates for Planet Nine detection. "
                    "These should be prioritized for further analysis.\n\n")
        elif total_good > 0:
            f.write("**Good matches found.** While no perfect matches were found, several good candidates "
                    "were identified that could potentially show Planet Nine. These should be examined further.\n\n")
        else:
            f.write("**Limited promising matches.** The search found few high-quality matches. Consider "
                    "expanding the search parameters or exploring additional repositories.\n\n")
        
        # Recommend next steps
        f.write("### Recommended Next Steps\n\n")
        
        if total_perfect > 0 or total_good > 0:
            f.write("1. Download and analyze the identified FITS files\n")
            f.write("2. Apply the shift-and-stack algorithm to detect potential motion\n")
            f.write("3. Compare results across different wavelengths\n")
        else:
            f.write("1. Expand search to additional repositories\n")
            f.write("2. Consider using a looser search radius to find more potential matches\n")
            f.write("3. For WISE data, prioritize W4 (22 micron) band which may be optimal for detecting cold objects\n")
    
    logger.info(f"Generated master summary report: {report_file}")
    return report_file

if __name__ == "__main__":
    main()