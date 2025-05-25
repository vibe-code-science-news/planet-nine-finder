#!/usr/bin/env python
"""
IRSA Catalog List Utility

This script lists all available catalogs in IRSA and saves them to a file.
This can help identify the correct catalog names for IRAS, AKARI, and WISE data.

Usage:
  python irsa_catalogs.py

"""

import os
from astroquery.irsa import Irsa

print("Fetching list of available IRSA catalogs...")
try:
    catalogs = Irsa.list_catalogs()
    print(f"Found {len(catalogs)} catalogs")
    
    # Filter for catalogs we're interested in
    iras_catalogs = [cat for cat in catalogs if 'iras' in cat.lower()]
    akari_catalogs = [cat for cat in catalogs if 'akari' in cat.lower()]
    wise_catalogs = [cat for cat in catalogs if 'wise' in cat.lower()]
    
    print("\nIRAS Catalogs:")
    for cat in iras_catalogs:
        print(f"  {cat}")
    
    print("\nAKARI Catalogs:")
    for cat in akari_catalogs:
        print(f"  {cat}")
    
    print("\nWISE Catalogs:")
    for cat in wise_catalogs:
        print(f"  {cat}")
    
    # Save all catalogs to a file
    with open('irsa_catalogs.txt', 'w') as f:
        f.write("# IRSA Catalogs\n\n")
        f.write("## All Catalogs\n")
        for cat in catalogs:
            f.write(f"{cat}\n")
        
        f.write("\n## IRAS Catalogs\n")
        for cat in iras_catalogs:
            f.write(f"{cat}\n")
        
        f.write("\n## AKARI Catalogs\n")
        for cat in akari_catalogs:
            f.write(f"{cat}\n")
        
        f.write("\n## WISE Catalogs\n")
        for cat in wise_catalogs:
            f.write(f"{cat}\n")
    
    print(f"\nCatalog list saved to irsa_catalogs.txt")
    
except Exception as e:
    print(f"Error fetching catalog list: {e}")