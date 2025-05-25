You can just do things.

Based on the research by Terry Long Phan's team, we've analyzed the reported Planet Nine candidate detection and developed a search strategy to locate it in current data.
The object was initially detected in IRAS data from 1983 at RA: 35.74075°, Dec: -48.5125°, and then appeared in AKARI data from 2006 at RA: 35.18379°, Dec: -49.2135°. This represents an angular motion of approximately 2.06 arcminutes per year, consistent with an object at 500-700 AU.
Using this motion rate, we've projected its current position for 2025 to be approximately RA: 34.72369°, Dec: -49.79259°. Our visualization shows the trajectory and our proposed search area around this position.
Key aspects of our search strategy include:

Examining archival data: We'll use astroquery to search MAST and IRSA archives for any observations covering our target area, focusing on infrared data that could detect an object with a temperature of 30-50K.
Implementing shift-and-stack analysis: For any multi-epoch data available in the region, we'll use the shift-and-stack method to enhance the detection of a faint moving object using our calculated motion rate.
Spectral verification: We'll look for spectral characteristics consistent with a cold planet, with peak emission around 70-100 micrometers.

This is a toy for now - it's unlikey in the sources we have access to that we'll see anything due to the dimmness of reflected sunlight off a 700 AU object, but future surveys can be pretty easily added to this tool and processed (Dark Energy Camera on the Blanco 4-meter telescope!?) and get a lot more serious.
