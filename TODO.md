TODO

We calculated where it ought to be today, need to account for where it would be at the time of every fits file

Obvs our sources are too limited, need to make fits_finder robust... 

More research on available sources

Put them together and we need to check a line in time

Make processing and results more robust


Eh, maybe not as big a deal as I thought:

Motion Rate Calculation
Based on the data in your code:

RA rate: -0.024216 degrees/year
Dec rate: -0.030478 degrees/year
Angular velocity: 2.06 arcminutes/year

Converting to a total angular motion:

Total angular motion rate = √(RA rate² + Dec rate²)
Total angular motion rate = √((-0.024216)² + (-0.030478)²)
Total angular motion rate ≈ 0.03895 degrees/year

Transit Time Calculation
For a field of view with 0.2-degree radius:

Diameter of field = 0.4 degrees
Time to cross = 0.4 degrees ÷ 0.03895 degrees/year
Time to cross ≈ 10.27 years

This means Planet Nine would take about 10-11 years to completely traverse one of your search fields.

https://irsa.ipac.caltech.edu/ibe/search/wise/allwise/p3am_cdd?POS=52.97,-50.03&SIZE=0.05&RESPONSEFORMAT=ipac