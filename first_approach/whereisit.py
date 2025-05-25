// Let's create a function to estimate the current position of Planet Nine
// based on the two known positions and the elapsed time

function calculateCurrentPosition(ra1, dec1, ra2, dec2, years1to2, yearsSince2) {
  // Calculate the rate of change per year
  const raRatePerYear = (ra2 - ra1) / years1to2;
  const decRatePerYear = (dec2 - dec1) / years1to2;
  
  // Calculate the estimated current position
  const currentRA = ra2 + (raRatePerYear * yearsSince2);
  const currentDec = dec2 + (decRatePerYear * yearsSince2);
  
  console.log("Position Rate of Change:");
  console.log(`RA: ${raRatePerYear.toFixed(6)} degrees/year`);
  console.log(`Dec: ${decRatePerYear.toFixed(6)} degrees/year`);
  console.log("\nEstimated Current Position:");
  console.log(`RA: ${currentRA.toFixed(5)} degrees`);
  console.log(`Dec: ${currentDec.toFixed(5)} degrees`);
  
  // Calculate angular separation between the points
  const angularSeparation1to2 = calculateAngularSeparation(ra1, dec1, ra2, dec2);
  const timeBetweenObservations = years1to2;
  const angularVelocity = angularSeparation1to2 / timeBetweenObservations;
  
  console.log(`\nAngular separation between 1983 and 2006 positions: ${angularSeparation1to2.toFixed(4)} arcminutes`);
  console.log(`Average angular velocity: ${angularVelocity.toFixed(4)} arcminutes/year`);
  
  // Calculate approximate position in 2025
  const totalYears = years1to2 + yearsSince2;
  const totalAngularMovement = angularVelocity * totalYears;
  
  console.log(`\nTotal predicted angular movement from 1983 to 2025: ${totalAngularMovement.toFixed(4)} arcminutes`);
  
  return {
    ra: currentRA,
    dec: currentDec,
    raRate: raRatePerYear,
    decRate: decRatePerYear,
    angularVelocity: angularVelocity
  };
}

// Function to calculate angular separation between two celestial coordinates (in arcminutes)
function calculateAngularSeparation(ra1, dec1, ra2, dec2) {
  // Convert to radians
  const ra1Rad = ra1 * Math.PI / 180;
  const dec1Rad = dec1 * Math.PI / 180;
  const ra2Rad = ra2 * Math.PI / 180;
  const dec2Rad = dec2 * Math.PI / 180;
  
  // Haversine formula
  const dRa = ra2Rad - ra1Rad;
  const dDec = dec2Rad - dec1Rad;
  
  const a = Math.sin(dDec/2) * Math.sin(dDec/2) +
           Math.cos(dec1Rad) * Math.cos(dec2Rad) *
           Math.sin(dRa/2) * Math.sin(dRa/2);
  
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
  const distRad = c;
  
  // Convert to arcminutes 
  const distArcmin = distRad * 180 / Math.PI * 60;
  
  return distArcmin;
}

// Known positions
const ra1983 = 35.74075;  // RA for IRAS detection in 1983
const dec1983 = -48.5125; // Dec for IRAS detection in 1983
const ra2006 = 35.18379;  // RA for AKARI detection in 2006
const dec2006 = -49.2135; // Dec for AKARI detection in 2006

// Time intervals
const years1983to2006 = 23; // Years between IRAS and AKARI
const years2006to2025 = 19; // Years between AKARI and now (2025)

// Calculate the current estimated position
const currentPosition = calculateCurrentPosition(
  ra1983, dec1983, 
  ra2006, dec2006, 
  years1983to2006, 
  years2006to2025
);

// Define search region based on positional uncertainty
// Assuming uncertainty grows roughly linearly with time
const baseUncertainty = 3; // arcminutes at time of AKARI
const timeRatio = years2006to2025 / years1983to2006;
const currentUncertainty = baseUncertainty * (1 + timeRatio);

console.log(`\nSearch radius needed due to positional uncertainty: ${currentUncertainty.toFixed(2)} arcminutes`);

// Generate a search strategy - square search area centered on predicted position
const searchRadiusDeg = currentUncertainty / 60; // convert arcmin to degrees
const squareAreaSqDeg = (searchRadiusDeg * 2) * (searchRadiusDeg * 2);

console.log("\nSearch Strategy:");
console.log(`Center search at RA: ${currentPosition.ra.toFixed(5)}, Dec: ${currentPosition.dec.toFixed(5)}`);
console.log(`Search radius: ${searchRadiusDeg.toFixed(4)} degrees`);
console.log(`Search area: approximately ${squareAreaSqDeg.toFixed(2)} square degrees`);
console.log(`This covers a region of about ${(searchRadiusDeg*2).toFixed(2)} × ${(searchRadiusDeg*2).toFixed(2)} degrees`);

// Now let's estimate the expected infrared characteristics
// Approximating as a blackbody with temperature around 30-40K
// Calculate expected wavelength of peak emission using Wien's displacement law
function calculatePeakWavelength(tempK) {
  const wienConstant = 2897.8; // µm·K
  return wienConstant / tempK; // in micrometers
}

// Consider a range of plausible temperatures
const temps = [30, 35, 40, 45, 50];
console.log("\nEstimated Peak Emission Wavelengths:");
temps.forEach(temp => {
  const peakWavelength = calculatePeakWavelength(temp);
  console.log(`At ${temp}K: ${peakWavelength.toFixed(1)} micrometers`);
});

// Suggestion for follow-up observations
console.log("\nRecommended Observation Strategy:");
console.log("1. Use wide-field infrared instruments to cover the search area");
console.log("2. Target wavelengths between 70-100 micrometers for optimal detection");
console.log("3. Prioritize observations in multiple far-infrared bands to confirm spectral characteristics");
