# Future Data Acquisition Plan

## Current Data Files
- `data/raw/housing_raw.csv`: Original housing dataset
- `data/processed/housing_cleaned.csv`: Processed housing dataset
- `data/raw/schools_raw.csv`: School information
- `data/processed/schools_metrics.csv`: School performance metrics
- `data/processed/neighborhood_schools.csv`: Neighborhood to school mapping
- `data/processed/schools_combined.csv`: Combined school data

## 1. Crime Data Acquisition

### Step 1: Identify Data Sources
- **Public Sources:** Use websites like NeighborhoodScout and AreaVibes to gather crime statistics for prominent neighborhoods in Ames.
- **Official Sources:** Contact the Ames Police Department or check the City of Ames open data portal for more granular, official crime data.

### Step 2: Data Extraction and Mapping
- **Scrape or download** crime data from public sources.
- **Map the data** to the neighborhood names in your dataset. This may require manual mapping if the boundaries do not align perfectly.

### Step 3: Data Validation and Integration
- **Validate the data** for accuracy and completeness.
- **Integrate the crime data** into your dataset, ensuring it aligns with the neighborhood information.

## 2. Geolocation and Address Information

### Step 1: Obtain Property Addresses
- **Source:** Identify a reliable source for property addresses in the Ames dataset. This could be a real estate database, city records, or a similar resource.

### Step 2: Geocode Addresses
- **Use a geocoding service** (e.g., Google Maps API, OpenStreetMap) to convert addresses into latitude and longitude coordinates.

### Step 3: Match Properties to Schools
- **Geocode school addresses** to obtain their coordinates.
- **Calculate distances** between properties and schools to determine the closest school for each property.

### Step 4: Data Integration
- **Integrate the geolocation data** into your dataset, allowing for more precise analysis of school proximity and its impact on property values.

## 3. School Metrics

### Step 1: Identify School Metrics
- **Research available school metrics** (e.g., test scores, graduation rates, student-teacher ratios) from sources like the National Center for Education Statistics (NCES) or state education departments.

### Step 2: Data Collection
- **Collect the relevant metrics** for each school in your dataset.

### Step 3: Data Integration
- **Integrate the school metrics** into your dataset, enhancing the analysis of how school quality may influence property values.

## School Metrics Data Scraping: Lessons Learned & Future Strategy

### Why Automated Scraping Was Not Successful
- Attempts to scrape school metrics from GreatSchools using Python scripts (requests + BeautifulSoup) were unsuccessful.
- The GreatSchools website loads search results and school data dynamically using JavaScript, which is not accessible to basic HTTP requests.
- The site may also block automated requests or require a real browser environment, returning only generic or incomplete HTML to scripts.
- As a result, no school pages or metrics could be reliably accessed or parsed programmatically.

### Recommended Future Strategy
- **Manual Collection:** For a small number of schools, manual lookup and entry of metrics is the most reliable and efficient approach.
- **Browser Automation:** For larger-scale or repeatable scraping, use browser automation tools like Selenium or Playwright, which can render JavaScript and mimic real user interactions. This requires more setup (browser drivers, handling CAPTCHAs, etc.) and should respect the site's terms of service.
- **Official Data Sources:** Always check for official data downloads or APIs from state education departments or the NCES, which may provide bulk school metrics in a more accessible and ethical manner.
- **Data Partnerships:** For production or commercial use, consider reaching out to data providers for licensed access or partnerships.

**Summary:** For this project, manual entry was chosen due to the small number of schools. For future projects, browser automation or official data sources are recommended for scalable, reliable data acquisition.

## Conclusion
This plan outlines the steps for acquiring crime data, geolocation information, and school metrics to enhance your dataset. Each step requires careful consideration of data sources, validation, and integration to ensure the accuracy and usefulness of the data for your analysis. 