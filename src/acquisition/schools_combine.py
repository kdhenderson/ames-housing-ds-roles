import csv

# File paths
MAPPING_CSV = '../data/processed/neighborhood_schools.csv'
SCHOOLS_CSV = '../data/raw/schools_raw.csv'
METRICS_CSV = '../data/processed/schools_metrics.csv'
OUTPUT_CSV = '../data/processed/schools_combined.csv'

# Load school address info
school_info = {}
with open(SCHOOLS_CSV, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        school_info[row['SCH_NAME']] = row

# Load school metrics
school_metrics = {}
with open(METRICS_CSV, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        school_metrics[row['School Name']] = row

# Prepare output
with open(MAPPING_CSV, newline='', encoding='utf-8') as infile, \
     open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = (
        ['Neighborhood'] +
        [f'{level} Name' for level in ['Elementary', 'Middle', 'High']] +
        [f'{level} Address' for level in ['Elementary', 'Middle', 'High']] +
        [f'{level} Rating' for level in ['Elementary', 'Middle', 'High']] +
        [f'{level} Student Progress' for level in ['Elementary', 'Middle', 'High']] +
        [f'{level} Test Score' for level in ['Elementary', 'Middle', 'High']] +
        [f'{level} Equity' for level in ['Elementary', 'Middle', 'High']] +
        [f'{level} Student-Teacher Ratio' for level in ['Elementary', 'Middle', 'High']] +
        [f'{level} Enrollment' for level in ['Elementary', 'Middle', 'High']]
    )
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in reader:
        out = {'Neighborhood': row['Neighborhood']}
        for level, col in zip(['Elementary', 'Middle', 'High'],
                              ['Elementary School', 'Middle School', 'High School']):
            school = row[col]
            # Address info
            addr = ''
            if school in school_info:
                addr = f"{school_info[school]['LSTREET1']}, {school_info[school]['LCITY']}, {school_info[school]['LSTATE']} {school_info[school]['LZIP']}"
            # Metrics
            metrics = school_metrics.get(school, {})
            out[f'{level} Name'] = school
            out[f'{level} Address'] = addr
            out[f'{level} Rating'] = metrics.get('Rating', '')
            out[f'{level} Student Progress'] = metrics.get('Student Progress', '')
            out[f'{level} Test Score'] = metrics.get('Test Score', '')
            out[f'{level} Equity'] = metrics.get('Equity', '')
            out[f'{level} Student-Teacher Ratio'] = metrics.get('Student-Teacher Ratio', '')
            out[f'{level} Enrollment'] = metrics.get('Enrollment', '')
        writer.writerow(out)

print(f'Combined data saved to {OUTPUT_CSV}') 