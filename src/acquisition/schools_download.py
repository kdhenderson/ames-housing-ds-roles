import requests
import zipfile
import io
import csv

# 1. Download NCES public school data (latest year)
nces_url = "https://nces.ed.gov/ccd/Data/zip/ccd_sch_029_2324_w_1a_073124.zip"  # 2021-22, update if needed
response = requests.get(nces_url)
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    # Find the CSV file in the zip
    csv_filename = [name for name in z.namelist() if name.endswith('.csv')][0]
    with z.open(csv_filename) as csvfile:
        reader = csv.DictReader(io.TextIOWrapper(csvfile, encoding='utf-8'))
        ames_schools = []
        for row in reader:
            if row['LCITY'].strip().upper() == 'AMES' and row['LSTATE'].strip().upper() == 'IA':
                # Extract relevant columns
                school = {
                    'NCESSCH': row['NCESSCH'],
                    'SCH_NAME': row['SCH_NAME'],
                    'LEA_NAME': row['LEA_NAME'],
                    'LSTREET1': row['LSTREET1'],
                    'LCITY': row['LCITY'],
                    'LSTATE': row['LSTATE'],
                    'LZIP': row['LZIP']
                }
                ames_schools.append(school)

# Save the results to a CSV file
output_file = '../data/raw/schools_raw.csv'
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=ames_schools[0].keys())
    writer.writeheader()
    writer.writerows(ames_schools)

print(f'Found {len(ames_schools)} schools in Ames, IA')
print(f'Data saved to {output_file}')

# Print the results for verification
for school in ames_schools:
    print(school) 