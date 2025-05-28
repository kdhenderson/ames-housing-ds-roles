# Data Access Guide

This guide explains how to access and use sensitive data in the project.

## Available Data

- Housing data with sensitive fields
- School data with performance metrics
- Neighborhood mapping data

## How to Access Data

1. Ensure you have been granted access by the Data Steward
2. Use the access script to retrieve data:
   ```bash
   python src/governance/access_control.py
   ```
3. Follow the prompts to:
   - Enter your user ID
   - Specify your purpose
   - Select the data you need

## Security Requirements

- Never share sensitive data outside approved channels
- Report any security concerns to the Data Steward
- Follow data masking guidelines when sharing results
- Keep access credentials secure

## User Responsibilities

- Use data only for approved purposes
- Maintain data confidentiality
- Report any data quality issues
- Follow project data handling policies

## Requesting Access

To request access to sensitive data:
1. Contact the Data Steward
2. Provide your user ID
3. Explain your need for access
4. Specify which data you require
5. Describe your intended use

## Data Locations

- Raw data: `data/raw/`
- Processed data: `data/processed/`
- Sensitive data: `data/sensitive/`

## Contact

For questions or to request access, contact the Data Steward. 