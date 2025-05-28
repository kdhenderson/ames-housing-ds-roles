# Data Steward Guide

This guide provides technical and administrative information for managing data access and security.

## Access Control System

### Files
- `access_control.py`: Script for controlling data access
- `access_monitor.py`: Script for monitoring access logs
- `data_access_log.db`: SQLite database for access logging (in `data/sensitive/`)

### User Management
1. To add a new user:
   - Edit the whitelist in `access_control.py`
   - Add user ID and access level
   - Document the addition in the access log

2. To remove a user:
   - Remove from whitelist
   - Document the removal
   - Review their access history

## Monitoring Access

1. Review access logs:
   ```bash
   python access_monitor.py
   ```

2. The script provides:
   - User access history
   - Data fields accessed
   - Access timestamps
   - Purpose of access

## Security Implementation

### File Permissions
- Sensitive data files: `chmod 600`
- Log database: `chmod 600`
- Scripts: `chmod 700`

### Access Levels
1. Public: Aggregated/anonymized data
2. Internal: Detailed data with sensitive fields masked
3. Sensitive: Full data access (restricted)

## Log Management

### Database Structure
- User ID
- Timestamp
- Fields accessed
- Purpose
- Action taken

### Maintenance
- Regular log review
- Archive old logs
- Monitor for unusual patterns
- Generate access reports

## Policy Enforcement

### Data Fields
| Field | Sensitivity | Masking Required | Access Level |
|-------|-------------|------------------|--------------|
| Address | High | Yes | Sensitive |
| Owner Info | High | Yes | Sensitive |
| Sale Date | Medium | Yes | Internal |
| Demographics | High | Yes | Sensitive |

### Compliance
- Regular access audits
- Policy updates
- User training
- Security reviews

## Contact

For technical support or policy questions, contact the project administrator. 