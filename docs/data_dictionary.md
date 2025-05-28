# Ames Housing Dataset Dictionary

## Overview
This document provides a comprehensive reference for the Ames Housing dataset after ETL processing. It includes original features, transformations applied, and new engineered features.

## Complete Feature List

### Numerical Features
| Feature Name | Description | Units | Missing Value Handling |
|--------------|-------------|-------|------------------------|
| Id | Unique identifier for each house | - | Preserved as is |
| MS SubClass | Building class | - | Mean imputation |
| Lot Frontage | Linear feet of street connected to property | feet | Mean imputation |
| Lot Area | Lot size in square feet | sq ft | Mean imputation |
| Overall Qual | Overall material and finish quality | 1-10 | Mean imputation |
| Overall Cond | Overall condition rating | 1-10 | Mean imputation |
| Year Built | Original construction date | year | Mean imputation |
| Year Remod/Add | Remodel date | year | Mean imputation |
| Mas Vnr Area | Masonry veneer area | sq ft | Mean imputation |
| BsmtFin SF 1 | Type 1 finished square feet | sq ft | Mean imputation |
| BsmtFin SF 2 | Type 2 finished square feet | sq ft | Mean imputation |
| Bsmt Unf SF | Unfinished square feet of basement area | sq ft | Mean imputation |
| Total Bsmt SF | Total square feet of basement area | sq ft | Mean imputation |
| 1st Flr SF | First floor square feet | sq ft | Mean imputation |
| 2nd Flr SF | Second floor square feet | sq ft | Mean imputation |
| Low Qual Fin SF | Low quality finished square feet | sq ft | Mean imputation |
| Gr Liv Area | Above grade (ground) living area | sq ft | Mean imputation |
| Bsmt Full Bath | Basement full bathrooms | count | Mean imputation |
| Bsmt Half Bath | Basement half bathrooms | count | Mean imputation |
| Full Bath | Full bathrooms above grade | count | Mean imputation |
| Half Bath | Half baths above grade | count | Mean imputation |
| Bedroom AbvGr | Bedrooms above grade | count | Mean imputation |
| Kitchen AbvGr | Kitchens above grade | count | Mean imputation |
| TotRms AbvGrd | Total rooms above grade | count | Mean imputation |
| Fireplaces | Number of fireplaces | count | Mean imputation |
| Garage Cars | Size of garage in car capacity | count | Mean imputation |
| Garage Area | Size of garage in square feet | sq ft | Mean imputation |
| Wood Deck SF | Wood deck area in square feet | sq ft | Mean imputation |
| Open Porch SF | Open porch area in square feet | sq ft | Mean imputation |
| Enclosed Porch | Enclosed porch area in square feet | sq ft | Mean imputation |
| 3Ssn Porch | Three season porch area in square feet | sq ft | Mean imputation |
| Screen Porch | Screen porch area in square feet | sq ft | Mean imputation |
| Pool Area | Pool area in square feet | sq ft | Mean imputation |
| Misc Val | Value of miscellaneous feature | dollars | Mean imputation |
| Mo Sold | Month Sold | 1-12 | Mean imputation |
| Yr Sold | Year Sold | year | Mean imputation |
| SalePrice | Target variable: Sale price | dollars | Preserved as is |
| TotalSF | Engineered: Total square feet | sq ft | Sum of components |

### Ordinal Features
| Feature Name | Description | Values (Best to Worst) | Missing Value Handling |
|--------------|-------------|------------------------|------------------------|
| Exter Qual | Exterior material quality | Ex, Gd, TA, Fa, Po | Mode imputation |
| Exter Cond | Exterior condition | Ex, Gd, TA, Fa, Po | Mode imputation |
| Bsmt Qual | Basement height quality | Ex, Gd, TA, Fa, Po, NA | Mode imputation |
| Bsmt Cond | Basement condition | Ex, Gd, TA, Fa, Po, NA | Mode imputation |
| Bsmt Exposure | Basement walkout or garden level walls | Gd, Av, Mn, No, NA | Mode imputation |
| BsmtFin Type 1 | Rating of basement finished area | GLQ, ALQ, BLQ, Rec, LwQ, Unf, NA | Mode imputation |
| BsmtFin Type 2 | Rating of basement finished area (if multiple types) | GLQ, ALQ, BLQ, Rec, LwQ, Unf, NA | Mode imputation |
| Heating QC | Heating quality and condition | Ex, Gd, TA, Fa, Po | Mode imputation |
| Kitchen Qual | Kitchen quality | Ex, Gd, TA, Fa, Po | Mode imputation |
| Fireplace Qu | Fireplace quality | Ex, Gd, TA, Fa, Po, NA | Mode imputation |
| Garage Finish | Interior finish of the garage | Fin, RFn, Unf, NA | Mode imputation |
| Garage Qual | Garage quality | Ex, Gd, TA, Fa, Po, NA | Mode imputation |
| Garage Cond | Garage condition | Ex, Gd, TA, Fa, Po, NA | Mode imputation |
| Pool QC | Pool quality | Ex, Gd, TA, Fa, NA | Mode imputation |
| Fence | Fence quality | GdPrv, MnPrv, GdWo, MnWw, NA | Mode imputation |
| Functional | Home functionality | Typ, Min1, Min2, Mod, Maj1, Maj2, Sev, Sal | Mode imputation |
| Land Slope | Slope of property | Gtl, Mod, Sev | Mode imputation |
| Lot Shape | General shape of property | Reg, IR1, IR2, IR3 | Mode imputation |
| Paved Drive | Paved driveway | Y, P, N | Mode imputation |
| Utilities | Type of utilities available | AllPub, NoSewr, NoSeWa, ELO | Mode imputation |

### Nominal Features
| Feature Name | Description | Categories | Missing Value Handling |
|--------------|-------------|------------|------------------------|
| MS Zoning | Identifies the general zoning classification | A, C, FV, I, RH, RL, RP, RM | Mode imputation |
| Street | Type of road access to property | Grvl, Pave | Mode imputation |
| Alley | Type of alley access | Grvl, Pave, NA | Mode imputation |
| Land Contour | Flatness of the property | Lvl, Bnk, HLS, Low | Mode imputation |
| Lot Config | Lot configuration | Inside, Corner, CulDSac, FR2, FR3 | Mode imputation |
| Neighborhood | Physical locations within Ames city limits | Blmngtn: Bloomington Heights, Blueste: Bluestem, BrDale: Briardale, BrkSide: Brookside, ClearCr: Clear Creek, CollgCr: College Creek, Crawfor: Crawford, Edwards: Edwards, Gilbert: Gilbert, Greens: Greens, GrnHill: Green Hills, IDOTRR: Iowa DOT and Rail Road, Landmrk: Landmark, MeadowV: Meadow Village, Mitchel: Mitchell, Names: North Ames, NoRidge: Northridge, NPkVill: Northpark Villa, NridgHt: Northridge Heights, NWAmes: Northwest Ames, OldTown: Old Town, SWISU: South & West of Iowa State University, Sawyer: Sawyer, SawyerW: Sawyer West, Somerst: Somerset, StoneBr: Stone Brook, Timber: Timberland, Veenker: Veenker | Mode imputation |
| Condition 1 | Proximity to various conditions | Norm, Feedr, PosN, PosA, RRNe, RRNn, RRAn, RRAe | Mode imputation |
| Condition 2 | Proximity to various conditions (if more than one is present) | Norm, Feedr, PosN, PosA, RRNe, RRNn, RRAn, RRAe | Mode imputation |
| Bldg Type | Type of dwelling | 1Fam, 2FmCon, Duplex, TwnhsE, Twnhs | Mode imputation |
| House Style | Style of dwelling | 1Story, 1.5Fin, 1.5Unf, 2Story, 2.5Fin, 2.5Unf, SFoyer, SLvl | Mode imputation |
| Roof Style | Type of roof | Flat, Gable, Gambrel, Hip, Mansard, Shed | Mode imputation |
| Roof Matl | Roof material | ClyTile, CompShg, Membran, Metal, Roll, Tar&Grv, WdShake, WdShngl | Mode imputation |
| Exterior 1st | Exterior covering on house | Multiple categories | Mode imputation |
| Exterior 2nd | Exterior covering on house (if more than one material) | Multiple categories | Mode imputation |
| Mas Vnr Type | Masonry veneer type | BrkCmn, BrkFace, CBlock, None, Stone | Mode imputation |
| Foundation | Type of foundation | BrkTil, CBlock, PConc, Slab, Stone, Wood | Mode imputation |
| Heating | Type of heating | Floor, GasA, GasW, Grav, OthW, Wall | Mode imputation |
| Central Air | Central air conditioning | N, Y | Mode imputation |
| Electrical | Electrical system | SBrkr, FuseA, FuseF, FuseP, Mix | Mode imputation |
| Misc Feature | Miscellaneous feature not covered in other categories | Elev, Gar2, Othr, Shed, TenC, NA | Mode imputation |
| Sale Type | Type of sale | WD, CWD, VWD, New, COD, Con, ConLw, ConLI, ConLD, Oth | Mode imputation |
| Sale Condition | Condition of sale | Normal, Abnorml, AdjLand, Alloca, Family | Mode imputation |

## Transformation Rules

### Categorical Encoding
1. **Ordinal Variables**
   - Quality ratings: Ex=4, Gd=3, TA=2, Fa=1, Po=0
   - Condition ratings: Ex=4, Gd=3, TA=2, Fa=1, Po=0
   - NA values are encoded as 0 where applicable

2. **Nominal Variables**
   - One-hot encoding creates binary columns
   - Column naming format: `{original_feature}_{category}`
   - Example: `MS_Zoning_RL`, `MS_Zoning_RM`, etc.

### Missing Value Handling
1. **Numeric Features**
   - Mean imputation for missing values
   - Exceptions: Id and SalePrice preserved as is

2. **Categorical Features**
   - Mode imputation for missing values
   - NA values preserved where they have meaning (e.g., basement features)

## Engineered Features

### TotalSF
- **Calculation**: Total Bsmt SF + 1st Flr SF + 2nd Flr SF
- **Purpose**: Provides total living area regardless of floor
- **Missing Value Handling**: Sum of available components, 0 for missing components

## Data Quality Notes
1. **Outliers**
   - No outlier removal in current ETL pipeline
   - Modelers should consider outlier analysis

2. **Data Validation**
   - All numeric values are non-negative
   - Categorical values are validated against known categories
   - UTF-8 encoding ensures proper handling of special characters

## Usage Notes for Modelers
1. **Feature Selection**
   - Consider correlation between TotalSF and individual square footage features
   - One-hot encoded features may require dimensionality reduction
   - Ordinal features maintain meaningful order for modeling

2. **Preprocessing Considerations**
   - Scale numeric features before modeling
   - Consider feature importance for dimensionality reduction
   - Handle any remaining missing values in the modeling pipeline

## Version History
- v1.0: Initial ETL pipeline implementation
  - Basic data cleaning and preprocessing
  - Missing value handling (mean/mode imputation)
  - Categorical encoding (label encoding for ordinal, one-hot for nominal)
  - TotalSF feature engineering
  - Comprehensive data dictionary documentation