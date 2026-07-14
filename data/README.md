## Data Dictionary

Based on the initial provided context and the actual sampled values from our data profiling, here is the refined breakdown of all 54 columns in the dataset.

| # | Column | Group | Definition |
|---|---|---|---|
| 1 | `period_id` | Core Identifiers & Time | Date, weekly granularity (e.g., '2023-09-23'). The start or end date of the tracking week. |
| 2 | `week_nm` | Core Identifiers & Time | Integer representing the week of the year (e.g., 38, 40, 50, 25). |
| 3 | `month_nm` | Core Identifiers & Time | The month name corresponding to the `period_id`. |
| 4 | `year_nm` | Core Identifiers & Time | The year of the data record (e.g., 'Y2022', 'Y2023', 'Y2024', 'Y2025'). |
| 5 | `refresh_cycle` | Core Identifiers & Time | An internal batch or refresh ID (constant value: 202511). |
| 6 | `product_id` | Product & Hierarchy | EAN/SKU code representing the specific item. |
| 7 | `rgm_ppg` | Product & Hierarchy | Group of products according to business definition. It is a concatenated string of PPG attributes (e.g., 'PPG_ATTR1_F639... \| PPG_ATTR2_2BEF...'). |
| 8 | `cmu` | Product & Hierarchy | Commercial/Category Management Unit (constant value: 'UK_DISHCARE'). |
| 9 | `category` | Product & Hierarchy | The broad category of the product (constant value: 'DISHWASHER PRODUCTS'). |
| 10 | `subcategory_nm` | Product & Hierarchy | Sub-segment of the category (duplicated exactly with `brand_nm` and others in this specific dataset). |
| 11 | `manufacturer_nm` | Product & Hierarchy | The company that manufactures the product (hashed values like 'HASH_F6394E71'). |
| 12 | `is_own_manufacturer` | Product & Hierarchy | Boolean or flag indicating if the manufacturer is the focal company. |
| 13 | `brand_nm` | Product & Hierarchy | The brand name of the product. |
| 14 | `subbrand_nm` | Product & Hierarchy | The sub-brand or specific line under the main brand. |
| 15 | `attribute_1` | Product Attributes | Product Form / Format (e.g., 'TABLETS', 'LIQUID', 'GEL', 'POWDER'). |
| 16 | `attribute_2` | Product Attributes | Product Weight / Volume (e.g., '841 GM', '250 ML', '2000 GM'). |
| 17 | `attribute_3` | Product Attributes | Promotional/Bonus Pack status (e.g., 'STANDARD', 'PMP', '20% EXTRA', 'BANDED PACK'). |
| 18 | `attribute_4` | Product Attributes | Pack Configuration / Multipack quantity (e.g., 'SINGLE', '2 PACK', '6 PACK'). |
| 19 | `attribute_5` | Product Attributes | Packaging Type (e.g., 'DOY', 'BOX', 'BOTTLE', 'TUB'). |
| 20 | `attribute_6` | Product Attributes | Product Subtype / Role (e.g., 'NOT APPLICABLE', 'ADDITIVES'). |
| 21 | `attribute_7` | Product Attributes | Benefit Type (e.g., 'MULTI BENEFIT', 'MONOBENEFIT'). |
| 22 | `attribute_8` | Product Attributes | Additional product descriptor (context dependent). |
| 23 | `attribute_9` | Product Attributes | Additional product descriptor (context dependent). |
| 24 | `ppg_attr1` | Product Attributes | Manufacturer Name (hashed) used to define the PPG string. |
| 25 | `ppg_attr2` | Product Attributes | Category ('DISHWASHER PRODUCTS') used to define the PPG string. |
| 26 | `ppg_attr3` | Product Attributes | Brand Name (hashed) used to define the PPG string. |
| 27 | `ppg_attr4` | Product Attributes | Subcategory / Product Type (e.g., 'DETERGENT', 'SALT', 'CLEANER', 'RINSE AID'). |
| 28 | `ppg_attr5` | Product Attributes | Sub-brand or product variant (hashed) used to define the PPG string. |
| 29 | `ppg_attr6` | Product Attributes | Product Form (e.g., 'TABLETS WITH SOLUBLE FILM', 'LIQUID', 'POWDER'). |
| 30 | `ppg_attr7` | Product Attributes | Scent / Fragrance (e.g., 'LEMON', 'ORIGINAL', 'APPLE & NASHI PEAR'). |
| 31 | `ppg_attr8` | Product Attributes | Pack Size and Unit of Measure (e.g., '52 CT', '2000 GM', '250 ML'). |
| 32 | `ppg_attr9` | Product Attributes | Pack Count / Multipack quantity (e.g., 1, 2, 4, 6). |
| 33 | `ppg_attr10` | Product Attributes | Completely empty/NULL column. |
| 34 | `ppg_attr11` | Product Attributes | Completely empty/NULL column. |
| 35 | `ppg_attr12` | Product Attributes | Constant value attribute related to PPG ('DISHWASHER PRODUCTS'). |
| 36 | `product_pack_size` | Pack & Size Dimensions | The numeric size of the pack (e.g., 52, 26, 2000, 250). |
| 37 | `product_unit_size` | Pack & Size Dimensions | The size of an individual unit within the pack. |
| 38 | `uom` | Pack & Size Dimensions | Unit of Measure for the pack sizes. |
| 39 | `product_is_multipack` | Pack & Size Dimensions | Flag indicating if it's a multipack bundle ('YES' or 'NO'). |
| 40 | `product_pack_count` | Pack & Size Dimensions | Number of individual items/packs bundled together (e.g., 1, 2, 4, 6, 8). |
| 41 | `retailer_nm` | Retailer & Geography | Retailer or store banner where the product was sold (hashed values like 'HASH_4456A3C8'). |
| 42 | `channel_nm` | Retailer & Geography | The retail channel classification (constant value: 'Supermarkets'). |
| 43 | `total_units` | Base Sales Metrics | Sales in number of packs. |
| 44 | `total_volume` | Base Sales Metrics | Sales in doses or standardized volume measure. |
| 45 | `total_sales` | Base Sales Metrics | Sales in currency. |
| 46 | `any_promo_units` | Promotional Sales & Volume | Number of packs sold while under any type of promotion. |
| 47 | `any_promo_amt` | Promotional Sales & Volume | Currency value of sales generated while under promotion. |
| 48 | `any_promo_eq` | Promotional Sales & Volume | Promoted volume (equivalized/doses). |
| 49 | `acv_pct` | Distribution & Execution Metrics | Overall availability in stores, weighted by retailer importance/size. |
| 50 | `any_promo_acv_pct` | Distribution & Execution Metrics | Weighted distribution of stores running *any* promotion for the product (e.g., values up to 88+). |
| 51 | `acv_tpr_only` | Distribution & Execution Metrics | Promo availability by promo mechanic (Temporary Price Reduction only). |
| 52 | `feature_and_display_acv_pct` | Distribution & Execution Metrics | Promo availability where product had both Feature and Display. |
| 53 | `feature_without_display_acv_pct` | Distribution & Execution Metrics | Promo availability (Feature without Display). |
| 54 | `display_without_feature_acv_pct` | Distribution & Execution Metrics | Promo availability (Display without Feature, e.g., values around 0 to 4). |