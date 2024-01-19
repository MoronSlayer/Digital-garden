---
title: "Snowflake"
date: 2023-12-17
lastmod: 2023-12-17
draft: false
garden_tags: ["data_warehouse","cloud"]
summary: " "
status: "seeding" 
---
<!-- Snowflake offers one of the best data-warehousing solutions for its unique architecture and capabilities. Unlike traditional data warehouses that require physical hardware and significant maintenance, Snowflake is entirely cloud-based, providing scalability, flexibility, and ease of management without the need for physical infrastructure. Snowflake separates compute and storage resources. This means that you can scale storage and compute independently, allowing for more efficient use of resources and cost-effectiveness. It can handle large volumes of data without impacting performance. Snowflake can manage both structured and semi-structured data (like JSON, Avro, or XML). This flexibility makes it suitable for various types of data analytics applications. Due to its unique architecture, Snowflake can perform high-speed data processing and analytics. It automatically optimizes both storage and compute operations, making it efficient for large-scale data operations. -->

# Snowflake: Cloud-Based Data Warehousing

Snowflake is a cloud-based data warehousing platform known for its unique architecture and capabilities. It offers various features that are beneficial for data analytics and storage.

## Key Aspects of Snowflake

### 1. Cloud-Based Data Warehouse
Snowflake is entirely cloud-based, providing scalability, flexibility, and ease of management without the need for physical infrastructure.

### 2. Unique Architecture
It separates compute and storage resources, allowing for independent scaling and cost-effective use of resources.

### 3. Support for Diverse Data
Snowflake can manage both structured and semi-structured data, like JSON, Avro, or XML.

### 4. Performance and Speed
Due to its architecture, Snowflake can perform high-speed data processing and analytics efficiently.

### 5. Compatibility and Integration
It integrates with a wide range of data integration, ETL (Extract, Transform, Load), and BI (Business Intelligence) tools.

### 6. Security and Compliance
Snowflake provides robust security features and complies with various regulatory standards.

### 7. Data Sharing Capabilities
Enables secure data sharing between Snowflake users, useful for collaborative data-driven projects.

### 8. Pay-As-You-Go Pricing
Its pricing model is typically usage-based, offering a cost-effective solution compared to traditional data warehousing.


>Snowflake's combination of performance, scalability, and ease of use makes it a popular choice for leveraging cloud-based data warehousing in data analytics.

---

# Notes from Collaboration, Marketplace & Cost Estimation Workshop:

Imagine your company has decided to use Snowflake, migrating from an older on-premise database system into the cloud. These are some notes on the management and administration of the extract process which can be redesigned so that customers can receive data via Snowflake sharing instead. 

### Understanding Inbound Shares:

- You can view the shared databases by going to the main account's page in Snowflake and then Data -> Databases or Private Sharing 
- By default you have two databases, SNOWFLAKE_SAMPLE_DATA and SNOWFLAKE
- Under Private sharing section we can see that the database called SNOWFLAKE_SAMPLE_DATA is coming from an account called SFSALESSHARED (followed by a schema that will vary by region). This account named their outbound share "SAMPLE_DATA. An account named SNOWFLAKE is the source of our database called SNOWFLAKE but their original data source is named ACCOUNT_USAGE
- SNOWFLAKE is a share given to every account and behind the scenes it's based on a direct share called ACCOUNT_USAGE, it includes schemas that are intended to help customers manage and understand their billing and usage.
- If you drop a default database like the SNOWFLAKE_SAMPLE_DATA, you can see from the Private Sharing page that it is downloadable now, and the title of the database is not displayed, i.e, instead of seeing both SNOWFLAKE_SAMPLE_DATA at the bottom of the tile and the original data source at the top, we only see SAMPLE_DATA which is the name of the original data source at the top of the tile.

Using the below SQL command, we can change the database names
```SQL
alter database DEMO 
rename to snowflake_sample_data;
```
- You cannot drop the SNOWFLAKE database(original data source name: ACCOUNT_USAGE, remember!)

- If a user e.g., SysAdmin does not have access to a specific database (SNOWFLAKE_SAMPLE_DATA), you can grant privilege from the database UI or use the SQL command below:

```SQL 
grant imported privileges
on database SNOWFLAKE_SAMPLE_DATA
to role SYSADMIN;
```
----
Using **SELECT** statements:

```SQL
--Check the range of values in the Market Segment Column
SELECT DISTINCT c_mktsegment
FROM SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.CUSTOMER;

--Find out which Market Segments have the most customers
SELECT c_mktsegment, COUNT(*)
FROM SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.CUSTOMER
GROUP BY c_mktsegment
ORDER BY COUNT(*);
```
{{< figure src="./SELECT_sampleData.jpg" title="Select statement for sample and data and view of Worksheet in Snowflake" width="100%" >}}

----

Joining and Aggregating data:



```SQL 
-- Nations Table
SELECT N_NATIONKEY, N_NAME, N_REGIONKEY
FROM SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.NATION;

-- Regions Table
SELECT R_REGIONKEY, R_NAME
FROM SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.REGION;

-- Join the Tables and Sort
SELECT R_NAME as Region, N_NAME as Nation
FROM SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.NATION 
JOIN SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.REGION 
ON N_REGIONKEY = R_REGIONKEY
ORDER BY R_NAME, N_NAME ASC;

--Group and Count Rows Per Region
SELECT R_NAME as Region, count(N_NAME) as NUM_COUNTRIES
FROM SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.NATION 
JOIN SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.REGION 
ON N_REGIONKEY = R_REGIONKEY
GROUP BY R_NAME;
```
{{< figure src="./JOIN_aggData.jpg" title="Join and Aggregate Shared Data" width="100%" >}}

-----
Warehouses can be costly when not carefully managed, so their use is restricted to Administrative users. Because of this, Warehouse functions are under the Admin menu. 

Any warehouse that has to be given access to another user, e.g., SYSADMIN should be given access to COMPUTE_WH can be done by granting priveleges.

{{< figure src="./grantComputeWHPrivelege.png"  width="100%" >}}

----

#### Creating a local database with a different user

{{< figure src="./newLocalDB1.png"  width="100%" >}}

{{< figure src="./newLocalDB2.png"  width="100%" >}}


-----

### Usage and Cost Management

Cost of different accounts and regions can be found at https://www.snowflake.com/en/data-cloud/pricing-options/

From the Admin account, we can see the cost of running any warehouse as shown in the image below

{{< figure src="./usageCost.png"  width="100%" >}}

In this image we have used 53% of 1 credit for the COMPUTE_WH warehouse

----

### Creating a Local Database and Warehouse

```SQL
use role SYSADMIN;

create database INTL_DB;

use schema INTL_DB.PUBLIC;
```
Here we created a new database valled INTL_DB, now let's create a Warehouse for this database

```SQL
use role SYSADMIN;

create warehouse INTL_WH 
with 
warehouse_size = 'XSMALL' 
warehouse_type = 'STANDARD' 
auto_suspend = 600 --600 seconds/10 mins
auto_resume = TRUE;

use warehouse INTL_WH;
```

Now, creating a New Table in our database INTL_DB

```SQL
create or replace table intl_db.public.INT_STDS_ORG_3166 
(iso_country_name varchar(100), 
 country_name_official varchar(200), 
 sovreignty varchar(40), 
 alpha_code_2digit varchar(2), 
 alpha_code_3digit varchar(3), 
 numeric_country_code integer,
 iso_subdivision varchar(15), 
 internet_domain_code varchar(10)
);
```
This will be created in the public schema.

Creating a file format to load the table:

```SQL
create or replace file format util_db.public.PIPE_DBLQUOTE_HEADER_CR 
  type = 'CSV' --use CSV for any flat file
  compression = 'AUTO' 
  field_delimiter = '|' --pipe or vertical bar
  record_delimiter = '\r' --carriage return
  skip_header = 1  --1 header row
  field_optionally_enclosed_by = '\042'  --double quotes
  trim_space = FALSE;

```
Here:
- We created a file format.
- file format will be used to process a flat text file.
- file format will skip the first row (header).
- file format will know how to separate values because of a pipe separator.

---

### Loading ISO Table Using File Format from a new stage

##### Creating Stage from UI

{{< figure src="./createStage_step1.png"  width="100%" >}}

{{< figure src="./createStage_step2.png"  width="100%" >}}

Now granting privileges to SysAdmin:

{{< figure src="./createStage_step3.png"  width="100%" >}}

Alternatively, using SQL:

```SQL
create stage util_db.public.aws_s3_bucket url = 's3://uni-cmcw';

```

We need to load a file called **iso_countries_utf8_pipe.csv** and AWS is case sensitive, so to check:
```SQL
list @util_db.public.aws_s3_bucket;
```

#### Loading data into the table from stage
```SQL
copy into INT_STDS_ORG_3166
from @UTIL_DB.PUBLIC.AWS_S3_BUCKET
files = ( 'ISO_Countries_UTF8_pipe.csv')
file_format = ( format_name='UTIL_DB.PUBLIC.PIPE_DBLQUOTE_HEADER_CR' );
```
{{< figure src="./createStage_step4.png"  width="100%" >}}

(*dropped ISO table as stage name and used SQL to recreate the stage*)

----

#### Testing that we loaded the expected number of rows

```SQL
select row_count
from INTL_DB.INFORMATION_SCHEMA.TABLES 
where table_schema='PUBLIC'
and table_name= 'INT_STDS_ORG_3166';
```

#### Joining local and shared data

```SQL
select  
     iso_country_name
    ,country_name_official,alpha_code_2digit
    ,r_name as region
from INTL_DB.PUBLIC.INT_STDS_ORG_3166 i
left join SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.NATION n
on upper(i.iso_country_name)= n.n_name
left join SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.REGION r
on n_regionkey = r_regionkey;
```

##### Converting Select statment into View

```SQL
create view intl_db.public.NATIONS_SAMPLE_PLUS_ISO 
( iso_country_name
  ,country_name_official
  ,alpha_code_2digit
  ,region) AS
  select  
    iso_country_name,
    country_name_official,
    alpha_code_2digit,
    r_name as region
    from INTL_DB.PUBLIC.INT_STDS_ORG_3166 i
    left join SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.NATION n
    on upper(i.iso_country_name)= n.n_name
    left join SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.REGION r
    on n_regionkey = r_regionkey;
;
```

- Creating a view makes a select statement behave like a table.

- If we find yourself using the logic of a select statement over and over again, it can be more convenient to wrap the statement in a view, and run simple queries on the view. 

- Views can make our code more modular and organized. 

###### Running a select statement on the view

```SQL
select *
from intl_db.public.NATIONS_SAMPLE_PLUS_ISO;
```

#### More data into INTL_DB, the database we created

Creating table currencies

```SQL
create table intl_db.public.CURRENCIES 
(
  currency_ID integer, 
  currency_char_code varchar(3), 
  currency_symbol varchar(4), 
  currency_digital_code varchar(3), 
  currency_digital_name varchar(30)
)
  comment = 'Information about currencies including character codes, symbols, digital codes, etc.';
```

Creating Table Country to Currency

```SQL
create table intl_db.public.COUNTRY_CODE_TO_CURRENCY_CODE 
  (
    country_char_code varchar(3), 
    country_numeric_code integer, 
    country_name varchar(100), 
    currency_name varchar(100), 
    currency_char_code varchar(3), 
    currency_numeric_code integer
  ) 
  comment = 'Mapping table currencies to countries';
```

Creating a File Format to Process files with Commas, Linefeeds and a Header Row

```SQL
create file format util_db.public.CSV_COMMA_LF_HEADER
  type = 'CSV' 
  field_delimiter = ',' 
  record_delimiter = '\n' -- the n represents a Line Feed character
  skip_header = 1 
;

```

Now, to load data from the S3 bucket in file stage

{{< figure src="./list_s3Files.png"  width="100%" >}}

```SQL
copy into CURRENCIES
from @UTIL_DB.PUBLIC.AWS_S3_BUCKET
files = ( 'currencies.csv')
file_format = ( format_name='UTIL_DB.PUBLIC.CSV_COMMA_LF_HEADER' );
```
```SQL
copy into COUNTRY_CODE_TO_CURRENCY_CODE
from @UTIL_DB.PUBLIC.AWS_S3_BUCKET
files = ( 'country_code_to_currency_code.csv')
file_format = ( format_name='UTIL_DB.PUBLIC.CSV_COMMA_LF_HEADER' );
```
{{< figure src="./newData.png"  width="100%" >}}


## Creating a new account for ACME organization with ORGADMIN role

{{< figure src="./orgAdmin.png"  width="100%" >}}

#### Listings 

To share data between different accounts, privately, listing can be used


{{< figure src="./listing_1.png"  width="100%" >}}
{{< figure src="./listing_2.png"  width="100%" >}}

### Pricing of Snowflake

Depends on 4 factors:

- Average compressed storage used per month
- compute credits based on amount of time warehouse is in use
- cloud services like state management
- serverless features like Snowpipe, replication, and clustering.