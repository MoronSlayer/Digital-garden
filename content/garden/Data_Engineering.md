---
title: "Data Engineering"
date: 2024-01-25
lastmod: 2024-01-25
draft: false
garden_tags: ["data"]
summary: "Notes from data engineering zoomcamp"
status: "seeding"
---
The goal of this course is to build a data pipeline using TLC Trip Record data, which contains pickups and dropoffs of Taxis in NYC. 
This is the architecture used in the course.

{{< figure src="./architecture.jpeg"  width="100%" >}}
Tools used as part of this course are:
- #### [Docker](#docker)
- #### [PostgresSQL](#ingesting-ny-taxi-data-to-postgres)
# Week 1

## Docker

Docker is a platform that uses containerization technology to simplify the process of developing, deploying, and running applications. By using Docker, you can package your application and its dependencies into a container, which can then run on any system that has Docker installed. This ensures consistency across environments and reduces the "it works on my machine" problem.

Key concepts in Docker include:

1. Containers: These are lightweight, standalone, and executable packages that include everything needed to run an application: code, runtime, system tools, system libraries, and settings. Containers are isolated from one another and the host system.

2. Images: A Docker image is a read-only template used to create containers. Images are created with the docker build command, and they are based on a Dockerfile.

3. Dockerfile: This is a text file containing a series of instructions for Docker to build an image. It specifies the base image to use, the software to install, the files to add, and the commands to run.

4. Docker Hub: A cloud-based registry service where you can find and share container images with your team. It's similar to GitHub but for Docker images.

5. Volumes: These are used to persist and share data between a container and the host file system.

6. Docker Compose: A tool for defining and running multi-container Docker applications. With Compose, you use a YAML file to configure your application’s services, networks, and volumes, and then create and start all the services from your configuration with a single command.

Docker simplifies the deployment process, ensures consistency, and is widely used in DevOps practices for continuous integration and continuous deployment (CI/CD) workflows.

## Ingesting NY Taxi Data to Postgres

PostgreSQL, often referred to as Postgres, is an open-source, powerful, advanced, and feature-rich relational database management system (RDBMS). It is designed to handle a range of workloads, from single machines to data warehouses or web services with many concurrent users.

Firstly, the docker command required to run postgres locally:

```docker
docker run -it \
    -e POSTGRES_USER="root" \
    -e POSTGRES_PASSWORD="root" \
    -e POSTGRES_DB="ny_taxi" \
    -v "absolute_path_to_folder_where_you_want_database:/var/lib/postgresql/data" \
    -p 5432:5432 \
    postgres:13
```

The -v flag is used for mapping a folder from host machine to a folder on our container, also called mounting. -e is environment flag, and port 5432 is used to connect external sources (here, our python script) to the database.

If the above code throws permission errors, use this code

```docker
docker volume create --name dtc_postgres_volume_local -d local
docker run -it\
  -e POSTGRES_USER="root" \
  -e POSTGRES_PASSWORD="root" \
  -e POSTGRES_DB="ny_taxi" \
  -v dtc_postgres_volume_local:/var/lib/postgresql/data \
  -p 5432:5432\
  postgres:13
```

Run the above code in your command line.

Now install pglci

```bash
pip install pgcli

pgcli
```

then connect it using

```bash
pgcli -h localhost -p 5432 -u root -d ny_taxi

```

Download data from [here](https://github.com/DataTalksClub/nyc-tlc-data/releases/tag/yellow)

```bash 
wget https://github.com/DataTalksClub/nyc-tlc-data/releases/download/yellow/yellow_tripdata_2021-07.csv.gz
```

using the following python command, you can import and visualize the data 

```python
# import libraries
import pandas as pd
from sqlalchemy import create_engine

# create engine and set the root as postgresql://user:password@host:port/database
engine = create_engine('postgresql://root:root@localhost:5432/ny_taxi')

df_iter = pd.read_csv('yellow_tripdata_2021-07.csv.gz', iterator=True, chunksize=100000)

while True: #iterate and read chunks of data and append it to the table
    df = next(df_iter)
    df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)
    df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
    df.to_sql(name='yellow_taxi_data', con=engine, if_exists='append')
```

Then using ```\dt``` command, we can view the tables in the database, and using ```\d yellow_taxi_data```, we see the imported data schema.

{{< figure src="./dt_command.png"  width="100%" >}}

