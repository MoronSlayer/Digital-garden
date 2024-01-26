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
- #### [pgAdmin](#connecting-pgadmin-and-postgres)
# Week 1

### Docker

Docker is a platform that uses containerization technology to simplify the process of developing, deploying, and running applications. By using Docker, you can package your application and its dependencies into a container, which can then run on any system that has Docker installed. This ensures consistency across environments and reduces the "it works on my machine" problem.

Key concepts in Docker include:

1. Containers: These are lightweight, standalone, and executable packages that include everything needed to run an application: code, runtime, system tools, system libraries, and settings. Containers are isolated from one another and the host system.

2. Images: A Docker image is a read-only template used to create containers. Images are created with the docker build command, and they are based on a Dockerfile.

3. Dockerfile: This is a text file containing a series of instructions for Docker to build an image. It specifies the base image to use, the software to install, the files to add, and the commands to run.

4. Docker Hub: A cloud-based registry service where you can find and share container images with your team. It's similar to GitHub but for Docker images.

5. Volumes: These are used to persist and share data between a container and the host file system.

6. Docker Compose: A tool for defining and running multi-container Docker applications. With Compose, you use a YAML file to configure your application’s services, networks, and volumes, and then create and start all the services from your configuration with a single command.

Docker simplifies the deployment process, ensures consistency, and is widely used in DevOps practices for continuous integration and continuous deployment (CI/CD) workflows.

### Ingesting NY Taxi Data to Postgres

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

We can also run SQL queries in pgcli as show below:

{{< figure src="./pgcli_queries.jpg"  width="100%" >}}

The maximum amount of money a customer had to pay was $1320.8 ?!

### Connecting pgAdmin and Postgres

We see that pgcli is good but not very convenient as it is part of the command line interface. For ease of use, we have a web based GUI tool called pgAdmin.

We can find the docker image for pgAdmin 4 (current version, 2024) from [here](https://www.pgadmin.org/download/pgadmin-4-container/) or in the docker hub [here](https://hub.docker.com/r/dpage/pgadmin4/).

> (use `ctrl`+`D` or `quit` command to quit pgcli in your terminal)

This is the docker command to run pgAdmin:

```docker
docker run -it \
    -e PGADMIN_DEFAULT_EMAIL="admin@admin.com" \
    -e PGADMIN_DEFAULT_PASSWORD="root" \
    -p "8080:80" \
    -d dpage/pgadmin4
```

The first two environment variables set default email and password for logging in, then the -p flag maps port 8080 from our host machine to port 80 on the container.pgAdmin will be running and listening for requests on port 80, so all requests we send to our host machine's port 8080 will be forwarded to port 80 on the container.

{{< figure src="./pgAdmin_docker.png"  width="100%" >}}

This will open up port 8080 on our local host machine and can be accessed from our browser.

{{< figure src="./localHost8080.png"  width="100%" >}}

`right click` on Servers, `click` Register and add your new server.

{{< figure src="./server_register.png"  width="100%" >}}

But when we try to establish a connection, we get an error

{{< figure src="./server_error.png"  width="100%" >}}

This is because our postgres/localhost is in one container and pgAdmin is running in a different container. We need to link these two, the database and pgAdmin by putting both these containers in a Docker network. 

{{< figure src="./docker_network.png"  width="100%" >}}

We do this by closing our container with the database and pgAdmin, then creating a docker network by the command 
```docker
docker network create pg-network
```

{{< figure src="./docker_network_cli.png"  width="100%" >}}

Then reinitializing the database container with our network using 

```docker
docker volume create --name dtc_postgres_volume_local -d local
docker run -it\
  -e POSTGRES_USER="root" \
  -e POSTGRES_PASSWORD="root" \
  -e POSTGRES_DB="ny_taxi" \
  -v dtc_postgres_volume_local:/var/lib/postgresql/data \
  -p 5432:5432\
  --network=pg-network \
  --name pg-database \
  postgres:13
```

We also add pgAdmin to our network 

```docker
docker run -it \
    -e PGADMIN_DEFAULT_EMAIL="admin@admin.com" \
    -e PGADMIN_DEFAULT_PASSWORD="root" \
    -p "8080:80" \
    --network=pg-network \
    --name pgadmin-2 \
    -d dpage/pgadmin4
```

Once we run the above command, we can restart localhost:8080 and register our new server with the host name we set to our database pg-network (see above).

{{< figure src="./server_register2.png"  width="100%" >}}


We are now connected to our database using pgAdmin and can view and query it. Let's see the first 100 rows, go to 

Databases --> Schemas --> Tables --> Yellow_taxi_data, right click --> View/Edit Data --> First 100 rows 

{{< figure src="./first100.png"  width="100%" >}}

### Dockerizing the Ingestion Script

Now, we convert the ipynb file to a script, using 

```python
 jupyter nbconvert --to=script upload_data.ipynb
```

We do this so that we can create a data ingestion pipeline using argparse in python, downloading the data and inserting the data into the table using a script.

We add the argarse and convert the notebook to an ingestion script, as seen in ingest_script.py.

We can now drop the table from pgAdmin using 
```SQL
DROP TABLE yellow_taxi_data 
```
and run the script to see if it's working.

We can now use the following script from the command line to run the data ingestion pipeline, downloading data and creating a table using ingest_data.py .

```bash
URL="https://github.com/DataTalksClub/nyc-tlc-data/releases/download/yellow/yellow_tripdata_2021-07.csv.gz"

python3 ingest_data.py \
  --user=root \
  --password=root \
  --host=localhost \
  --port=5432 \
  --db=ny_taxi \
  --table_name=yellow_taxi_trips \
  --url=${URL}
```

Now, we will dockerize this ingestion process by creating a Dockerfile in our working directory:

```Docker
FROM python:3.9.1

RUN apt-get install wget
RUN pip install pandas sqlalchemy psycopg2

WORKDIR /app
COPY ingest_data.py ingest_data.py 

ENTRYPOINT [ "python", "ingest_data.py" ]
```

and then running this in our command line:

```bash
docker build -t taxi_ingestion:v001 .
```

Once it's built, we create a docker script to run the ingestion pipeline using

```docker
docker run -it \
 --network=pg-network \
 taxi_ingestion:v001 \
  --user=root \
  --password=root \
  --host=pg-database \
  --port=5432 \
  --db=ny_taxi \
  --table_name=yellow_taxi_trips \
  --url=${URL}

```
### Running Postgres and pgAdmin with Docker compose

Previously, we run postgres and pgAdmin in one Docker network and there was a lot of configuration to be done. So instead of using two docker commands, we can specify a single YAML file that will help us do this task.

We create a compose-docker.yaml file as shown below

```docker
services:
  pgdatabase:
    image: postgres:13
    environment:
      - POSTGRES_USER=root
      - POSTGRES_PASSWORD=root
      - POSTGRES_DB=ny_taxi
    volumes:
      - "./ny_taxi_postgres_data:/var/lib/postgresql/data:rw"
    ports:
      - "5432:5432"
  pgadmin:
    image: dpage/pgadmin4
    environment:
      - PGADMIN_DEFAULT_EMAIL=admin@admin.com
      - PGADMIN_DEFAULT_PASSWORD=root
    ports:
      - "8080:80"
```

Now, pgdatabase will be the name we can access postgres with and both postgres and pgadmin will be in the same network as we are using docker compose, no need for creating another network.

Now, we stop the containers with postgres and pgadmin.

Ensure nothing is running with ``` docker ps ``` and use ` docker kill <containerID>` to stop any container you want.

We can now run ` docker-compose up ` to run docker compose using our YAML file and configure both pgadmin and postgres. 

Running `docker-compose up`and restarting localhost:8080 created problems in my system when trying to create a new server, as there were problems with postgres.
Specificially, a local bind ./ny_taxi_postgres_data seemed to throwing a permissions error, to resolve this, we can change the docker-compose.YAML file as follows:

```docker
services:
  pgdatabase:
    image: postgres:13
    environment:
      - POSTGRES_USER=root
      - POSTGRES_PASSWORD=root
      - POSTGRES_DB=ny_taxi
    volumes:
      - dtc_postgres_volume_local:/var/lib/postgresql/data
    ports:
      - "5432:5432"
  pgadmin:
    image: dpage/pgadmin4
    environment:
      - PGADMIN_DEFAULT_EMAIL=admin@admin.com
      - PGADMIN_DEFAULT_PASSWORD=root
    ports:
      - "8080:80"
volumes:
    dtc_postgres_volume_local:
```

Here, instead of using a bind mount from a local directory ("./ny_taxi_postgres_data:/var/lib/postgresql/data:rw"), we use a named volume that is actually present in a docker container, this resolves the permission errors and we can now access the database from pgAdmin.

However, here I was unable to access the table after connecting to pgAdmin, so I had to reingest the data using   

```docker
URL="https://github.com/DataTalksClub/nyc-tlc-data/releases/download/yellow/yellow_tripdata_2021-07.csv.gz"

docker run -it \
 --network=docker_sql_default \
 taxi_ingestion:v001 \
  --user=root \
  --password=root \
  --host=pgdatabase \
  --port=5432 \
  --db=ny_taxi \
  --table_name=yellow_taxi_trips \
  --url=${URL}

```
docker_sql_default is the name from system when running docker-compose up and the new host is pgdatabase as specified in our YAML file.

{{< figure src="./docker_compose_up.png"  width="100%" >}}

We can also run docker compose in detached mode using 

```docker
docker-compose up -d 
```
where we will be able to use the terminal after running it.

To shut it down, use 
```docker
docker-compose down
```