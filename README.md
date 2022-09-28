# d2m

Data to model - d2m.

This is a project using [DVC](https://dvc.org/) for setting up a flexible and
robust pipeline for machine learning experiments.


## Usage

Tested on:

- Linux
- macOS
- Windows with WSL 2


1. Clone/download this repository.
2. Place your datafiles (csv) in a folder with the name of your dataset (`DATASET`) inside `assets/data/raw/`, so the path to the files is `assets/data/raw/[DATASET]/`.
3. Update `params.yaml` with the name of your dataset (`DATASET`), the target variable, and other configuration parameters.
4. Build Docker container:

```
docker build -t d2m -f Dockerfile .
```

5. Run an interactive session with a Docker volume:

```
docker run -it -v [HOST DIR]:/usr/d2m/assets d2m bash
```

6. In another terminal, copy `params.yaml` from the host to the container (find `CONTAINER_NAME` by running `docker ps`):

```
docker cp params.yaml  [CONTAINER_NAME]:/usr/d2m/params.yaml
```

7. Inside the interactive session in the container, run:

```
dvc repro
```



## Docker notes

### Building the Docker container

```
docker build -t d2m -f Dockerfile .
```

### Running the Docker container


Running an interactive session with a Docker volume:

```
docker run -it -v [HOST DIR]:/usr/d2m/assets d2m bash
```

Running the Docker container with port forwarding in order to access the API (NB: NOT WORKING WITH THE CURRENT DOCKER IMAGE):

```
docker run -p 5000:5000 -it -v [HOST DIR]:/usr/d2m/assets d2m
```

