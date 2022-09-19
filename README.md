# d2m

Data to model - d2m.

This is a project using [DVC](https://dvc.org/) for setting up a flexible and
robust pipeline for machine learning experiments.


## Docker

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

