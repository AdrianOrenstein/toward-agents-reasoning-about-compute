# Toward Agents That Reason About Their Computation

To replicate results:

```bash
# pull the container
docker pull adrianorenstein/fabric_dqn:latest

# run the container
docker run -it --rm --volume=$(pwd):/app/:rw -w /app --gpus all adrianorenstein/fabric_dqn:latest /bin/bash

# inside of the container, you can ask for the command line arguments
PYTHONPATH=$PWD:$PYTHONPATH python src/main.py --help

# inside of the container, a dqn experiment can be launched with default kwargs
PYTHONPATH=$PWD:$PYTHONPATH python src/main.py dqn

# inside of the container, a compute dqn (option dqn) experiment can be launched with default kwargs
PYTHONPATH=$PWD:$PYTHONPATH python src/main.py option_dqn

# optionally build the container yourself, but you should change config.yaml
make build fabric_dqn
make run fabric_dqn
```
