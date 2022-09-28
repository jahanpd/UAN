FROM ubuntu:latest

# Grab requirements.txt.
COPY . .
# ADD ./requirements.txt /tmp/requirements.txt

# Install dependencies
RUN apt update
RUN apt-get install -y python3.8 python3-pip
RUN python3 -V
RUN pip3 --version
RUN pip3 install -r requirements.txt
RUN pip3 install torch==1.7.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install nflows
RUN pip3 install adabelief-pytorch

# Add our code
ADD . .
WORKDIR .


CMD gunicorn --bind 0.0.0.0:$PORT CardiacML