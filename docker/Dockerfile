FROM ubuntu:latest

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y software-properties-common gcc gfortran && \
    add-apt-repository -y ppa:deadsnakes/ppa

RUN apt-get update && apt-get install -y python3.9 python3-distutils python3-pip python3-apt
RUN python3 -m pip install numpy==1.25.2
RUN python3 -m pip install PyYAML==6.0.1
RUN python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN python3 -m pip install .

COPY celltype_assignment.py . 
ENTRYPOINT ["python3", "celltype_assignment.py"]