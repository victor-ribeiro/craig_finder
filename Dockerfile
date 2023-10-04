FROM continuumio/miniconda3

WORKDIR /app
COPY environment.yml /app/
COPY config.sh /app/

RUN echo $(uname -a)
RUN conda update -n base -c defaults conda
RUN conda env create -f environment.yml
RUN conda init

CMD [ "config.sh" ]