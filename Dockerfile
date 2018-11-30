# As per instructions: https://mybinder.readthedocs.io/en/latest/tutorials/dockerfile.html#preparing-your-dockerfile
FROM jupyter/datascience-notebook:1145fb1198b2
ARG NB_USER
ARG NB_UID=1000
ENV HOME /home/${NB_USER}
ENV USER ${NB_USER}
USER ${NB_USER}

# COPY doesn't handle substitutions so forced to assume that NB_UID will always be 1000
COPY --chown=1000 . ${HOME}
WORKDIR ${HOME}

RUN pip install -r requirements.txt
RUN Rscript -e "options(unzip = 'internal'); devtools::install_github('MindFoundry/optaas-r-client@1.3.5')"

CMD ["jupyter", "notebook", "--ip", "0.0.0.0"]