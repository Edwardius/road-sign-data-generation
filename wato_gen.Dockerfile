FROM pure/python:3.8-cuda10.2-base
WORKDIR /project

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python3 -m pip install pip --upgrade
RUN pip install \  
	Pillow \
        av \
    tqdm>=4.29.0 \
	'iopath<0.1.9,>=0.1.7' 

WORKDIR /project/wato-gen



# python3 ./main.py --config configs/ROAD/SLOWFAST_R50_ACAR_HR2O.yaml