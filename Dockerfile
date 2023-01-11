FROM python:3.8-slim               

# #Create directory for the code 
RUN mkdir /code
WORKDIR  /code

# # Add requirement file and install dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip 
RUN pip install -r requirements.txt

# Add src file and input data 
COPY src/ ./src 
COPY data/ ./data
COPY train.py .
COPY predict.py .  

# crate directory to save model and results 
RUN mkdir ./model ./results

# Stream output
ENV PYTHONUNBUFFERED=1

# COPY the pipeline shell script
COPY run_full_pipeline.sh . 
COPY run_full_pipeline_small.sh . 

# Run training script by default
CMD ["./run_full_pipeline.sh"]

