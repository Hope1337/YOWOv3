FROM pytorch/pytorch

COPY . .

RUN pip install -r requirements.txt

WORKDIR .

CMD ["python", "train.py"]