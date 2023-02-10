FROM mavencodev/minio:v.0.1

WORKDIR /app

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install -U scikit-learn

COPY . .