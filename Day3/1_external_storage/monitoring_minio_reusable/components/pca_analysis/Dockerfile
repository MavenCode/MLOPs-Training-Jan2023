FROM mavencodev/minio:v.0.1

WORKDIR /app

RUN pip3 install --upgrade pip
RUN pip3 install joblib pandas numpy && pip3 install -U scikit-learn

COPY . .