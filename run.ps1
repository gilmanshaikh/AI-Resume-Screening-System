name: CI-CD Pipeline

on:
  push:
    branches:
      - main

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Build Docker Image
        run: docker build -t myapp .

  deploy:
    needs: build
    runs-on: ubuntu-latest

    steps:
      - name: Deploy to AWS EC2
        uses: appleboy/ssh-action@v1.0.3
        with:
          host: ${{ secrets.EC2_HOST }}
          username: ubuntu
          key: ${{ secrets.EC2_SSH_KEY }}
          script: |
            cd app
            git pull
            docker stop myapp || true
            docker rm myapp || true
            docker build -t myapp .
            docker run -d -p 8501:8501 --name myapp myapp
