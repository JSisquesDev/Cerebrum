FROM ubuntu:20.04

USER root

WORKDIR /app
COPY package.json ./

COPY ./ /app

# Instalación de los programas
RUN apt-get update 
RUN apt-get upgrade -y
RUN apt-get install python3.9 -y
RUN apt-get install python3-pip -y
RUN apt-get install nodejs -y
RUN apt install npm -y

RUN npm install --legacy-peer-deps

RUN python3 /app/install.py

CMD ["npm", "start"]