# Setup Guide: Running the Project on NVIDIA Jetson Orin Nano
Follow these steps carefully to deploy the local AI assistant and the web interface on your Jetson hardware.

## 1. Clone the RepositoryFirst, download the project files from GitHub:
- git clone (https://github.com/KamikotoBaka/liva-jetson-ai.git)
- cd /liva-jetson-ai

## 2. Download the LLM ModelsThe system relies on local LLMs. 
For the Jetson Orin Nano, you must use GGUF format models from Hugging Face.

Place the models in to this folder: /project/models/.

⚠️ Warning: The default configuration is optimized for Qwen 2.5. If you decide to use a different model, you must update the model file names in your docker-compose.yml accordingly.

## 3. Build the Docker ImagesCompile the required Dockerfiles using the following command.
- docker compose build

⚠️ Warning: Building the containers on the Jetson Nano architecture can take up to 60 minutes. Please be patient and do not interrupt the process.

## 4. Deploy Nginx Proxy Manager (NPM)To secure the web interface with HTTPS, we need to deploy the Nginx Proxy Manager first.
Create a docker-compose.yml for NPM in a separate directory:
- mkdir -p ~/npm && cd ~/npm
- nano docker-compose.yml
Paste the official NPM stack configuration into the file and save it.
Start the NPM container:
- docker compose up -d

## 5. Generate Self-Signed SSL Certificates
Since modern browsers strictly block unsecured connections and standard IP-based SSL, you must generate a certificate that includes a Subject Alternative Name (SAN) for your Jetson's IP address.
Run this command on your Jetson (replace YOUR_JETSON_IP with your actual local IP, e.g., 192.168.0.179):
  -openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout privkey.pem \
  -out fullchain.pem \
  -subj "/CN=YOUR_JETSON_IP" \
  -addext "subjectAltName = IP:YOUR_JETSON_IP"

## 6. Configure Nginx Proxy Manager
Open your browser and navigate to the NPM admin dashboard at http://<YOUR_JETSON_IP>:81. 

- Log in (or create your admin account on first start). 

- Upload the SSL Keys:
Go to SSL Certificates -> Add SSL Certificate -> Custom.Name it (e.g., "Jetson Local SSL") and upload your newly generated privkey.pem and fullchain.pem files.

- Create the Proxy Host:
Navigate to Hosts -> Proxy Hosts -> Add Proxy Host.Set the following 

configurations:
Field                   Value 
Domain Names:           jetson-YOURNAME.local (and your Jetson IP)
Scheme                  http 
Forward Hostname / IP   172.17.0.1 (Docker Gateway IP to reach the host)
Forward Port            5173 (React/Vite default port)
Access List             Public
Websockets              SupportON 🟢 (Required for Vite Hot-Reloading)

In the SSL Tab:
   Select your uploaded custom certificate.
   Turn Force SSL -> ON 🟢.
   

## 7. Start the Core Application Stack
Now, navigate back to your main project folder to launch the AI components and the React frontend.
- cd /liva-jetson-ai/project
- docker compose up -d
💡 Developer Note for React (Vite): Ensure your vite.config.js has host: true enabled and contains allowedHosts: ['jetson-YOURNAME.local'] to allow traffic coming through the Nginx Reverse Proxy.

## 8. Access the ApplicationYour secure, privacy-friendly AI user interface is now live! 
Open your browser and visit:👉 https://jetson-YOURNAME.local (or https://<YOUR_JETSON_IP>)(Note: Since this uses a self-signed certificate, your browser will show a warning. Click "Advanced" and "Proceed" to continue safely.)


