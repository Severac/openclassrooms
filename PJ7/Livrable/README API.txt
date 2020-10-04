URL of user interface (hosted on AWS) : 
https://pj7.analysons.com/

To install the UI :

Run an AWS EC2 instance (Ubuntu) and configure it  
(add streamlit listen port 8506 on IP filtering.  Configure fixed IP for the instance.  Generate key .PEM file to be able to connect via SSH)

Create a directory streamlit_OC_PJ7 and upload following files on it :
- UI_PJ7.py
- logo-chien.jpg
- model_endsave directory


Connected as ssh and run commands :

sudo apt-get update
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda
echo "PATH=$PATH:$HOME/miniconda/bin" >> ~/.bashrc
source ~/.bashrc

pip install streamlit
pip install plotly_express

sudo apt-get install tmux
tmux new -s StreamSession_PJ7
cd streamlit_OC_PJ7
streamlit run UI_PJ7.py --server.port=8506 --browser.serverPort='8506' --browser.serverAddress='pj7.analysons.com'

Exit TMUX session with : Ctrl + B  then D (streamlit will continue to run in the background after your disconnect)

To reattach to the session, in order to stop or relaunch streamlit :
tmux attach -t StreamSession_PJ7

Create a shell script to directly launch streamlit session in background (equivalent of above commands tmux new -s then streamlit reun) :
RUN_CMD.sh:
#!/bin/bash
tmux new-session -d -s StreamSession_PJ7 "streamlit run UI_PJ7.py --server.port=8506 --browser.serverPort='8506' --browser.serverAddress='pj7.analysons.com'"

Web server configuration for redirection :

sudo apt-get update
sudo apt-get upgrade

sudo apt-get install apache2

sudo a2enmod proxy && sudo a2enmod proxy_http
sudo a2enmod rewrite
sudo a2enmod proxy_wstunnel

sudo service apache2 restart

Pour logger l'url:  ajout de %V dans le LogFormat du /etc/apache2/apache2.conf

/etc/apache2/sites-available/000-default.conf (edit as root) :

<VirtualHost *:80>
    ServerName pj7.analysons.com
    RewriteEngine on
    RewriteCond %{SERVER_NAME} =pj7.analysons.com
    RewriteRule ^ https://%{SERVER_NAME}%{REQUEST_URI} [END,NE,R=permanent]
</VirtualHost>

Generate https certificate :
sudo certbot certonly --apache

<VirtualHost *:443>
   ServerName pj7.analysons.com

   RewriteEngine On
   RewriteCond %{HTTP:Upgrade} =websocket
   RewriteRule /(.*) ws://localhost:8506/$1 [P]
   RewriteCond %{HTTP:Upgrade} !=websocket
   RewriteRule /(.*) http://localhost:8506/$1 [P]
   ProxyPassReverse / http://localhost:8506

   SSLEngine On
   SSLCertificateFile /etc/letsencrypt/live/pj7.analysons.com/fullchain.pem
   SSLCertificateKeyFile /etc/letsencrypt/live/pj7.analysons.com/privkey.pem

   ErrorLog ${APACHE_LOG_DIR}/error.log
   CustomLog ${APACHE_LOG_DIR}/access.log combined
</VirtualHost>

sudo service apache2 restart

Add pj7.analysons.com on DNS (service route53 on AWS) as alias of analysons.com
