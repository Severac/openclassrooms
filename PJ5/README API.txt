URL of user interface (hosted on AWS) : 
https://pj5.analysons.com/

To install the UI :

Run an AWS EC2 instance (Ubuntu) and configure it  
(add streamlit listen port 8504 on IP filtering.  Configure fixed IP for the instance.  Generate key .PEM file to be able to connect via SSH)

Create a directory streamlit_OC_PJ5 and upload following files on it :
- UI_PJ5.py
- functions.py
- API_model_PJ5.pickle
- UI_input_template.csv
- graph_model_ui.png
- ecommerce.png
- 

Connected as ssh and run commands :

sudo apt-get update
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda
echo "PATH=$PATH:$HOME/miniconda/bin" >> ~/.bashrc
source ~/.bashrc

pip install streamlit
pip install plotly_express

sudo apt-get install tmux
tmux new -s StreamSession_PJ5
cd streamlit_OC_PJ5
streamlit run UI_PJ5.py --server.port=8504 --browser.serverPort='8504' --browser.serverAddress='pj5.analysons.com'

Exit TMUX session with : Ctrl + B  then D (streamlit will continue to run in the background after your disconnect)

To reattach to the session, in order to stop or relaunch streamlit :
tmux attach -t StreamSession_PJ5

Web server configuration for redirection :

sudo apt-get update
sudo apt-get upgrade

sudo apt-get install apache2

sudo a2enmod proxy && sudo a2enmod proxy_http
sudo a2enmod rewrite
sudo a2enmod proxy_wstunnel

sudo service apache2 restart

Pour logger l'url:  ajout de %V dans le LogFormat du /etc/apache2/apache2.conf

/etc/apache2/sites-available/000-default.conf :

<VirtualHost *:80>
    ServerName pj5.analysons.com
    RewriteEngine on
    RewriteCond %{SERVER_NAME} =pj5.analysons.com
    RewriteRule ^ https://%{SERVER_NAME}%{REQUEST_URI} [END,NE,R=permanent]
</VirtualHost>

Generate https certificate :
sudo certbot certonly --apache

<VirtualHost *:443>
   ServerName pj5.analysons.com

   RewriteEngine On
   RewriteCond %{HTTP:Upgrade} =websocket
   RewriteRule /(.*) ws://localhost:8504/$1 [P]
   RewriteCond %{HTTP:Upgrade} !=websocket
   RewriteRule /(.*) http://localhost:8504/$1 [P]
   ProxyPassReverse / http://localhost:8504

   SSLEngine On
   SSLCertificateFile /etc/letsencrypt/live/pj5.analysons.com/fullchain.pem
   SSLCertificateKeyFile /etc/letsencrypt/live/pj5.analysons.com/privkey.pem

   ErrorLog ${APACHE_LOG_DIR}/error.log
   CustomLog ${APACHE_LOG_DIR}/access.log combined
</VirtualHost>


