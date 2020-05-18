URL of user interface (hosted on AWS) : 
http://3.20.50.249:8501/

To install the UI :

Run an AWS EC2 instance (Ubuntu) and configure it  
(add streamlit listen port 8501 on IP filtering.  Configure fixed IP for the instance.  Generate key .PEM file to be able to connect via SSH)

Create a directory streamlit_OC_PJ4 and upload following files on it :
- UI.py
- functions.py
- API_model_PJ4.pickle
- plane_image.png

Connected as ssh and run commands :

sudo apt-get update
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda
echo "PATH=$PATH:$HOME/miniconda/bin" >> ~/.bashrc
source ~/.bashrc

pip install streamlit
pip install plotly_express

sudo apt-get install tmux
tmux new -s StreamSession
cd streamlit_OC_PJ4
streamlit run UI.py --server.port=8501 --browser.serverPort='8501' --browser.serverAddress='pj4.analysons.com'

Exit TMUX session with : Ctrl + B  then D (streamlit will continue to run in the background after your disconnect)

To reattach to the session, in order to stop or relaunch streamlit :
tmux attach -t StreamSession

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
   ServerName pj4.analysons.com

   RewriteEngine On
   RewriteCond %{HTTP:Upgrade} =websocket
   RewriteRule /(.*) ws://localhost:8501/$1 [P]
   RewriteCond %{HTTP:Upgrade} !=websocket
   RewriteRule /(.*) http://localhost:8501/$1 [P]
   ProxyPassReverse / http://localhost:8501

   ErrorLog ${APACHE_LOG_DIR}/error.log
   CustomLog ${APACHE_LOG_DIR}/access.log combined
</VirtualHost>





Note :

Values to have positive delay prediction:

ORIGIN                                OAK
CRS_DEP_TIME                         1450
MONTH                                   3
DAY_OF_MONTH                            8
DAY_OF_WEEK                             2
UNIQUE_CARRIER                         WN
DEST                                  SAN
CRS_ARR_TIME                         1610
DISTANCE                              446
CRS_ELAPSED_TIME                       80
ARR_DELAY                               9
DEP_DELAY                              13
TAXI_OUT                                8
TAIL_NUM                           N272WN
NBFLIGHTS_FORDAY_FORAIRPORT           132
NBFLIGHTS_FORDAYHOUR_FORAIRPORT         8
