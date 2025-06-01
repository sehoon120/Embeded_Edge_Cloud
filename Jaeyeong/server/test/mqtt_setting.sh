sudo systemctl start mosquitto
sudo netstat -tulpn | grep 1883


sudo nano /etc/mosquitto/mosquitto.conf
#bind_address 127.0.0.1 <- erase it
listener 1883  <- upload
allow_anonymous true <- upload 

sudo systemctl restart mosquitto
