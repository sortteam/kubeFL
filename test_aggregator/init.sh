sudo swapoff -a
sudo dd if=/dev/zero of=/swapfile bs=1G count=4
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
sudo swapon -s
sudo apt update
sudo apt install -y python3 python3-pip
pip3 install torchvision torch
pip3 install flask boto3 awscli
sudo apt install python3-matplotlib -y
wget https://raw.githubusercontent.com/sortteam/kubeFL/master/test_aggregator/test_flask.py
python3 test_flask.py
