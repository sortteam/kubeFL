#!/bin/bash
round=${round}
server=${server}
epoch=${epoch}
model=${model}

while [ $# -gt 0 ]; do
   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        # echo $1 $2 // Optional to see the parameter:value result
   fi
  shift
done

if [ -z $round ]; then
  echo 'round is not set'
  exit 1
fi

if [ -z $server ]; then
  echo 'server ip is not set'
  exit 1
fi

if [ -z $epoch ]; then
  echo '$epoch is not set'
  exit 1
fi

echo 'Round' $round
echo 'Server Ip' $server
echo 'Epoch' $epoch
ansible-playbook -i ./inventory/ec2.py \
      --limit "tag_type_client" \
      -u ubuntu \
      --private-key ~/.ssh/SoRT.pem train.yaml \
      --extra-vars "round=$round server=$server epoch=$epoch web_model=$model"
