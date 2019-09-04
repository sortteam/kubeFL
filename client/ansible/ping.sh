ansible-playbook -i ./inventory/ec2.py \
      --limit "tag_type_client" \
      -u ubuntu \
      --private-key ~/.ssh/SoRT.pem ping.yaml -vvvv
