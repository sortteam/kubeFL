ansible-playbook -i ./inventory/ec2.py \
      --limit "tag_type_worker" \
      -u ubuntu \
      --private-key ~/.ssh/SoRT.pem worker.yaml -vvvv
