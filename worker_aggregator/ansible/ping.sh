ansible-playbook -i ./inventory/ec2.py \
      --limit "tag_type_worker_aggregator" \
      -u ubuntu \
      --private-key ~/.ssh/SoRT.pem ping.yaml -vvvv
