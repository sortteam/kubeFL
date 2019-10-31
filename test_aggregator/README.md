## Test Aggregator
### init
- 메모리 swapping 설정
- flask boto3 awscli 설치
- python3-matplotlib, torch 설치
```shell script
$ export AWS_ACCESS_KEY_ID=''
$ export AWS_SECRET_ACCESS_KEY=''
export LC_ALL="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"

wget https://raw.githubusercontent.com/sortteam/kubeFL/master/test_aggregator/init.sh
chmod +x init.sh && ./init.sh
```

### Logger File
- [loss.txt](/tmp/loss.txt)
```text

```

### Evalute Acc
```shell script
python evaluate.py --max_len 10
```
