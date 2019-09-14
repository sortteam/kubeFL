## Client
```shell
$ terraform -v
Terraform v0.12.7
$ ansible --version
ansible 2.8.4
```



ec2 인스턴스 실행 및 Ansible ping command을 위해 `private.key`를 `~/.ssh`에 넣습니다.

```shell
$ export AWS_ACCESS_KEY_ID=''
$ export AWS_SECRET_ACCESS_KEY=''

$ mv client/config ~/.ssh/config
$ cd client/terraform
$ terraform init && terraform apply
$ cd ../ansible
$ chmod +x ping.sh && ./ping.sh
```



## Client init

- apt 업데이트와 pip3 설치
- ubuntu swap memory 1G로 제한
- pytorch=1.1.0 설치

```shell
$ cd ../ansible
$ chmod +x init.sh && ./init.sh
```



## Data Splitter

데이터를 레이블 별로 uniform하기 나눈 후 각 Client에 scp로 전송해줍니다. 전송된 파일은 `/tmp/data.pt`에 직렬화 됩니다.

- `n_label` : 데이터 레이블 갯수를 나타냅니다.
- `key_path` : private ssh key 절대경로 입니다.
- `saved_dir` : splitted된 데이터가 저장될 위치입니다.
- `n_data` : 각 client가 한 레이블당 갖고 있는 데이터의 갯수입니다. 

```shell
$ python data_splitter.py \
		--n_label 10 \
		--key_path '/home/ssh.pem' \
		--saved_dir './data/' \
		--n_data 16
```

## Client Training
Client를 Training 시키기 전에 아래 3가지가 끝나야합니다.
1. Client Provisioning
2. Client init
3. Data Splitting
그 후에 다음과 같은 명령어로 ansible에서 train.py 스크립트를 각 Client에 실행합니다.
```shell
$ chmod +x train.sh
$ ./train.sh [round number(int)]
```

## Client Restart

```shell
$ cd ../ansible
$ chmod +x restart.sh && ./restart.sh
```

