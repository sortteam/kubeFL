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
- pytorch=1.1.0과 torchvision=0.3.0 설치

```shell
$ cd ../ansible
$ chmod +x init.sh && ./init.sh
```



## Client Restart

```shell
$ cd ../ansible
$ chmod +x restart.sh && ./restart.sh
```

