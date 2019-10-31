## Worker Aggregator
```shell script
$ export AWS_ACCESS_KEY_ID=''
$ export AWS_SECRET_ACCESS_KEY=''

# init worker aggregator
chmod +x init.sh && ./init.sh

# start flask app
chmod +x worker.sh
./worker.sh [upload url(str, http://127.0.0.1/upload)]
```
