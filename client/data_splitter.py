import boto3
import random
import torch
import os
import argparse
from torchvision import datasets, transforms

from fabric import Connection

def get_public_dns(region_name, tag, value):
    public_dns = []
    ec2 = boto3.resource('ec2', region_name=region_name)
    filters = [{'Name': 'tag:' + tag, 'Values': value}]
    for instance in ec2.instances.filter(Filters=filters):
        public_dns.append(instance.public_dns_name)
    return public_dns

def pick_uniform(dataset, num_label, num_instances, num_data, saved_dir):
    if os.path.exists(os.path.join(saved_dir, 'client0.pt')):
        return

    uniform_data = [[] for _ in range(num_label)]
    for i, (data, target) in enumerate(dataset):
        uniform_data[target].append((data, target))

    for i, datas in enumerate(uniform_data):
        print('label', i, len(datas))

    for k in range(num_instances):
        samping_data = []
        for i, datas in enumerate(uniform_data):
            sampling = random.choices(datas, k=num_data)
            samping_data.append(sampling)
        torch.save(samping_data, os.path.join(saved_dir, 'client' + str(k) + '.pt'))

def dataload_unittest(file_path):
    data = torch.load(file_path)
    print(len(data[0]))

def main(args):
    public_dns = get_public_dns(region_name='ap-northeast-2',
                                tag='type', value=['client'])
    if '' in public_dns:
        public_dns.remove('')
    num_instances = len(public_dns)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    dataset = datasets.MNIST('/tmp/MNIST', train=True,
                             transform=transform, download=True)
    pick_uniform(dataset,
             num_label=args.n_label,
             num_instances=num_instances,
             num_data=args.n_data,
            saved_dir=args.saved_dir)

    for k in range(num_instances):
        file_path = os.path.join(args.saved_dir, 'client' + str(k) + '.pt')
        if os.path.exists(file_path):
            with Connection(host=public_dns[k],
                            user='ubuntu',
                            connect_kwargs={"key_filename": args.key_path}
                            ) as c:
                print(public_dns[k] + ' Data Send!')
                c.put(file_path, '/tmp/data.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_label', help='number of label', default=10)
    parser.add_argument('--key_path', help='private key path')
    parser.add_argument('--saved_dir', help='saved data folder', default='./data/')
    parser.add_argument('--n_data', default=32,
                        help='number of data in one label which will send to clients')
    known_args, _ = parser.parse_known_args()
    print(known_args)
    main(args=known_args)
