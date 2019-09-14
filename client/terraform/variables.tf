# Networking setup
variable "keypair_name" {
  default = "SoRT"
}

variable "region" {
  default = "ap-northeast-2"
}

variable zone {
  default = "ap-northeast-2a"
}

variable "number_of_worker" {
  default = 10
}

variable client_instance_type {
  default = "t2.small"
}

variable "volume_size" {
  default = 15
}

variable client_subnet {
  default = "subnet-35d73e5e"
}

variable "client_security_groups" {
  default = "sg-04f233861af814f57"
}

variable "default_keypair_name" {
  default = "SoRT"
}

# Instances Setup
variable amis {
  description = "Default AMIs to use for nodes depending on the region"
  type = "map"
  default = {
    ap-northeast-2 = "ami-067c32f3d5b9ace91"
    ap-northeast-1 = "ami-0567c164"
    ap-southeast-1 = "ami-a1288ec2"
    cn-north-1 = "ami-d9f226b4"
    eu-central-1 = "ami-8504fdea"
    eu-west-1 = "ami-0d77397e"
    sa-east-1 = "ami-e93da085"
    us-east-1 = "ami-40d28157"
    us-west-1 = "ami-6e165d0e"
    us-west-2 = "ami-a9d276c9"
  }
}
