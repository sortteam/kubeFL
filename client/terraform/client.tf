provider "aws" {
  region = "${var.region}"
}

resource "aws_instance" "client" {
    count = "${var.number_of_worker}"
    ami = "${lookup(var.amis, var.region)}"
    instance_type = "${var.client_instance_type}"

    subnet_id = "${var.client_subnet}"
    associate_public_ip_address = true # Instances have public, dynamic IP

    availability_zone = "${var.zone}"
    security_groups = ["${var.client_security_groups}"]
    key_name = "${var.default_keypair_name}"

    tags = {
        type = "client"
    }
}
