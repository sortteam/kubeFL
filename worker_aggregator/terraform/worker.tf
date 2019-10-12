provider "aws" {
  region = "${var.region}"
}

resource "aws_instance" "worker" {
    count = "${var.number_of_worker}"
    ami = "${lookup(var.amis, var.region)}"
    instance_type = "${var.client_instance_type}"

    subnet_id = "${var.client_subnet}"
    associate_public_ip_address = true # Instances have public, dynamic IP

    availability_zone = "${var.zone}"
    security_groups = ["${var.client_security_groups}"]
    key_name = "${var.default_keypair_name}"

    tags = {
        type = "worker"
    }

    root_block_device {
        volume_size = "${var.volume_size}"
    }
}

# Create a new load balancer
resource "aws_elb" "worker_aggregator_elb" {
    name               = "elb"
    availability_zones = ["${var.zone}"]

    instances = "${aws_instance.worker.*.id}"

    listener {
      instance_port     = 5000
      instance_protocol = "http"
      lb_port           = 80
      lb_protocol       = "http"
    }

    health_check {
      healthy_threshold   = 2
      unhealthy_threshold = 2
      timeout             = 3
      target              = "HTTP:5000/"
      interval            = 30
    }

    cross_zone_load_balancing   = true
    idle_timeout                = 100
    connection_draining         = true
    connection_draining_timeout = 100

    tags = {
        type = "worker"
    }
}
