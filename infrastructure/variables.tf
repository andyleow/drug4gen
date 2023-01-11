ariable "name_prefix" {
  default = "nlp-ner"
}

variable "region" {
  default = "eu-east-2"
}

variable "vpc_cidr_block" {
  default = "10.0.0.0/16"
}

variable "subnet_cidr_block" {
  default = "10.0.1.0/24"
}

variable "destination_cidr_block" {
  default = "0.0.0.0/0"
}

variable "ingress_cidr_block" {
  type    = list(any)
  default = ["0.0.0.0/0"]
}

variable "ec2_instance_type" {
  description = "ECS cluster instance type"
  default     = "t3.large"
}

variable "max_cluster_size" {
  description = "Maximum number of instances in the cluster"
  default     = 1
}

variable "min_cluster_size" {
  description = "Minimum number of instances in the cluster"
  default     = 1
}

variable "desired_capacity" {
  description = "Desired number of instances in the cluster"
  default     = 1
}

variable "lambda_payload_filename" {
  default = "lambda-handler/target/java-events-1.0-SNAPSHOT.jar"
}