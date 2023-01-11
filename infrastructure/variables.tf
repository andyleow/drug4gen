ariable "name_prefix" {
  default = "dev"
}

variable "region" {
  default = "eu-east-2"
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
