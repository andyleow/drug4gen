terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.16"
    }
  }

  backend "s3" {
    bucket = "dev-terraform-state-bucket"
    key = "test_service.dev"
  }

  required_version = ">= 1.2.0"
}

provider "aws" {
  region  = var.region
}

# Security group 
resource "aws_security_group" "ecs_sg" {
    vpc_id      = aws_vpc.vpc.id

    ingress {
        from_port       = 22
        to_port         = 22
        protocol        = "tcp"
        cidr_blocks     = ["0.0.0.0/0"]
    }

    ingress {
        from_port       = 443
        to_port         = 443
        protocol        = "tcp"
        cidr_blocks     = ["0.0.0.0/0"]
    }

    egress {
        from_port       = 0
        to_port         = 65535
        protocol        = "tcp"
        cidr_blocks     = ["0.0.0.0/0"]
    }
}

# Create autoscaling group 
resource "aws_launch_configuration" "ecs_launch_config" {
    image_id             = "ami-094d4d00fd7462815"
    iam_instance_profile = aws_iam_instance_profile.ecs_agent.name
    security_groups      = [aws_security_group.ecs_sg.id]
    user_data            = "#!/bin/bash\necho ECS_CLUSTER=my-cluster >> /etc/ecs/ecs.config"
    instance_type        = var.ec2_instance_type
}

resource "aws_autoscaling_group" "failure_analysis_ecs_asg" {
    name                      = "asg"
    vpc_zone_identifier       = [aws_subnet.pub_subnet.id]
    launch_configuration      = aws_launch_configuration.ecs_launch_config.name

    desired_capacity          = var.desired_capacity
    min_size                  = var.min_cluster_size
    max_size                  = var.min_cluster_size
    health_check_grace_period = 300
    health_check_type         = "EC2"
}


# S3 to store the data
resource "aws_s3_bucket" "bucket" {
  acl           = "private"
  force_destroy = true
}

# ECR to store the image artifact
resource "aws_ecr_repository" "dev_ecr_repo" {
    name = "dev-test_service_ecr_repo"
}

# ECS cluster
resource "aws_ecs_cluster" "ecs_cluster" {
    name  = "my-cluster"
}

# Bind the cluster with task
resource "aws_ecs_service" "worker" {
  name            = "worker"
  cluster         = aws_ecs_cluster.ecs_cluster.id
  task_definition = aws_ecs_task_definition.task_definition.arn
  desired_count   = 2
}