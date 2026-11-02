output "cluster_name" {
  description = "EKS Cluster Name"
  value       = aws_eks_cluster.genebridge.name
}

output "cluster_endpoint" {
  description = "Endpoint for EKS Kubernetes control plane API"
  value       = aws_eks_cluster.genebridge.endpoint
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = aws_eks_cluster.genebridge.vpc_config[0].cluster_security_group_id
}

output "configure_kubectl_command" {
  description = "Run this command to configure kubectl credentials"
  value       = "aws eks --region ${var.aws_region} update-kubeconfig --name ${aws_eks_cluster.genebridge.name}"
}

