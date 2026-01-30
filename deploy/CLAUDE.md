# Deploy
**Path**: `/mnt/projects/t4d/t4dm/deploy/`

## What
Deployment configurations for Docker, Kubernetes, and Helm targeting T4DM's dual-store architecture (Neo4j + Qdrant).

## How
- **Docker**: Production Compose file and worker Dockerfile
- **Kubernetes**: Full manifest set with Kustomize overlays
- **Helm**: Helm chart for templated Kubernetes deployment

## Why
Supports multiple deployment targets from single-node Docker Compose to production Kubernetes clusters with autoscaling, ingress, and secrets management.

## Key Files
| File | Purpose |
|------|---------|
| `docker/docker-compose.prod.yml` | Production Docker Compose |
| `docker/Dockerfile.worker` | Worker container image |
| `kubernetes/deployment.yaml` | K8s deployment spec |
| `kubernetes/service.yaml` | K8s service definition |
| `kubernetes/ingress.yaml` | Ingress routing rules |
| `kubernetes/hpa.yaml` | Horizontal pod autoscaler |
| `kubernetes/secrets.yaml` | Secrets for Neo4j/Qdrant credentials |
| `kubernetes/configmap.yaml` | Application configuration |
| `kubernetes/storage.yaml` | Persistent volume claims |
| `kubernetes/kustomization.yaml` | Kustomize base config |
| `kubernetes/overlays/` | Environment-specific overrides |
| `helm/world-weaver/` | Helm chart |

## Data Flow
```
Helm values / Kustomize overlays → K8s manifests → Cluster
Docker Compose → Local/staging deployment
```

## Integration Points
- **CI/CD**: Build and deploy pipelines consume these configs
- **Scripts**: `scripts/migrate.sh` runs pre-deployment
- **Monitoring**: HPA and health probes configured in manifests
