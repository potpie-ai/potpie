# Distributed Parsing Fix - Root Cause Analysis and Implementation Guide

## Date: 2025-11-17

## Executive Summary

GitBucket repository cloning was failing with exit code 128 (authentication error) when using distributed Celery workers. The root cause was a race condition in the distributed clone logic where early-finishing workers would delete the cloned repository before other workers could use it.

---

## Root Cause Analysis

### The Problem

The distributed parsing system introduced in commits `e00271a` and `a732796` added logic for workers to independently clone repositories when they didn't find them on their filesystem. This caused several critical issues:

1. **Race Condition with Reference Counting**
   - Orchestrator clones repo, initializes refcount to N workers
   - Workers on same node find repo exists, don't increment refcount
   - First worker to finish decrements refcount: N → N-1 → ... → 0
   - First finishing worker sees refcount=0, **deletes the repo**
   - Other workers still parsing suddenly lose access to files
   - Late-starting workers see repo missing, try to clone themselves → **Exit code 128**

2. **Why Exit Code 128?**
   - When a worker doesn't find the repo (because another worker deleted it), it tries to clone
   - The clone operation requires GitBucket credentials (`GITBUCKET_USERNAME`, `GITBUCKET_PASSWORD`)
   - If credentials have special characters or worker environment differs, authentication fails
   - Git returns exit code 128 for authentication/permission errors

3. **Additional Complexity Issues**
   - Path mismatches between orchestrator and workers on different nodes
   - File locking contention between concurrent workers
   - Complex refcount management prone to race conditions

### Code Location

**Problematic Code**: `app/celery/tasks/parsing_tasks.py`
- Lines 430-620: Worker-side cloning logic
- Lines 69-230: Reference counting functions
- Lines 698-720: Per-worker cleanup in finally block

---

## The Fix

### Approach: Simplify with Shared Storage

Instead of having each worker potentially clone repositories independently, use a **shared filesystem** (PersistentVolumeClaim with ReadWriteMany access mode) where:

1. Orchestrator clones repository **once** to shared storage
2. All workers access the **same files** via shared mount
3. Cleanup happens **once** in the aggregation callback after ALL workers complete

### Code Changes Made

**File**: `app/celery/tasks/parsing_tasks.py`

1. **Reverted to working version** (commit `1fea6ca`)
   - Removed ~350 lines of distributed clone complexity
   - Removed all refcount management functions
   - Removed per-worker cloning logic
   - Removed file locking mechanisms

2. **Added simple callback cleanup** (~50 lines)
   - Pass `repo_path` to aggregation callback
   - Clean up cloned repo after ALL workers finish
   - Best-effort cleanup on errors

**Diff Summary**:
```diff
# Orchestrator passes repo_path to callback
callback = aggregate_and_resolve_references.s(
    ...
    repo_path=project_path  # NEW: For cleanup
)

# Callback signature updated
def aggregate_and_resolve_references(
    ...
    repo_path: str = None  # NEW: Optional for backward compatibility
):
    import shutil  # NEW

    # After all parsing complete, cleanup
    if repo_path and os.path.exists(repo_path):
        shutil.rmtree(repo_path)
```

---

## Manual Steps Required

### 1. Set Up Shared Storage (PersistentVolumeClaim)

#### Option A: AWS EKS with EFS

```bash
# From jumphost, install EFS CSI driver
kubectl apply -k "github.com/kubernetes-sigs/aws-efs-csi-driver/deploy/kubernetes/overlays/stable/?ref=release-1.5"

# Create EFS filesystem in AWS Console or via CLI
aws efs create-file-system --creation-token momentum-parsing-storage --tags Key=Name,Value=momentum-parsing

# Get the FileSystemId (e.g., fs-12345678)
```

Create StorageClass:
```yaml
# efs-storageclass.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: efs-sc
provisioner: efs.csi.aws.com
parameters:
  provisioningMode: efs-ap
  fileSystemId: fs-12345678  # Your EFS ID
  directoryPerms: "700"
```

#### Option B: Generic NFS

If you have an NFS server:
```yaml
# nfs-storageclass.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: nfs-client
provisioner: nfs-subdir-external-provisioner
parameters:
  archiveOnDelete: "false"
```

### 2. Create PersistentVolumeClaim

```yaml
# parsing-pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: parsing-shared-storage
  namespace: <your-namespace>
spec:
  accessModes:
    - ReadWriteMany  # Critical: allows multiple pods to write
  storageClassName: efs-sc  # or nfs-client
  resources:
    requests:
      storage: 100Gi  # Adjust based on repo sizes
```

Apply:
```bash
kubectl apply -f parsing-pvc.yaml
```

### 3. Update Celery Worker Deployment

Add volume mount to your Celery worker deployment:

```yaml
# In your celery-worker deployment spec
spec:
  template:
    spec:
      containers:
        - name: celery-worker
          # ... existing config ...
          volumeMounts:
            - name: shared-projects
              mountPath: /app/projects  # This becomes PROJECT_PATH
          env:
            - name: PROJECT_PATH
              value: "/app/projects"
            # Ensure these are set:
            - name: GITBUCKET_USERNAME
              valueFrom:
                secretKeyRef:
                  name: gitbucket-credentials
                  key: username
            - name: GITBUCKET_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: gitbucket-credentials
                  key: password
      volumes:
        - name: shared-projects
          persistentVolumeClaim:
            claimName: parsing-shared-storage
```

### 4. Update API/Orchestrator Deployment

The orchestrator (API server or initial Celery worker) also needs the same mount:

```yaml
# In your API/orchestrator deployment
spec:
  template:
    spec:
      containers:
        - name: api
          volumeMounts:
            - name: shared-projects
              mountPath: /app/projects
          env:
            - name: PROJECT_PATH
              value: "/app/projects"
      volumes:
        - name: shared-projects
          persistentVolumeClaim:
            claimName: parsing-shared-storage
```

### 5. Deploy the Code Changes

```bash
# Build and push new image with the fixed code
docker build -t your-registry/momentum-server:v1.2.3 .
docker push your-registry/momentum-server:v1.2.3

# Update deployment
kubectl set image deployment/celery-worker celery-worker=your-registry/momentum-server:v1.2.3
kubectl set image deployment/api api=your-registry/momentum-server:v1.2.3

# Or apply updated manifests
kubectl apply -f k8s/
```

### 6. Verify Setup

```bash
# Check PVC is bound
kubectl get pvc parsing-shared-storage

# Check all pods have the mount
kubectl describe pod <celery-worker-pod> | grep -A5 "Mounts:"

# Verify shared storage works
kubectl exec -it <celery-worker-pod-1> -- touch /app/projects/test-file
kubectl exec -it <celery-worker-pod-2> -- ls /app/projects/test-file
kubectl exec -it <celery-worker-pod-1> -- rm /app/projects/test-file
```

---

## Testing the Fix

### 1. Basic Functionality Test

```bash
# Trigger a parsing job for a GitBucket repository
curl -X POST http://your-api/parse \
  -H "Content-Type: application/json" \
  -d '{"repo_name": "owner/repo", "branch": "main"}'
```

### 2. Monitor Logs

```bash
# Watch orchestrator logs
kubectl logs -f deployment/api | grep -E "ParsingHelper|work units"

# Watch worker logs
kubectl logs -f deployment/celery-worker | grep -E "\[Unit|Parsed|Cleaning"
```

### 3. Expected Log Flow

```
# Orchestrator
ParsingHelper: Cloning repository 'owner/repo' branch 'main' using git
ParsingHelper: Successfully cloned repository to: /app/projects/owner-repo-main-userid
Created 10 work units for 500 files

# Workers (all accessing same path)
[Unit 0] Starting: src/components (50 files)
[Unit 1] Starting: src/services (45 files)
...
[Unit 9] Complete: tests (55 files)

# Callback
Aggregating results from 10 work units
Cleaning up cloned repository at: /app/projects/owner-repo-main-userid
Successfully cleaned up repository
```

### 4. Verify No Exit Code 128 Errors

```bash
# Should NOT see these errors anymore
kubectl logs deployment/celery-worker | grep -i "exit code 128"
kubectl logs deployment/celery-worker | grep -i "authentication failed"
```

---

## Rollback Plan

If issues occur:

```bash
# Revert code to previous version
git checkout <previous-commit> -- app/celery/tasks/parsing_tasks.py

# Rebuild and redeploy
docker build -t your-registry/momentum-server:rollback .
kubectl set image deployment/celery-worker celery-worker=your-registry/momentum-server:rollback
```

---

## Performance Considerations

1. **Storage I/O**
   - EFS/NFS is slower than local SSD
   - Consider using Provisioned Throughput for EFS if parsing large repos
   - Monitor I/O latency in CloudWatch/metrics

2. **Cleanup Timing**
   - Repo cleanup happens after ALL workers finish
   - Large repos stay on disk longer
   - Monitor disk usage: `kubectl exec ... -- df -h /app/projects`

3. **Concurrent Parsing Jobs**
   - Each parsing job creates its own directory with unique user_id
   - No conflicts between concurrent jobs
   - Disk space = sum of all active parsing jobs

---

## Future Improvements (Optional)

1. **Add disk space monitoring**
   - Alert when shared storage exceeds 80%
   - Implement TTL-based cleanup for orphaned directories

2. **Add health check**
   - Verify shared storage is writable before accepting parsing jobs
   - Fail fast if PVC not mounted

3. **Consider caching**
   - Keep recently parsed repos for faster re-parsing
   - LRU eviction based on disk space

---

## Files Changed

- `app/celery/tasks/parsing_tasks.py` - Simplified distributed parsing logic
- `app/modules/parsing/graph_construction/parsing_helper.py` - No changes needed (reverted `os.path.abspath()` changes are fine to keep)

## Related Commits

- `1fea6ca` - Last working version (reverted to this)
- `e00271a` - Introduced problematic distributed clone logic
- `a732796` - Added more complexity to distributed cloning

---

## Contact

For questions about this fix, refer to:
- This document
- Git history: `git log --oneline app/celery/tasks/parsing_tasks.py`
- Celery task logs in your monitoring system
