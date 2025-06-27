# Media Handling Configuration

This document outlines the configuration required for the media handling functionality in the Potpie application.

## Environment Variables

The following environment variables need to be set for media handling to work properly:

### Google Cloud Storage Configuration

```bash
# Required for GCS storage
GCS_PROJECT_ID=your-gcp-project-id
GCS_BUCKET_NAME=potpie-media-attachments

# GCS authentication - one of the following:
# Option 1: Service account key file
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

# Option 2: ADC (Application Default Credentials) - for production
# This is automatically available in GCP environments
```

## Google Cloud Storage Setup

### 1. Create a GCS Bucket

```bash
# Create bucket (replace with your preferred region)
gsutil mb -p your-project-id -c STANDARD -l us-central1 gs://potpie-media-attachments

# Set bucket permissions for your service account
gsutil iam ch serviceAccount:potpie-service@your-project-id.iam.gserviceaccount.com:roles/storage.objectAdmin gs://potpie-media-attachments
```

### 2. Service Account Setup

Create a service account with the following permissions:
- Storage Object Admin (for the specific bucket)
- Storage Legacy Bucket Writer (if needed)

```bash
# Create service account
gcloud iam service-accounts create potpie-media-service \
    --description="Service account for Potpie media handling" \
    --display-name="Potpie Media Service"

# Grant necessary permissions
gcloud projects add-iam-policy-binding your-project-id \
    --member="serviceAccount:potpie-media-service@your-project-id.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

# Create and download key file
gcloud iam service-accounts keys create potpie-media-key.json \
    --iam-account=potpie-media-service@your-project-id.iam.gserviceaccount.com
```

## Media Upload Limits

The following limits are configured in the application:

- **Maximum file size**: 10 MB
- **Maximum image dimension**: 4096 pixels (width or height)
- **Supported formats**: JPEG, PNG, WebP, GIF
- **Maximum images per message**: Configurable (default: no limit in current implementation)

## API Endpoints

### Upload Image
```
POST /api/v1/media/upload
Content-Type: multipart/form-data

Parameters:
- file: Image file
- message_id (optional): Link to specific message
```

### Get Attachment Access URL
```
GET /api/v1/media/{attachment_id}/access?expiration_minutes=60
```

### Get Attachment Info
```
GET /api/v1/media/{attachment_id}/info
```

### Delete Attachment
```
DELETE /api/v1/media/{attachment_id}
```

### Get Message Attachments
```
GET /api/v1/messages/{message_id}/attachments
```

### Post Message with Images
```
POST /api/v1/conversations/{conversation_id}/message/
Content-Type: multipart/form-data

Parameters:
- content: Message text
- node_ids (optional): JSON string of node contexts
- images (optional): List of image files
```

## Security Features

1. **Access Control**: Only users with conversation access can view/download attachments
2. **Signed URLs**: Temporary URLs with configurable expiration (1-1440 minutes)
3. **File Validation**: MIME type and file content validation
4. **Size Limits**: Enforced file size and dimension limits
5. **Clean Deletion**: Orphaned attachments are cleaned up

## Database Schema

### message_attachments Table
```sql
CREATE TABLE message_attachments (
    id VARCHAR(255) PRIMARY KEY,
    message_id VARCHAR(255) NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    attachment_type VARCHAR(50) NOT NULL, -- 'image', 'video', 'audio', 'document'
    file_name VARCHAR(255) NOT NULL,
    file_size INTEGER NOT NULL,
    mime_type VARCHAR(100) NOT NULL,
    storage_path VARCHAR(500) NOT NULL,
    storage_provider VARCHAR(50) NOT NULL DEFAULT 'gcs',
    file_metadata JSONB, -- dimensions, format, processing info
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_message_attachments_message_id ON message_attachments(message_id);
```

### messages Table Updates
```sql
ALTER TABLE messages ADD COLUMN has_attachments BOOLEAN DEFAULT FALSE NOT NULL;
```

## Error Handling

The system handles various error scenarios:

1. **Upload failures**: Automatic cleanup of partially uploaded files
2. **Storage errors**: Graceful degradation with error logging
3. **Access denied**: Proper HTTP status codes and error messages
4. **File corruption**: Validation during upload and processing
5. **Quota limits**: GCS quota and rate limiting handled gracefully

## Monitoring and Logging

Key metrics to monitor:

1. **Upload success/failure rates**
2. **Storage usage and costs**
3. **API response times**
4. **Image processing performance**
5. **Access pattern analytics**

## Future Extensions

The current implementation supports:
- Images only (JPEG, PNG, WebP, GIF)
- Google Cloud Storage

Planned extensions:
- Video support
- Document support (PDF, DOCX, etc.)
- Audio support
- Multiple storage providers (S3, Azure)
- Advanced image processing (thumbnails, watermarks)
- CDN integration 