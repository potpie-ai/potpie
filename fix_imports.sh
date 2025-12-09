#!/bin/bash
# Fix import paths from src.* to app.*

echo "Fixing import paths in workflow code..."

# Fix imports in executions module
find app/core/executions -name "*.py" -type f -exec sed -i '' 's/from src\./from app./g' {} \;
find app/core/executions -name "*.py" -type f -exec sed -i '' 's/import src\./import app./g' {} \;

# Fix imports in nodes module
find app/core/nodes -name "*.py" -type f -exec sed -i '' 's/from src\./from app./g' {} \;
find app/core/nodes -name "*.py" -type f -exec sed -i '' 's/import src\./import app./g' {} \;

# Fix imports in celery_config
sed -i '' 's/from src\./from app./g' app/celery_config.py
sed -i '' 's/import src\./import app./g' app/celery_config.py

# Fix imports in other core files
if [ -f app/core/workflows.py ]; then
    sed -i '' 's/from src\./from app./g' app/core/workflows.py
fi

if [ -f app/core/intelligence.py ]; then
    sed -i '' 's/from src\./from app./g' app/core/intelligence.py
fi

if [ -f app/core/trigger_hashes.py ]; then
    sed -i '' 's/from src\./from app./g' app/core/trigger_hashes.py
fi

echo "✅ Import paths fixed!"

