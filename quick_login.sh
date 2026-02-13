#!/bin/bash

# Quick login script to get bearer token

echo "🔐 Firebase Login"
echo "================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found"
    exit 1
fi

# Get the API key from .env
API_KEY=$(grep GOOGLE_IDENTITY_TOOL_KIT_KEY .env | cut -d '=' -f2)

if [ -z "$API_KEY" ]; then
    echo "❌ Error: GOOGLE_IDENTITY_TOOL_KIT_KEY not found in .env"
    exit 1
fi

# Ask for credentials
read -p "Email: " EMAIL
read -sp "Password: " PASSWORD
echo ""
echo ""

# Login
echo "🔄 Logging in..."
RESPONSE=$(curl -s -X POST "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=$API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"$EMAIL\",\"password\":\"$PASSWORD\",\"returnSecureToken\":true}")

# Check for error
if echo "$RESPONSE" | grep -q "error"; then
    echo "❌ Login failed!"
    echo "$RESPONSE" | grep -o '"message":"[^"]*"' | cut -d '"' -f4
    exit 1
fi

# Extract token and user info
TOKEN=$(echo "$RESPONSE" | grep -o '"idToken":"[^"]*"' | cut -d '"' -f4)
USER_ID=$(echo "$RESPONSE" | grep -o '"localId":"[^"]*"' | cut -d '"' -f4)
EMAIL_FROM_RESPONSE=$(echo "$RESPONSE" | grep -o '"email":"[^"]*"' | cut -d '"' -f4)

if [ -z "$TOKEN" ]; then
    echo "❌ Failed to extract token"
    exit 1
fi

echo "✅ Login successful!"
echo ""
echo "============================================"
echo "YOUR CREDENTIALS:"
echo "============================================"
echo "User ID:  $USER_ID"
echo "Email:    $EMAIL_FROM_RESPONSE"
echo ""
echo "Bearer Token:"
echo "$TOKEN"
echo "============================================"
echo ""

# Save to file
cat > my_token.txt << EOF
USER_ID=$USER_ID
EMAIL=$EMAIL_FROM_RESPONSE
TOKEN=$TOKEN
EOF

echo "💾 Saved to: my_token.txt"
echo ""
echo "🚀 To test the API in FastAPI docs:"
echo "   1. Go to http://localhost:8001/docs"
echo "   2. Click 'Authorize' button (top right)"
echo "   3. Paste this token: $TOKEN"
echo "   4. Click 'Authorize' then 'Close'"
echo ""
echo "🔬 Or test with curl:"
echo "curl \"http://localhost:8001/api/v1/analytics/user/$USER_ID?days=7\" \\"
echo "  -H \"Authorization: Bearer $TOKEN\""
echo ""
