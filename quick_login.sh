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
API_KEY=$(grep GOOGLE_IDENTITY_TOOL_KIT_KEY .env | cut -d '=' -f2-)

if [ -z "$API_KEY" ]; then
    echo "❌ Error: GOOGLE_IDENTITY_TOOL_KIT_KEY not found in .env"
    exit 1
fi

# Ask for credentials
read -p "Email: " EMAIL
read -sp "Password: " PASSWORD
echo ""
echo ""

# Build JSON body safely (handles special chars in email/password)
if ! command -v jq &> /dev/null; then
    echo "❌ Error: jq is required. Install with: brew install jq"
    exit 1
fi
BODY=$(jq -n --arg email "$EMAIL" --arg password "$PASSWORD" '{email:$email,password:$password,returnSecureToken:true}')

# Login
echo "🔄 Logging in..."
RESPONSE=$(curl -s -X POST "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=$API_KEY" \
  -H "Content-Type: application/json" \
  -d "$BODY")

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
echo "User ID:  $USER_ID"
echo "Email:    $EMAIL_FROM_RESPONSE"
echo "============================================"
echo ""

# Save to file (no token printed to stdout)
cat > my_token.txt << EOF
USER_ID=$USER_ID
EMAIL=$EMAIL_FROM_RESPONSE
TOKEN=$TOKEN
EOF
chmod 600 my_token.txt

echo "💾 Credentials saved to: my_token.txt (token not shown for security)"
echo ""
echo "🚀 To test the API:"
echo "   1. FastAPI docs: http://localhost:8001/docs → Authorize → paste token from my_token.txt"
echo "   2. Curl example (use token from file):"
echo "      TOKEN=\$(grep TOKEN= my_token.txt | cut -d= -f2-)"
echo "      curl \"http://localhost:8001/api/v1/analytics/summary?start_date=2026-01-01&end_date=2026-02-12\" \\"
echo "        -H \"Authorization: Bearer \$TOKEN\""
echo ""
