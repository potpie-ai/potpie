# Ngrok Setup Guide

## What is Ngrok?

Ngrok creates secure tunnels to localhost, allowing you to expose your local server to the internet. This is useful for:
- Testing webhooks
- Sharing your local development server
- Testing mobile apps with local APIs

## Installing Ngrok

### macOS
```bash
# Using Homebrew
brew install ngrok/ngrok/ngrok

# Or download from https://ngrok.com/download
```

### Linux
```bash
# Download and install
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | \
  sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && \
  echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | \
  sudo tee /etc/apt/sources.list.d/ngrok.list && \
  sudo apt update && sudo apt install ngrok
```

### Windows
Download from https://ngrok.com/download and extract the executable.

## Setting Up Ngrok

1. **Sign up for a free account** at https://dashboard.ngrok.com/signup
2. **Get your authtoken** from https://dashboard.ngrok.com/get-started/your-authtoken
3. **Configure ngrok**:
   ```bash
   ngrok config add-authtoken YOUR_AUTH_TOKEN
   ```

## Using Ngrok Terminal UI

### Basic Usage

1. **Start your local server** (e.g., your FastAPI app on port 8001):
   ```bash
   # Your app should be running on a port, e.g., 8001
   ```

2. **Start ngrok tunnel**:
   ```bash
   ngrok http 8001
   ```

3. **Access the Terminal UI**:
   - Open your browser and go to: **http://localhost:4040**
   - This is ngrok's web-based terminal UI

### Terminal UI Features

The ngrok web UI (http://localhost:4040) provides:

- **Request Inspector**: See all HTTP requests in real-time
- **Request Details**: View headers, body, query parameters
- **Replay Requests**: Click "Replay" to resend any request
- **Webhook Testing**: Perfect for testing webhook endpoints
- **Traffic Analysis**: See request/response statistics

### Advanced Usage

#### Custom Domain (requires paid plan)
```bash
ngrok http 8001 --domain=your-custom-domain.ngrok-free.app
```

#### Static Domain (free tier available)
```bash
ngrok http 8001 --domain=static-name.ngrok-free.app
```

#### With Authentication
```bash
ngrok http 8001 --basic-auth="username:password"
```

#### Inspect Traffic in Terminal
```bash
# View requests in terminal instead of web UI
ngrok http 8001 --log=stdout
```

## For Your Webhook Testing

1. **Start your Potpie server**:
   ```bash
   ./start.sh
   # Server runs on port 8001
   ```

2. **Start ngrok tunnel**:
   ```bash
   ngrok http 8001
   ```

3. **Copy the forwarding URL** (e.g., `https://5393cace9a3e.ngrok-free.app`)

4. **Access the Terminal UI**:
   - Open http://localhost:4040
   - You'll see all incoming requests in real-time
   - Click on any request to see full details
   - Use "Replay" to resend requests

5. **Test your webhook**:
   ```bash
   # Use the script we created
   uv run python scripts/test_webhook.py https://YOUR-NGROK-URL.ngrok-free.app/api/v1/webhook/d8dc81da-6093-405c-b430-0fad2566e772
   ```

## Troubleshooting

### "ERR_NGROK_3200: The endpoint is offline"
- Make sure your local server is running
- Verify ngrok is pointing to the correct port
- Check that ngrok tunnel is active (visit http://localhost:4040)

### "ERR_NGROK_702: Account limit exceeded"
- Free tier has connection limits
- Wait a few minutes or upgrade to a paid plan

### "Tunnel not found"
- Verify the ngrok URL is correct
- Check that ngrok is still running
- Restart ngrok if needed

## Ngrok Configuration File

You can create a config file at `~/.ngrok2/ngrok.yml`:

```yaml
version: "2"
authtoken: YOUR_AUTH_TOKEN
tunnels:
  potpie:
    addr: 8001
    proto: http
```

Then start with:
```bash
ngrok start potpie
```

## Useful Commands

```bash
# List all active tunnels
ngrok api tunnels list

# Get tunnel details
ngrok api tunnels detail TUNNEL_ID

# Stop a tunnel
ngrok api tunnels stop TUNNEL_ID

# View ngrok version
ngrok version

# View help
ngrok help
```

## Tips

1. **Keep the Terminal UI open** (http://localhost:4040) to monitor all requests
2. **Use the Replay feature** to test webhooks multiple times
3. **Check the Request Inspector** to see exactly what data your webhook receives
4. **Free tier URLs change** each time you restart ngrok (unless using static domain)
5. **For production**, consider using ngrok's static domains or paid plans
