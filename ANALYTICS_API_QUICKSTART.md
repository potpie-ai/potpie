# Analytics API - Quick Start Guide

## ✅ What's Been Built

A complete analytics API that queries Logfire data and returns JSON suitable for graphing. The API uses Bearer auth; the user is derived from the token. Endpoints:

- **GET `/api/v1/analytics/tokens-by-day`** - Total tokens per day, grouped by project (`start_date`, `end_date`)
- **GET `/api/v1/analytics/summary`** - Aggregated analytics (costs, runs, conversations) (`start_date`, `end_date`)
- **GET `/api/v1/analytics/raw`** - Raw span data for debugging (`start_date`, `end_date`, `limit`)
- **GET `/api/v1/analytics/debug/raw`** - Raw Logfire response (debug; only in local/development)

## 🚀 How to Test It

### Step 1: Make Sure Your Server is Running

Start your FastAPI server if it's not already running:

```bash
# From the project root
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8001

# Or using uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

### Step 2: Get Your Auth Token

You need a valid Firebase authentication token. The API derives the user from the token.

**To get your auth token**, you can:
- Use your frontend login flow and copy the token
- Run `python get_token.py --email YOUR_EMAIL --password YOUR_PASSWORD` (saves to `my_credentials.txt`)
- Or check your browser's localStorage/cookies after logging in

### Step 3: Run the Test Script

Using the provided test script:

```bash
# Option 1: Command line arguments
python test_analytics_api.py \
  --token "your_auth_token_here" \
  --start-date 2026-01-01 \
  --end-date 2026-02-12

# Option 2: Environment variables (token from file)
export TEST_AUTH_TOKEN=$(grep AUTH_TOKEN my_credentials.txt | cut -d= -f2-)
python test_analytics_api.py
```

The test script will:
- Query both endpoints
- Display the results
- Save the full response to `analytics_response.json`

### Step 4: Test with cURL

Or test directly with cURL (user is derived from token):

```bash
# Token from file or env
AUTH_TOKEN="your_auth_token"

# Test the summary endpoint (date range)
curl -X GET "http://localhost:8001/api/v1/analytics/summary?start_date=2026-01-01&end_date=2026-02-12" \
  -H "Authorization: Bearer ${AUTH_TOKEN}" \
  | jq .

# Test the tokens-by-day endpoint
curl -X GET "http://localhost:8001/api/v1/analytics/tokens-by-day?start_date=2026-01-01&end_date=2026-02-12" \
  -H "Authorization: Bearer ${AUTH_TOKEN}" \
  | jq .

# Test the raw spans endpoint
curl -X GET "http://localhost:8001/api/v1/analytics/raw?start_date=2026-01-01&end_date=2026-02-12&limit=10" \
  -H "Authorization: Bearer ${AUTH_TOKEN}" \
  | jq .
```

### Step 5: Check the API Documentation

FastAPI automatically generates interactive docs:

1. Open your browser to: `http://localhost:8001/docs`
2. Find the "Analytics" section
3. Try out the endpoints interactively

## 📊 Expected Response Format

### Aggregated Analytics Response

```json
{
  "user_id": "user_123",
  "period": {
    "start": "2026-01-13T00:00:00+00:00",
    "end": "2026-02-12T00:00:00+00:00",
    "days": 30
  },
  "summary": {
    "total_cost": 12.45,
    "total_llm_calls": 156,
    "avg_duration_ms": 2340.5,
    "success_rate": 0.94
  },
  "daily_costs": [
    {"date": "2026-01-13", "cost": 0.42, "run_count": 5},
    {"date": "2026-01-14", "cost": 0.38, "run_count": 4}
  ],
  "agent_runs_by_outcome": {
    "success": 147,
    "error": 9
  },
  "conversation_stats": [
    {"date": "2026-01-13", "count": 12, "avg_messages": 3.5}
  ]
}
```

## 🎨 Using the Data for Graphs

### Daily Costs Line Chart

```javascript
// Chart.js example
const chartData = {
  labels: analytics.daily_costs.map(d => d.date),
  datasets: [{
    label: 'Daily LLM Cost',
    data: analytics.daily_costs.map(d => d.cost),
    borderColor: 'rgb(75, 192, 192)',
    tension: 0.1
  }]
};
```

### Agent Runs Pie Chart

```javascript
const pieData = {
  labels: Object.keys(analytics.agent_runs_by_outcome),
  datasets: [{
    data: Object.values(analytics.agent_runs_by_outcome),
    backgroundColor: ['#36A2EB', '#FF6384', '#FFCE56']
  }]
};
```

### Summary Cards

```jsx
// React example
<div className="stats">
  <StatCard 
    title="Total Cost"
    value={`$${analytics.summary.total_cost.toFixed(2)}`}
  />
  <StatCard 
    title="Agent Runs"
    value={analytics.summary.total_agent_runs}
  />
  <StatCard 
    title="Success Rate"
    value={`${(analytics.summary.success_rate * 100).toFixed(1)}%`}
  />
  <StatCard 
    title="Avg Duration"
    value={`${analytics.summary.avg_duration_ms.toFixed(0)}ms`}
  />
</div>
```

## 🐛 Troubleshooting

### "LOGFIRE_READ_TOKEN not configured"

**Fix:** Check that your `.env` file has:
```
LOGFIRE_READ_TOKEN=your_actual_token_here
```
Then restart your server.

### 401 Unauthorized

**Fix:** Your auth token might be expired or invalid. Get a fresh token.

### 500 Internal Server Error

**Check:**
1. Server logs for detailed error messages
2. That the Logfire read token is valid
3. That your Logfire project has data

### Empty arrays in response

**This is normal if:**
- The user has no activity in the time period
- The user_id doesn't match what's in Logfire
- You haven't generated any traces yet

**To verify:**
- Check `/api/v1/analytics/raw` to see if ANY spans exist for your token
- Look at your Logfire UI to confirm data is being sent

## 📁 Files Created

```
app/modules/analytics/
├── __init__.py                  # Module initialization
├── schemas.py                   # Pydantic models
├── analytics_service.py         # Business logic & Logfire queries
├── analytics_router.py          # FastAPI endpoints
└── README.md                    # Detailed documentation

test_analytics_api.py            # Test script
ANALYTICS_API_QUICKSTART.md      # This file
```

## 🔗 Next Steps for UI

Once you verify the API works, you can build a UI to visualize this data:

1. **Choose a charting library:**
   - Chart.js (simple, popular)
   - Recharts (React-friendly)
   - D3.js (powerful, complex)
   - ApexCharts (modern, feature-rich)

2. **Create a dashboard page:**
   - Summary cards at the top
   - Line chart for daily costs
   - Bar chart for agent runs by outcome
   - Table for conversation stats

3. **Add filters:**
   - Date range selector
   - Project/agent filters
   - Export to CSV button

4. **Polish:**
   - Loading states
   - Error handling
   - Refresh button
   - Auto-refresh every 5 minutes

## 📚 Additional Resources

- [Full Analytics API Documentation](app/modules/analytics/README.md)
- [Logfire Query API Docs](https://logfire.pydantic.dev/docs/how-to-guides/query-api/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)

---

**Need Help?** Check the logs at `app.log` or run the test script with verbose output.
