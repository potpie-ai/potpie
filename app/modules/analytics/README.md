# Analytics API Module

This module provides API endpoints to query Logfire analytics data for users, including LLM costs, agent execution metrics, and conversation statistics.

## Overview

The Analytics API connects to Logfire using the Query API to retrieve the last 30 days (configurable) of user activity data. The data is aggregated and formatted for easy graphing and visualization.

## Setup

### 1. Get Logfire Read Token

You need a read token to query Logfire data:

**Via Web UI:**
1. Go to [logfire.pydantic.dev](https://logfire.pydantic.dev)
2. Select your project (e.g., "potpie-ai")
3. Click ⚙️ Settings → Read tokens
4. Click "Create read token"
5. Copy the token

**Via CLI:**
```bash
logfire read-tokens --project <organization>/potpie-ai create
```

### 2. Add Token to Environment

Add the token to your `.env` file:

```bash
LOGFIRE_READ_TOKEN=your_token_here
```

### 3. Restart the API Server

If the server is already running, restart it to pick up the new environment variable:

```bash
# Stop the current server (Ctrl+C) and restart
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8001
```

## API Endpoints

### GET `/api/v1/analytics/user/{user_id}`

Get aggregated analytics data for a user.

**Parameters:**
- `user_id` (path): The user ID to query
- `days` (query, optional): Number of days to look back (default: 30, max: 90)

**Authentication:** Required (Bearer token)

**Response:**

```json
{
  "user_id": "user_123",
  "period": {
    "start": "2026-01-13T00:00:00Z",
    "end": "2026-02-12T00:00:00Z",
    "days": 30
  },
  "summary": {
    "total_cost": 12.45,
    "total_agent_runs": 156,
    "avg_duration_ms": 2340.5,
    "success_rate": 0.94
  },
  "daily_costs": [
    {
      "date": "2026-01-13",
      "cost": 0.42,
      "run_count": 5
    },
    {
      "date": "2026-01-14",
      "cost": 0.38,
      "run_count": 4
    }
  ],
  "agent_runs_by_outcome": {
    "success": 147,
    "error": 9
  },
  "conversation_stats": [
    {
      "date": "2026-01-13",
      "count": 12,
      "avg_messages": 3.5
    }
  ]
}
```

### GET `/api/v1/analytics/user/{user_id}/raw`

Get raw Logfire span data (useful for debugging or custom analysis).

**Parameters:**
- `user_id` (path): The user ID to query
- `days` (query, optional): Number of days to look back (default: 7, max: 30)
- `limit` (query, optional): Max number of spans (default: 100, max: 1000)

**Authentication:** Required (Bearer token)

**Response:** Array of raw span objects

## Example Usage

### Using cURL

```bash
# Get analytics for last 30 days
curl -X GET "http://localhost:8001/api/v1/analytics/user/YOUR_USER_ID?days=30" \
  -H "Authorization: Bearer YOUR_AUTH_TOKEN"

# Get analytics for last 7 days
curl -X GET "http://localhost:8001/api/v1/analytics/user/YOUR_USER_ID?days=7" \
  -H "Authorization: Bearer YOUR_AUTH_TOKEN"

# Get raw spans
curl -X GET "http://localhost:8001/api/v1/analytics/user/YOUR_USER_ID/raw?days=7&limit=100" \
  -H "Authorization: Bearer YOUR_AUTH_TOKEN"
```

### Using Python Requests

```python
import requests

BASE_URL = "http://localhost:8001"
USER_ID = "your_user_id"
AUTH_TOKEN = "your_auth_token"

headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}

# Get analytics
response = requests.get(
    f"{BASE_URL}/api/v1/analytics/user/{USER_ID}",
    params={"days": 30},
    headers=headers
)

if response.status_code == 200:
    data = response.json()
    print(f"Total cost: ${data['summary']['total_cost']}")
    print(f"Total runs: {data['summary']['total_agent_runs']}")
    print(f"Success rate: {data['summary']['success_rate']*100}%")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

### Using JavaScript/TypeScript

```typescript
const BASE_URL = "http://localhost:8001";
const USER_ID = "your_user_id";
const AUTH_TOKEN = "your_auth_token";

async function getAnalytics(userId: string, days: number = 30) {
  const response = await fetch(
    `${BASE_URL}/api/v1/analytics/user/${userId}?days=${days}`,
    {
      headers: {
        Authorization: `Bearer ${AUTH_TOKEN}`,
      },
    }
  );

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
}

// Usage
getAnalytics(USER_ID, 30)
  .then((data) => {
    console.log("Total cost:", data.summary.total_cost);
    console.log("Daily costs:", data.daily_costs);
  })
  .catch((error) => console.error("Error:", error));
```

## Data Structure

### What's Tracked

The API queries the following data from Logfire:

1. **LLM Costs** - From `agent_run_usage` spans:
   - `actual_cost`: Cost per agent run
   - `outcome`: Success/failure status
   - `usage_count`: Number of API calls

2. **Agent Executions** - From spans matching patterns:
   - Agent ID, Run ID
   - Duration in milliseconds
   - Span names and timestamps

3. **Conversations** - From spans with conversation_id:
   - Conversation counts by date
   - Message activity

### Graphing the Data

The response format is optimized for common charting libraries:

**For daily costs (line chart):**
```javascript
const chartData = analytics.daily_costs.map(d => ({
  x: d.date,
  y: d.cost
}));
```

**For agent runs by outcome (pie/bar chart):**
```javascript
const outcomesData = Object.entries(analytics.agent_runs_by_outcome).map(([outcome, count]) => ({
  label: outcome,
  value: count
}));
```

## Troubleshooting

### Error: "LOGFIRE_READ_TOKEN not configured"

**Solution:** Make sure you've added the read token to your `.env` file and restarted the server.

### Error: 403 Forbidden

**Solution:** Make sure you're providing a valid authentication token in the `Authorization` header.

### No data returned / Empty arrays

**Possible causes:**
1. The user has no activity in the specified time period
2. The user_id doesn't match what's being logged to Logfire
3. The Logfire project doesn't have data yet

**Debug:**
- Check the `/raw` endpoint to see if any spans exist
- Verify the user_id is correct in your Logfire UI
- Check application logs for errors

### Query timeout or slow response

**Solutions:**
- Reduce the `days` parameter (query less data)
- Check Logfire query limits in your plan
- Consider adding caching for frequently accessed data

## Architecture

```
┌─────────────────┐
│  FastAPI Route  │
│  (auth check)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ AnalyticsService│
│  - get_user_    │
│    analytics()  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LogfireQuery    │
│    Client       │
│  (SQL queries)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Logfire API   │
│ (pydantic.dev)  │
└─────────────────┘
```

## Future Enhancements

Potential improvements for the analytics API:

1. **Caching**: Add Redis caching for frequently accessed data
2. **Real-time**: WebSocket endpoint for live updates
3. **Custom Metrics**: Allow filtering by project_id, agent_id, etc.
4. **Export**: Add CSV/Excel export functionality
5. **Alerts**: Integration with alerting when costs exceed thresholds
6. **Aggregations**: Pre-computed daily/weekly/monthly rollups
7. **Comparisons**: Compare periods (this week vs last week)

## References

- [Logfire Query API Documentation](https://logfire.pydantic.dev/docs/how-to-guides/query-api/)
- [Logfire SQL Reference](https://logfire.pydantic.dev/docs/reference/sql/)
