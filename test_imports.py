"""Test if analytics module can be imported."""

print("Testing analytics module imports...")
print("=" * 60)

try:
    print("\n1. Testing logfire.query_client...")
    from logfire.query_client import LogfireQueryClient
    print("   ✅ LogfireQueryClient imported successfully")
except ImportError as e:
    print(f"   ❌ Failed to import LogfireQueryClient: {e}")
    print("   💡 Try: pip install httpx (required for query_client)")

try:
    print("\n2. Testing analytics schemas...")
    from app.modules.analytics.schemas import UserAnalyticsResponse
    print("   ✅ Schemas imported successfully")
except Exception as e:
    print(f"   ❌ Failed to import schemas: {e}")

try:
    print("\n3. Testing analytics service...")
    from app.modules.analytics.analytics_service import AnalyticsService
    print("   ✅ AnalyticsService imported successfully")
except Exception as e:
    print(f"   ❌ Failed to import AnalyticsService: {e}")

try:
    print("\n4. Testing analytics router...")
    from app.modules.analytics.analytics_router import router
    print("   ✅ Analytics router imported successfully")
    print(f"   📍 Router has {len(router.routes)} routes")
except Exception as e:
    print(f"   ❌ Failed to import analytics router: {e}")

try:
    print("\n5. Testing main app import...")
    from app.main import app
    print("   ✅ Main app imported successfully")
    
    # Check if analytics routes are registered
    analytics_routes = [r for r in app.routes if '/analytics' in str(r.path)]
    print(f"   📍 Found {len(analytics_routes)} analytics routes:")
    for route in analytics_routes:
        print(f"      - {route.methods} {route.path}")
    
    if not analytics_routes:
        print("   ⚠️  No analytics routes found in app!")
        
except Exception as e:
    print(f"   ❌ Failed to import main app: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Diagnostic complete!")
