from pydantic import ValidationError

from app.modules.search.search_schema import SearchRequest


def run_test_empty_query():
    try:
        SearchRequest(project_id="proj", query="")
        print("test_empty_query: FAILED (no ValidationError)")
        return 1
    except ValidationError:
        print("test_empty_query: PASSED")
    except Exception as e:
        print(f"test_empty_query: FAILED (unexpected exception: {e})")
        return 1
    return 0


def run_test_whitespace_query():
    try:
        SearchRequest(project_id="proj", query="   \t\n ")
        print("test_whitespace_query: FAILED (no ValidationError)")
        return 1
    except ValidationError:
        print("test_whitespace_query: PASSED")
    except Exception as e:
        print(f"test_whitespace_query: FAILED (unexpected exception: {e})")
        return 1
    return 0


if __name__ == "__main__":
    rc = 0
    rc |= run_test_empty_query()
    rc |= run_test_whitespace_query()
    raise SystemExit(rc)
