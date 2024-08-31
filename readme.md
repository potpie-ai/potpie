
Sample commnd to test stream chat :
    http --stream POST 'http://localhost:8001/api/v1/conversations/{convid}/message/?user_id=defaultuser' \
    Content-Type:application/json \
    content="Hello world, let me know about Eminem"
