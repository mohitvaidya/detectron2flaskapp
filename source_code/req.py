import requests

r = requests.post("http://0.0.0.0:5000/", json={'ID':'10078'})
print(r.status_code, r.reason)
print(r.text)

# from urllib.parse import urlencode
# from urllib.request import Request, urlopen



# url = 'http://0.0.0.0:5000/' # Set destination URL here
# post_fields = {'ID': '12345'}  # Set POST fields here

# request = Request(url, urlencode(post_fields).encode())
# json = urlopen(request).read().decode()
# print(json)
# Include fps and duration on req
