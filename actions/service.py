import requests


address = "https://int-api.mx.com/users/USR-8fe5e260-fe63-47dd-b8cd-438cd5a48b94/accounts?page=1&records_per_page=10"
headers = {
            "Accept": "application/vnd.mx.api.v1+json",
            "Content-Type": "application/json",
    }
userName = "653f7dba-265c-4d70-ba21-3b94dc126361"
password = "0effbf747bfe42043998c4510489cf39d39c30ed"

def get_mx_balance():
    
    response = requests.get(
            address, verify=False, headers=headers, auth=(userName, password)
    ).json()["accounts"][0]["balance"]

    return response