import requests
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

API_HOST = os.environ.get('API_HOST')
endpoint = f"{API_HOST}/api/v1/mx";
def balance_endpoint(guid): 
        return f"{endpoint}/user/{guid}/accounts?page=1&records_per_page=10"

def transaction_endpoint(guid, startDate, endDate):
        return f"{endpoint}/user/{guid}/transactions?fromDate={startDate}&toDate={endDate}"


def get_mx_balance(guid):
    
    response = requests.get(
            balance_endpoint(guid), verify=True
    ).json()["accounts"]
    
    filteredByChecking = filter(lambda item: item['type'] == "CHECKING", response)
    result = map(lambda item: item['balance'], list(filteredByChecking))
    return list(result)[0]

def get_mx_transaction(guid, startDate, endDate):
        response = requests.get(transaction_endpoint(guid, startDate, endDate), verify=True).json()["transactions"]
        firstThreeTransactions = np.array(response)[0:3]
        return firstThreeTransactions